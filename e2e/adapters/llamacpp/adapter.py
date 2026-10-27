"""llama.cpp / ggml. One class; each step is a method that turns the run context and the
options into the arguments of a function from the sibling modules, which do the work."""
from __future__ import annotations

import urllib.request
from pathlib import Path

from ...context import E2EContext, Options
from ...errors import StepFailed
from ..base import FrameworkAdapter, Step

HEADER_BYTES = 64 * 1024 * 1024    # the GGUF header (metadata + tensor table) fits well inside this


def require(value, option: str, step: str):
    if value is None:
        raise StepFailed(f"the {step} step needs {option}")
    return value


class LlamaCppAdapter(FrameworkAdapter):
    name = "llamacpp"
    reference_author = "reference-scalar"          # a scalar port of ggml's own vec_dot
    baseline_author = "baseline-llamacpp-arm"      # calls ggml_mul_mat on the same blocks

    prepare_steps = (
        Step("inventory", outputs=lambda ctx: [ctx.run.inventory]),
        Step("calibrate", box=True),
    )
    evaluate_steps = (
        Step("kernel_set", outputs=lambda ctx: [ctx.run.manifest]),
        Step("deploy"),
        Step("measure", box=True, outputs=lambda ctx: [ctx.run.results]),
        Step("report"),
    )

    # -- prepare: once per model ---------------------------------------------------------

    def inventory(self, ctx: E2EContext, opts: Options) -> dict:
        """Per-tensor weight traffic of the model, from its GGUF header."""
        from .inventory import build_inventory
        head = opts.gguf_head
        if head is None:
            head = ctx.run.gguf_head
            if not Path(head).exists():
                print(f"  fetching the GGUF header of {ctx.model} -> {head}")
                self.fetch_header(ctx.model_config, Path(head))
        return build_inventory(head, ctx.run.inventory)

    # -- evaluate: after each agent run ---------------------------------------------------

    def kernel_set(self, ctx: E2EContext, opts: Options):
        """Compile an author's submitted solutions into the override manifest, then check the guards."""
        from .kernel_set import build_kernel_set
        result = build_kernel_set(
            require(opts.author, "--author", "kernel_set"), ctx.run.manifest, isa=ctx.isa,
            solutions_root=opts.root / "solutions", only=opts.only, only_types=opts.only_types,
            only_roles=opts.only_roles, cxx=opts.cxx, march=opts.march, rows_abi=opts.rows_abi,
            expect_count=opts.expect_count, reference_author=opts.reference_author,
            expect_changed=opts.expect_changed, expect_same=opts.expect_same, forbid=opts.forbid)
        if result.guard_failures:
            raise StepFailed("\n  " + "\n  ".join(result.guard_failures))
        return result

    def deploy(self, ctx: E2EContext, opts: Options) -> None:
        """llama.cpp tools tree, the model and (with agent_build) the override build, on the box `label`."""
        from .deploy import deploy
        deploy(require(opts.label, "--label", "deploy"), ctx.model, opts.agent_build)

    def measure(self, ctx: E2EContext, opts: Options) -> dict:
        """llama-bench (pp / tg) and llama-perplexity for stock / norepack / agent. Run it on the box."""
        from .measure import measure
        llama = Path(opts.llama_dir).expanduser()
        builds = {name: llama / f"build-{name}" for name in ("stock", "norepack", "agent")
                  if (llama / f"build-{name}").is_dir()}
        if "stock" not in builds:
            raise StepFailed(f"no build-stock under {llama}; run the deploy step first")
        overrides = {"agent": ctx.run.manifest} if "agent" in builds and ctx.run.manifest.exists() else {}
        model = Path("~/models").expanduser() / ctx.model_config["file"]
        return measure(model, builds, overrides=overrides, threads=opts.threads, pp=opts.pp, tg=opts.tg,
                       reps=opts.reps, fa=opts.fa, ub=opts.ub,
                       perplexity=Path(opts.perplexity) if opts.perplexity else None,
                       ppl_chunks=opts.ppl_chunks, out=ctx.run.results)

    def report(self, ctx: E2EContext, opts: Options) -> str:
        """Throughput, speedup and perplexity of the measured builds, as one table."""
        from .report import render_report
        text = render_report(ctx.run.results, opts.baseline)
        print(text)
        return text

    # -- model file ------------------------------------------------------------------------

    def fetch_header(self, model_config: dict, dest: Path) -> Path:
        url = f"https://huggingface.co/{model_config['repo']}/resolve/main/{model_config['file']}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
        with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
            out.write(response.read())
        return dest
