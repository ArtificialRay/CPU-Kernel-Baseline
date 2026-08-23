# mcp_app — MCP server for kernel-optimization sessions

Exposes `compile`/`evaluate`/`disassemble`/`submit` as an MCP server running
in-process on the target instance, so an external agent harness (nanobot,
etc.) can drive kernel-optimization sessions. Zero coupling to `eval/` or
`skills/` — never provisions instances, never imports either.

## 1. Components

```
mcp_app/
    agent_tools/
        base.py                 # KernelSession ABC — compile/evaluate/disassemble/submit.
                                 #   Multi-definition: self._definitions keyed by definition
                                 #   name, nothing evicted for the process's life.
        baseline_readiness.py   # per-definition baseline check-then-collect + the
                                 #   dataset -> baseline_author default table; called lazily
                                 #   from KernelSession.compile() on first touch
        ncnn.py, simd_loop.py, llama_cpp.py   # per-dataset KernelSession subclasses
        registry.py              # resolve_tools(dataset) -> Type[KernelSession]
        ops.py                    # compile_kernel/evaluate_kernel/disassemble_so
        isa.py                     # march_for_isa(isa) + verify_isa_available(isa)
        trajectory.py                # TrajectoryWriter — per-definition audit trail
    session.py                # SessionConfig + build_tools() — server-side bootstrap;
                               #   also eagerly writes every definition's
                               #   reference-scalar-kernel.cpp at startup
    server.py                  # the MCP server process itself (--transport stdio|streamable-http)
    resources.py                 # MCP Resources over a session's run_dir —
                                 #   nested one dir per definition
    smoke_test_driver.py            # sequential, non-nanobot smoke-test/verification driver
    scripts/
        _local_ssh.py                # smoke_test_driver.py's own SSH/rsync
        test_mcp_client.py            # plain MCP client for manual/smoke-test runs
```

## 2. Starting the server

```bash
# On the target instance, once the repo is synced there.
python -m mcp_app.server --dataset <ncnn|simd-loop|llama.cpp> --author <tag> \
    --isa <neon|sve|sve2|sme2> --run-dir ~/arm-bench/agent-runs-mcp/<author> \
    --transport stdio
```

One process serves **every** definition in `--dataset` — `compile()` takes
`definition` as a per-call argument, not a startup flag. `--baseline-author`
is optional (auto-derived from `--dataset`).

This mcp application provides two transport options:
- streamable-http: run the mcp at a remote instance, e.g. AWS gravision
- stdio: run the mcp at the local

## 3. Starting the server before agent harness

Agent harness spawns one `mcp_app.server`. `skills/launch/launch_session.py` is the
harness-agnostic script that provisions/reaches an instance, starts a
persistent `mcp_app.server` on it in streamable-http mode reached over an
SSH local-port-forward (not exposed publicly — the compile/evaluate tool
surface is effectively remote code execution), and prints the endpoint for
that harness's MCP config :

```bash
python3 skills/launch/launch_session.py launch \
    --isa <sve|sve2> --dataset <ncnn|simd-loop|llama.cpp> \
    --local-repo-dir <path to your local CPU-Kernel-Baseline checkout>
```

See `skills/README.md` for the full command surface (`provision` /
`prepare-session` / `sync-results` / `status` / `teardown`) and each
harness's own skill `README.md` (e.g.
`skills/nanobot/nanobot-kernel-session/README.md`) for wiring the printed
endpoint into that harness's own config.