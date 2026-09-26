# Controller drivers (copied from the e2e controller, 2026-09-26)

One-shot shell drivers that ran on the controller to produce the paper's end-to-end numbers.
They provision a box, run, copy results back and tear down. Paths assume the controller layout
(`~/arm-bench-e2e`, `~/e2e/...`); treat them as a record of exactly what was run.

| driver | produced |
|---|---|
| `measure5.sh` | Qwen3.5-4B definitive e2e table (stock / norepack / Fable gate-v2, ppl 64 chunks, sweeps) |
| `llama_dump_fix.sh` | Llama-3.1-8B real-activation workloads + calibrated SQNR floors (`--no-warmup` matters) |
| `e2e_llama_lane.sh` | Claude Code + Fable 5.1 agent runs on the 7 Llama-3.1-8B kernels (3 lanes) |
| `measure_llama.sh` | Llama-3.1-8B e2e table, sweeps, and kernel-vs-repack comparison |
| `kvr_qwen.sh` | Qwen3.5-4B per-kernel comparison vs llama.cpp's repacked kernels (`../kernel_vs_repack/`) |
| `profile_run2.sh` | perf operator breakdown of stock vs agent builds (`../profile_ops2.sh`) |
| `quantref.sh`, `ppl64.sh` | quantization-ladder perplexity reference |
| `s9.sh`, `s7_*.sh` | measurement-noise (S9) and agent-seed (S7) variance runs |
| `kleidiai_run_v2.sh` | KleidiAI gemm_fp32 arms |
| `e2e_fable3_lane.sh`, `e2e_sol_lane.sh` | earlier Qwen gate-v2 re-runs (Fable, GPT-5.6-sol) |
