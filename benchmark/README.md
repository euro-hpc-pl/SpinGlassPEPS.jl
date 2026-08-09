# Benchmark harness

Every performance-relevant refactor phase must be gated on this harness: record a
baseline before the change, re-run after, and `compare.jl` flags >5% regressions in
warm solve time or allocations, and any change in the returned ground energy.

```sh
julia --project=benchmark benchmark/run.jl --set=quick --tag=baseline
julia --project=benchmark benchmark/run.jl --set=quick --tag=my-change
julia benchmark/compare.jl benchmark/results/<old>.json benchmark/results/<new>.json
```

Add `--sweep` to also time the whole eight-transformation protocol — the serial
loop the published examples spell out, versus the governed concurrent sweep — and
record what the device-memory governor decided (calibrated peak, reservation,
admission limit, waits). Requires `julia -t auto` to measure anything: with one
thread the sweep runs serially.

```sh
julia --project=benchmark -t auto benchmark/run.jl --set=quick --sweep --tag=sweep
```

- `--set=quick` — pathological chimera (3,4,3) + chimera_droplets 128power. Minutes.
- `--set=quick+` — quick cases plus sparse and King-geometry coverage.
- `--set=full` — adds chimera_droplets 2048power (Sparse, Zipper). Much longer.
- `--set=big` — runs only the largest 2048power case.

Each case runs twice in-process: the *cold* numbers include compilation (a TTFX
proxy), the *warm* numbers are the measurement. Uses the GPU when
`CUDA.functional()`, CPU otherwise; the device is recorded in the output.

Results are JSON files in `benchmark/results/`, keyed by commit hash
(`results/` is gitignored except for committed baselines).

Planned extensions (Phase 3+): peak-GPU-memory tracking, per-phase split of the
solve (boundary-MPS preprocessing vs branch-and-bound), Pegasus/Zephyr cases, and
a Float32 leg.
