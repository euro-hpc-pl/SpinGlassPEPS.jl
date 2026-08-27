# Exhaustive search

`SpinGlassPEPS.jl` includes CPU and CUDA brute-force routines for small Ising models. The
GPU routines enumerate every state, so their memory and runtime requirements grow
exponentially with the number of spins.

Use `brute_force` for the CPU implementation. When `CUDA.functional()` is true,
`exhaustive_search` returns the full energy ordering and `exhaustive_search_bucket`
retains only the requested number of low-energy states.

## References

1. Jałowiecki, K., Rams, M. M., & Gardas, B. (2021). *Brute-forcing spin-glass
   problems with CUDA*. Computer Physics Communications, 260, 107728.
2. Tao, M. et al. (2020). *A Work-Time Optimal Parallel Exhaustive Search Algorithm for
   the QUBO and the Ising model, with GPU implementation*. IPDPSW.
3. Cook, C. et al. (2018). *GPU based parallel Ising computing for combinatorial
   optimization problems in VLSI physical design*. arXiv:1807.10750.
