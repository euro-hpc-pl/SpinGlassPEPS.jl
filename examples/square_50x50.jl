# Benchmark-scale example: a 2500-spin (50x50) square-lattice spin glass, the size
# used by the figures of the original SoftwareX article.
#
#   julia --project=. -t auto examples/square_50x50.jl
#
# The `-t auto` matters for step 3: with a single thread the transformation sweep
# runs serially and reports `admitted = 1`.
#
# It walks the three capabilities added in 2.0.0 on one instance:
#
#   1. contraction error control  -- how much weight the boundary MPS discarded
#   2. an inverse-temperature ladder with warm-started boundary MPS
#   3. the concurrent transformation sweep and its agreement diagnostics
#
# On this instance size the CPU path is the faster one, so `onGPU` is false by
# default; see the documentation for where the device begins to pay.

using SpinGlassPEPS
using CUDA
using Base.ScopedValues: with
using Printf

instance(k::Int = 1) = joinpath(
    pkgdir(SpinGlassPEPS), "benchmark", "instances", "square_50x50",
    lpad(k, 3, '0') * ".txt",
)

const M, N, T = 50, 50, 1          # 50 x 50 clusters, one spin each => 2500 spins
const ONGPU = false                # measured: the host is faster at this size

hamiltonian(k) = potts_hamiltonian(
    ising_graph(instance(k));
    spectrum = full_spectrum,
    cluster_assignment_rule = super_square_lattice((M, N, T)),
)

contractor(potts_h; beta, bond_dim = 8, transform = rotation(0)) = MpsContractor(
    Zipper,
    PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(M, N, potts_h, transform),
    MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1e-8, num_sweeps = 4);
    onGPU = ONGPU, beta = beta, graduate_truncation = true,
)

search = SearchParameters(; max_states = 256, cutoff_prob = 0.0)

# ---------------------------------------------------------------------------
# 1. How much did the contraction throw away?
# ---------------------------------------------------------------------------
function error_control(potts_h)
    println("\n== contraction error at two bond dimensions ==")
    for D ∈ (4, 32)
        log = TruncationLog()
        sol = with(TRUNCATION_LOG => log) do
            first(low_energy_spectrum(contractor(potts_h; beta = 4.0, bond_dim = D),
                                      search; show_progress = false))
        end
        s = truncation_stats(log)
        @printf("  D=%-3d E=%.6f   Σε=%.3e  bond-limited=%d/%d  kept=%d/%d\n",
                D, first(sol.energies), s.discarded_sum, s.saturated, s.count,
                s.dims_kept, s.dims_offered)
    end
    println("  A bond-limited count well below the total means D was rarely the")
    println("  binding constraint; at zero, raising it cannot change the answer.")
    println("  Note Σε is a *sum* over truncations: values approaching or exceeding 1")
    println("  mean the linearised 'accumulated fidelity loss' reading no longer holds,")
    println("  and the contraction should simply be treated as untrustworthy.")
end

# ---------------------------------------------------------------------------
# 2. Which beta? Scan one, and read the two diagnostics separately.
# ---------------------------------------------------------------------------
function beta_scan(potts_h)
    println("\n== inverse-temperature ladder (warm-started) ==")
    ladder = beta_ladder(contractor(potts_h; beta = 2.0), [2.0, 3.0, 4.0, 6.0], search)
    for s ∈ ladder.steps
        @printf("  β=%-4.1f E=%.6f  Σε=%.3e  %5.1fs  %s\n", s.beta, s.energy,
                s.truncation.discarded_sum, s.wall_time,
                s.warm_started ? "warm" : "cold")
    end
    @printf("  selected: β=%.1f\n", ladder.betas[ladder.selected_index])
    # Two things this scan illustrates. Warm-started rungs report Σε == 0 because a
    # warm start never performs a truncating factorization -- that is the documented
    # incompatibility, not an accuracy claim; audit with `warm_start = false`.
    # And Σε is not a ranking of solution quality: it is non-monotone in β, and on
    # ten instances of this family the best β carried among the *lowest* discarded
    # weight. Select on energy; read Σε as a separate statement about the
    # contraction. This single instance also disagrees with that ten-instance
    # median at β = 6, which is why one run is not evidence.
end

# ---------------------------------------------------------------------------
# 3. All eight lattice transformations, with agreement diagnostics.
# ---------------------------------------------------------------------------
function transformation_sweep(potts_h)
    println("\n== sweep over all lattice transformations ==")
    sweep = sweep_transformations(t -> contractor(potts_h; beta = 4.0, transform = t),
                                  search)
    r = sweep.report
    @printf("  best energy   : %.6f\n", first(best_solution(sweep).energies))
    @printf("  consensus     : %d/%d transformations reached it\n",
            r.consensus, length(sweep.transformations))
    @printf("  energy spread : %.3e\n", r.energy_spread)
    @printf("  admitted      : %d concurrently (%.1fs total)\n",
            r.max_concurrency, r.wall_time)
    println("  Independent contraction orders agreeing is oracle-free evidence;")
    println("  a large spread says the result should not be believed.")
end

function main()
    @printf("instance: %s  (%d spins)\n", basename(instance(1)), M * N * T)
    potts_h = hamiltonian(1)
    error_control(potts_h)
    beta_scan(potts_h)
    transformation_sweep(potts_h)
end

main()
