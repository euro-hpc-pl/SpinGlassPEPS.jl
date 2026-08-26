# Quick start for the two single-solve additions in 2.0.0: contraction error
# control, and an inverse-temperature ladder with warm-started boundary MPS.
#
#   julia --project=. examples/beta_ladder.jl
#
# Runs in seconds on an 18-spin king's-graph instance, so it is meant to be read
# and modified. For the same three features at the scale used by the article's
# figures, see square_50x50.jl.

using SpinGlassPEPS
using Base.ScopedValues: with
using Printf

const TOPOLOGY = (3, 3, 2)         # 3 x 3 clusters, 2 spins each => 18 spins

instance() = joinpath(pkgdir(SpinGlassPEPS), "examples", "instances", "3x3x2.txt")

potts() = potts_hamiltonian(
    ising_graph(instance());
    spectrum = full_spectrum,
    cluster_assignment_rule = super_square_lattice(TOPOLOGY),
)

function contractor(potts_h; beta, bond_dim = 16)
    m, n, _ = TOPOLOGY
    MpsContractor(
        SVDTruncate,
        PEPSNetwork{KingSingleNode{GaugesEnergy},Dense,Float64}(m, n, potts_h, rotation(0)),
        MpsParameters{Float64}(; bond_dim = bond_dim, num_sweeps = 1);
        onGPU = false, beta = beta, graduate_truncation = true,
    )
end

const SEARCH = SearchParameters(; max_states = 2^8, cutoff_prob = 1e-4)

# ---------------------------------------------------------------------------
# 1. Ask the contraction how much weight it threw away.
# ---------------------------------------------------------------------------
# `TRUNCATION_LOG` is a ScopedValue, so it applies to this task and any task
# spawned inside the block, and costs nothing when it is not installed.
function error_control(potts_h)
    println("== contraction error ==")
    for D ∈ (2, 16)
        log = TruncationLog()
        sol = with(TRUNCATION_LOG => log) do
            first(low_energy_spectrum(contractor(potts_h; beta = 2.0, bond_dim = D),
                                      SEARCH; show_progress = false))
        end
        s = truncation_stats(log)
        @printf("  D=%-3d E=%.6f  Σε=%.3e  bond-limited=%d/%d\n",
                D, first(sol.energies), s.discarded_sum, s.saturated, s.count)
    end
    println("""
      `saturated` counts bond-capped truncations whose discarded weight exceeds
      1e-12. At zero, no change from raising D is expected at the reported
      precision. Σε is a sum over truncations, so it is only readable as an
      accumulated fidelity loss while it stays well below 1; above that it means
      "do not trust this contraction".

      Note what the two rows above show: D=2 truncated at every opportunity and
      discarded a few percent of the weight, yet recovered the same energy as the
      untruncated D=16 run. Σε bounds what the contraction dropped, which is an
      upper bound on the damage, not a measurement of it.
    """)
end

# ---------------------------------------------------------------------------
# 2. Sweep beta, reusing each rung's boundary MPS to start the next.
# ---------------------------------------------------------------------------
function ladder(potts_h)
    println("== beta ladder ==")
    l = beta_ladder(contractor(potts_h; beta = 1.0), [1.0, 2.0, 4.0], SEARCH)
    for s ∈ l.steps
        @printf("  β=%-4.1f E=%.6f  Σε=%.3e  %6.3fs  %s\n", s.beta, s.energy,
                s.truncation.discarded_sum, s.wall_time, s.warm_started ? "warm" : "cold")
    end
    @printf("  selected: β=%.1f  (selection is on energy)\n", l.betas[l.selected_index])
    println("""
      Warm-started rungs report Σε == 0 because reusing a converged boundary MPS
      performs no truncating factorization -- that is a documented consequence of
      the warm start, not an accuracy claim. To audit the error of every rung,
      pass `warm_start = false`.

      Σε is also not a ranking of solution quality: it is non-monotone in β, and
      on the article's 2500-spin instances the best β carried among the *lowest*
      discarded weight. Select on energy; read Σε separately, as a statement
      about the contraction rather than about the answer.
    """)
    l
end

# ---------------------------------------------------------------------------
# 3. The ladder mutates its contractor, so drive it by hand when you want the
#    intermediate solutions for something else.
# ---------------------------------------------------------------------------
function manual(potts_h)
    println("== the same schedule, driven by hand ==")
    ctr = contractor(potts_h; beta = 1.0)
    for β ∈ (1.0, 2.0, 4.0)
        set_beta!(ctr, β)                      # warm_start = false to drop the guesses
        sol, _ = low_energy_spectrum(ctr, SEARCH; show_progress = false)
        @printf("  β=%-4.1f E=%.6f  states=%d\n", β, first(sol.energies), length(sol.energies))
    end
end

function main()
    @printf("instance: %s  (%d spins)\n\n", basename(instance()), prod(TOPOLOGY))
    potts_h = potts()
    error_control(potts_h)
    l = ladder(potts_h)
    manual(potts_h)
    @printf("\nground-state energy: %.6f\n", first(selected_solution(l).energies))
end

main()
