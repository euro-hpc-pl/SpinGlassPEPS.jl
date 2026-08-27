# ============================================================================
# known_failures.jl -- probability-vs-Boltzmann mismatch inventory
# ============================================================================
#
# PROVENANCE
#   Distilled on 2026-07-12 from three untracked failure-counting harnesses the
#   authors kept in the pre-monorepo package repo SpinGlassEngine.jl
#   (mwe.jl, mwe2.jl, mwe3.jl).  Those scripts swept the solver configuration
#   matrix (Strategy x Sparsity x Layout x lattice transform x lattice/instance)
#   and counted configs for which the normalized branch-and-bound probabilities
#   exp.(sol.probabilities .- sol.probabilities[1]) did NOT reproduce the
#   Boltzmann weights exp.(-beta .* (sol.energies .- sol.energies[1])), plus
#   (mwe2) decoded-state energy consistency and (mwe3) SquareDoubleNode-vs-
#   SquareSingleNode spectrum/overlap agreement.
#
# RE-RUN RESULT (monorepo, this file's baseline)
#   The three harnesses were adapted to the monorepo (instances resolved to
#   test/engine/instances/..., julia --project=<repository root>,
#   onGPU = true) and instrumented to RECORD every failing
#   (Strategy, Sparsity, Layout, transform, Lattice, instance, check) tuple
#   together with observed vs expected values.
#
#   Environment: Julia 1.12.3, NVIDIA RTX 5080 (CUDA functional), monorepo
#   SpinGlassPEPS v1.5.0 with SpinGlassEngine v1.6.0, SpinGlassNetworks v1.4.0,
#   SpinGlassTensors v1.3.0, SpinGlassExhaustive v1.1.0.
#
#   ZERO FAILURES REPRODUCED.  Every configuration in every harness passed on
#   every repetition.  A negative control (comparing the same solver output
#   against Boltzmann weights at a deliberately wrong beta) confirmed the check
#   machinery does trigger on genuine mismatches, and the true-beta agreement
#   held to ~3e-16 -- i.e. the checks are live, the properties genuinely hold.
#
# FULL FAILURE INVENTORY (as re-measured 2026-07-12)
#
#   | harness | instance                            | network(s)                            | strategies          | sparsity      | layouts                                 | transforms | checks                                                              | repeats | failures |
#   |---------|-------------------------------------|---------------------------------------|---------------------|---------------|-----------------------------------------|------------|---------------------------------------------------------------------|---------|----------|
#   | mwe.jl  | pathological/cross_2_4_mdd.txt      | KingSingleNode                        | Zipper, SVDTruncate | Dense, Sparse | GaugesEnergy, EngGaugesEng, EnergyGauges | all 8      | prob ~ Boltzmann (beta=3.0, 128 states)                             | 5       | 0        |
#   | mwe2.jl | pathological/cross_3_2.txt          | KingSingleNode                        | SVDTruncate, Zipper | Sparse, Dense | EnergyGauges, GaugesEnergy, EngGaugesEng | all 8      | E ~ E_ising, E ~ E_potts, prob ~ Boltzmann (beta=1.0, 22 states)    | 2       | 0        |
#   | mwe3.jl | pegasus_nondiag/3x2x1.txt           | SquareDoubleNode vs SquareSingleNode  | SVDTruncate, Zipper | Dense, Sparse | EnergyGauges, GaugesEnergy               | all 8      | E_dbl ~ E_sgl, prob ~ Boltzmann (both), mps/mps_top overlap ~ 1     | 1       | 0        |
#   | mwe3.jl | pathological/pegasus_nd_3_4_1.txt   | SquareDoubleNode vs SquareSingleNode  | SVDTruncate, Zipper | Dense, Sparse | EnergyGauges, GaugesEnergy               | all 8      | E_dbl ~ E_sgl, prob ~ Boltzmann (both), mps/mps_top overlap ~ 1     | 1       | 0        |
#
#   Total: 320 configuration runs per full sweep (96 + 96 + 64 + 64),
#   0 recorded failures across all sweeps and repeats.
#
# CONSEQUENCE FOR THIS FILE
#   There is nothing to mark @test_broken: an @test_broken on a passing
#   property would itself fail the suite.  Instead this file pins the
#   historically-suspect properties with plain @test on one representative
#   configuration per harness family (covering every distinct failure axis the
#   authors swept: both strategies, both sparsities, each layout family, a
#   non-identity transform, both single- and double-node lattices).  If the
#   historical probability-vs-Boltzmann bug ever resurfaces, these tests catch
#   it.  Should any of them start failing, move the failing assertion to
#   @test_broken and record the tuple in the inventory table above.
#
# NOTE: not wired into runtests.jl on purpose -- the orchestrator decides.
# ============================================================================

using SpinGlassPEPS.SpinGlassEngine
using SpinGlassPEPS.SpinGlassNetworks
using SpinGlassPEPS.SpinGlassTensors
using Test
using Logging

disable_logging(LogLevel(1))

# Respect runtests.jl's onGPU if included from there; default to GPU standalone.
@isdefined(onGPU) || (onGPU = true)

@testset verbose = true "known failures distilled from authors' mwe scripts (all pass as of 2026-07-12)" begin

    # ------------------------------------------------------------------
    # Family 1 (authors' mwe.jl): KingSingleNode on cross_2_4_mdd, beta = 3.0.
    # Historically the prob-vs-Boltzmann assertion was the one being counted.
    # Representative axes: Zipper/Dense/GaugesEnergy and SVDTruncate/Sparse/
    # EnergyGauges, identity + rotation transform.
    # ------------------------------------------------------------------
    @testset "mwe1 family: prob ~ Boltzmann, KingSingleNode, cross_2_4_mdd" begin
        m, n, t = 2, 4, 3
        β = 3.0
        instance = joinpath(@__DIR__, "instances", "pathological", "cross_2_4_mdd.txt")
        ig = ising_graph(instance)
        potts_h = potts_hamiltonian(
            ig,
            spectrum = full_spectrum,
            cluster_assignment_rule = super_square_lattice((m, n, t)),
        )
        params = MpsParameters{Float64}(; bond_dim = 16, var_tol = 1E-8, num_sweeps = 4)
        search_params = SearchParameters(; max_states = 128, cutoff_prob = 0.0)

        for (Strategy, Sparsity, Layout, tidx) in (
            (Zipper, Dense, GaugesEnergy, 1),
            (Zipper, Sparse, EngGaugesEng, 6),
            (SVDTruncate, Sparse, EnergyGauges, 4),
        )
            transform = all_lattice_transformations[tidx]
            net = PEPSNetwork{KingSingleNode{Layout},Sparsity,Float64}(m, n, potts_h, transform)
            ctr = MpsContractor{Strategy,NoUpdate,Float64}(
                net,
                params;
                onGPU = onGPU,
                beta = β,
                graduate_truncation = true,
            )
            sol, s = low_energy_spectrum(ctr, search_params)
            norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
            exct_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
            @test norm_prob ≈ exct_prob
            clear_memoize_cache()
        end
    end

    # ------------------------------------------------------------------
    # Family 2 (authors' mwe2.jl): KingSingleNode on cross_3_2, beta = 1.0,
    # with decoded-state energy consistency in addition to the prob check.
    # ------------------------------------------------------------------
    @testset "mwe2 family: energies + prob ~ Boltzmann, KingSingleNode, cross_3_2" begin
        m, n, t = 2, 3, 1
        β = 1.0
        instance = joinpath(@__DIR__, "instances", "pathological", "cross_3_2.txt")
        ig = ising_graph(instance)
        potts_h = potts_hamiltonian(
            ig,
            spectrum = full_spectrum,
            cluster_assignment_rule = super_square_lattice((m, n, t)),
        )
        params = MpsParameters{Float64}(; bond_dim = 16, var_tol = 1E-8, num_sweeps = 4)
        search_params = SearchParameters(; max_states = 22, cutoff_prob = 0.0)

        for (Strategy, Sparsity, Layout, tidx) in (
            (SVDTruncate, Dense, GaugesEnergy, 2),
            (Zipper, Sparse, EnergyGauges, 7),
        )
            transform = all_lattice_transformations[tidx]
            net = PEPSNetwork{KingSingleNode{Layout},Sparsity,Float64}(m, n, potts_h, transform)
            ctr = MpsContractor{Strategy,NoUpdate,Float64}(
                net,
                params;
                onGPU = onGPU,
                beta = β,
                graduate_truncation = true,
            )
            sol, s = low_energy_spectrum(ctr, search_params)

            ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
            @test sol.energies ≈ energy.(Ref(ig), ig_states)

            potts_h_states = decode_state.(Ref(net), sol.states)
            @test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

            norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
            @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))
            clear_memoize_cache()
        end
    end

    # ------------------------------------------------------------------
    # Family 3 (authors' mwe3.jl): SquareDoubleNode (pegasus_lattice) vs
    # SquareSingleNode (super_square_lattice) on the non-diagonal pegasus
    # instance, beta = 2.0: spectra of the two encodings must agree, both must
    # reproduce Boltzmann weights, and row MPS/MPS-top overlaps must be ~ 1.
    # (The smaller 3x2x1 instance is used here to keep the sentinel fast; the
    # harness also swept pathological/pegasus_nd_3_4_1.txt with 0 failures.)
    # ------------------------------------------------------------------
    @testset "mwe3 family: SquareDoubleNode vs SquareSingleNode, pegasus_nondiag 3x2x1" begin
        m, n, t = 3, 2, 1
        β = 2.0
        num_states = 512
        instance = joinpath(@__DIR__, "instances", "pegasus_nondiag", "3x2x1.txt")
        ig = ising_graph(instance)
        potts_h = potts_hamiltonian(
            ig,
            spectrum = full_spectrum,
            cluster_assignment_rule = pegasus_lattice((m, n, t)),
        )
        potts_h2 = potts_hamiltonian(
            ig,
            spectrum = full_spectrum,
            cluster_assignment_rule = super_square_lattice((m, n, 8)),
        )
        params = MpsParameters{Float64}(; bond_dim = 16, var_tol = 1E-8, num_sweeps = 4)
        search_params = SearchParameters(; max_states = num_states, cutoff_prob = 1e-10)

        for (Strategy, Sparsity, Layout, tidx) in (
            (SVDTruncate, Dense, EnergyGauges, 1),
            (Zipper, Sparse, GaugesEnergy, 5),
        )
            tran = all_lattice_transformations[tidx]
            net = PEPSNetwork{SquareDoubleNode{Layout},Sparsity,Float64}(m, n, potts_h, tran)
            net2 = PEPSNetwork{SquareSingleNode{Layout},Sparsity,Float64}(m, n, potts_h2, tran)
            ctr = MpsContractor{Strategy,NoUpdate,Float64}(
                net,
                params;
                onGPU = onGPU,
                beta = β,
                graduate_truncation = true,
            )
            ctr2 = MpsContractor{Strategy,NoUpdate,Float64}(
                net2,
                params;
                onGPU = onGPU,
                beta = β,
                graduate_truncation = true,
            )
            sol, s = low_energy_spectrum(ctr, search_params)
            sol2, s2 = low_energy_spectrum(ctr2, search_params)

            ncmp = min(div(num_states, 8), length(sol.energies), length(sol2.energies))
            @test sol.energies[1:ncmp] ≈ sol2.energies[1:ncmp]

            norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
            @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

            norm_prob2 = exp.(sol2.probabilities .- sol2.probabilities[1])
            @test norm_prob2 ≈ exp.(-β .* (sol2.energies .- sol2.energies[1]))

            for ii ∈ 1:ctr.peps.nrows+1
                ψ1, ψ2 = mps(ctr, ii), mps(ctr2, ii)
                @test ψ1 * ψ2 / sqrt((ψ1 * ψ1) * (ψ2 * ψ2)) ≈ 1.0
            end
            for ii ∈ 0:ctr.peps.nrows
                ψ1t, ψ2t = mps_top(ctr, ii), mps_top(ctr2, ii)
                @test ψ1t * ψ2t / sqrt((ψ1t * ψ1t) * (ψ2t * ψ2t)) ≈ 1.0
            end
            clear_memoize_cache()
        end
    end
end
