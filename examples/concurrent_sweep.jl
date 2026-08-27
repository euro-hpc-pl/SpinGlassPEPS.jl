# Concurrent sweep over the lattice transformations, with a device-memory
# governor and contraction-error reporting.
#
# Compare with ising_model_on_a_kings_graph.jl, which writes the same protocol as
# a serial loop. Run with more than one Julia thread for the solves to overlap:
#
#   julia --project=. -t auto examples/concurrent_sweep.jl

using SpinGlassPEPS
using CUDA

function get_instance(topology::NTuple{3,Int})
    m, n, t = topology
    joinpath(pkgdir(SpinGlassPEPS), "examples", "instances", "$(m)x$(n)x$(t).txt")
end

function run_sweep(::Type{T}; topology::NTuple{3,Int}) where {T}
    m, n, _ = topology
    onGPU = CUDA.functional()

    potts_h = potts_hamiltonian(
        ising_graph(get_instance(topology));
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice(topology),
    )

    params = MpsParameters{T}(; bond_dim = 16, num_sweeps = 1)
    search_params = SearchParameters(; max_states = 2^8, cutoff_prob = 1e-4)

    # Called once per transformation, inside the task that runs it, so every
    # solve gets its own network, projector workspace and contraction cache.
    build = transform -> MpsContractor(
        SVDTruncate,
        PEPSNetwork{KingSingleNode{GaugesEnergy},Dense,T}(m, n, potts_h, transform),
        params;
        onGPU = onGPU,
        beta = T(2),
        graduate_truncation = true,
    )

    sweep = sweep_transformations(
        build,
        search_params;
        merge_strategy = ctr -> merge_branches(
            ctr;
            merge_prob = :none,
            droplets_encoding = SingleLayerDroplets(;
                max_energy = 10,
                min_size = 5,
                metric = :hamming,
            ),
        ),
    )

    sol = best_solution(sweep)
    r = sweep.report

    println("Best energy found: $(first(sol.energies))")
    println()
    println("Agreement across contraction orders")
    println("  consensus     : $(r.consensus)/$(length(sweep.transformations)) transformations")
    println("  energy spread : $(r.energy_spread)")
    println()
    println("Device-memory governor")
    println("  calibrated peak : $(round(r.calibrated_peak / 2^20, digits = 1)) MiB")
    println("  reservation     : $(round(r.reservation / 2^20, digits = 1)) MiB")
    println("  admitted at once: $(r.max_concurrency)")
    println("  blocked on VRAM : $(r.waits)")
    println()
    println("Per-transformation contraction error")
    for tr ∈ r.per_transform
        t = tr.truncation
        println(
            "  #$(tr.index): E = $(round(tr.energy, digits = 6))",
            "  Σε = $(round(t.discarded_sum, sigdigits = 3))",
            "  bond-limited truncations = $(t.saturated)/$(t.count)",
            "  $(round(tr.wall_time, digits = 2)) s",
        )
    end

    # `saturated == 0` on every transformation means that no bond-capped discard
    # exceeded 1e-12, so no change from raising D is expected at the reported
    # precision.
    if all(tr -> tr.truncation.saturated == 0, r.per_transform)
        println(
            "\nNo bond-capped discard exceeded 1e-12: ",
            "no change from raising D is expected at the reported precision.",
        )
    end

    sol
end

@time run_sweep(Float64; topology = (3, 3, 2))
