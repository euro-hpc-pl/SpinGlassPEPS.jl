using SpinGlassPEPS
using CUDA
using Test

# These tests cover what only the umbrella can: that the nested-module API is
# reexported consistently and that the whole stack solves a small instance.

@testset "Every reexported name is defined" begin
    for mod ∈ (SpinGlassTensors, SpinGlassNetworks, SpinGlassEngine, SpinGlassExhaustive)
        for name ∈ names(mod)
            @test isdefined(mod, name)
        end
    end
end

@testset "No ambiguous bindings across the stack" begin
    # Two packages exporting *distinct* functions under one name is a hard
    # UndefVarError for downstream users on Julia >= 1.12. Same-named exports
    # must be the same binding (method extension), not parallel definitions.
    mods = (SpinGlassTensors, SpinGlassNetworks, SpinGlassEngine, SpinGlassExhaustive)
    owners = Dict{Symbol,Vector{Any}}()
    for mod ∈ mods, name ∈ names(mod)
        name == nameof(mod) && continue
        push!(get!(owners, name, Any[]), getglobal(mod, name))
    end
    for (name, bindings) ∈ owners
        @test length(unique(objectid, bindings)) == 1
    end
end

@testset "End-to-end: smallest chimera instance solves through the umbrella" begin
    exact_energies = [-2.6, -1.1, -0.6, -0.4, -0.4, 1.1, 1.9, 2.1]

    m, n, t = 3, 1, 1
    β = 1.0
    instance = joinpath(
        @__DIR__,
        "engine",
        "instances",
        "pathological",
        "chim_$(n)_$(m)_$(t).txt",
    )

    ig = ising_graph(instance)
    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )

    params = MpsParameters{Float64}(; bond_dim = 16, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = 2^8, cutoff_prob = 0.0)

    net = PEPSNetwork{SquareSingleNode{EnergyGauges},Dense,Float64}(
        m,
        n,
        potts_h,
        rotation(0),
    )
    ctr = MpsContractor{SVDTruncate,NoUpdate,Float64}(
        net,
        params;
        onGPU = CUDA.functional(),
        beta = β,
        graduate_truncation = true,
    )
    sol, _ = low_energy_spectrum(ctr, search_params)
    @test sol.energies ≈ exact_energies
    clear_memoize_cache()
end
