# Solver benchmark harness. Run from the repo root:
#
#   julia --project=benchmark benchmark/run.jl [--set=quick|full] [--tag=NAME]
#
# Each case is solved twice in-process: run 1 includes compilation (TTFX proxy),
# run 2 is the warm measurement. Results land in benchmark/results/ as JSON,
# keyed by commit, so refactor phases can be compared:
#
#   julia --project=benchmark benchmark/compare.jl results/A.json results/B.json

using SpinGlassPEPS
using CUDA

const REPO = dirname(@__DIR__)
const ENGINE_INSTANCES = joinpath(REPO, "lib", "SpinGlassEngine", "test", "instances")

struct Case
    name::String
    instance::String
    clustering::NTuple{3,Int}
    max_cl_states::Int
    geometry::Any
    strategy::Any
    sparsity::Any
    beta::Float64
    bond_dim::Int
    num_states::Int
    cutoff_dE::Float64
end

const CASES = Dict(
    "quick" => [
        Case(
            "chimera_pathological_3_4_3",
            joinpath(ENGINE_INSTANCES, "pathological", "chim_3_4_3.txt"),
            (3, 4, 3),
            typemax(Int),
            SquareSingleNode{GaugesEnergy},
            SVDTruncate,
            Dense,
            1.0,
            16,
            2^8,
            Inf,
        ),
        Case(
            "chimera_droplets_128power",
            joinpath(ENGINE_INSTANCES, "chimera_droplets", "128power", "001.txt"),
            (4, 4, 8),
            2^8,
            SquareSingleNode{GaugesEnergy},
            Zipper,
            Dense,
            3.0,
            32,
            500,
            3.0,
        ),
    ],
    "full" => [
        Case(
            "chimera_droplets_2048power",
            joinpath(ENGINE_INSTANCES, "chimera_droplets", "2048power", "001.txt"),
            (16, 16, 8),
            2^8,
            SquareSingleNode{GaugesEnergy},
            Zipper,
            Sparse,
            3.0,
            32,
            500,
            3.0,
        ),
    ],
)

json(x::Number) = isfinite(x) ? repr(x) : "\"$(x)\""
json(x::Bool) = x ? "true" : "false"
json(x::AbstractString) = repr(String(x))
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(x::AbstractDict) =
    "{" * join(("$(repr(String(k))):$(json(v))" for (k, v) in x), ",") * "}"

function solve_once(case::Case, onGPU::Bool)
    m, n, t = case.clustering
    t_potts = @elapsed potts_h = potts_hamiltonian(
        ising_graph(case.instance),
        case.max_cl_states,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(;
        bond_dim = case.bond_dim,
        var_tol = 1E-8,
        num_sweeps = 4,
        tol_SVD = 1E-16,
    )
    δp = isfinite(case.cutoff_dE) ? exp(-case.beta * case.cutoff_dE) : 0.0
    search_params = SearchParameters(; max_states = case.num_states, cutoff_prob = δp)
    t_net = @elapsed begin
        net = PEPSNetwork{case.geometry,case.sparsity,Float64}(
            m,
            n,
            potts_h,
            rotation(0),
        )
        ctr = MpsContractor{case.strategy,NoUpdate,Float64}(
            net,
            params;
            onGPU = onGPU,
            beta = case.beta,
            graduate_truncation = true,
        )
    end
    stats = @timed low_energy_spectrum(ctr, search_params)
    sol, _ = stats.value
    clear_memoize_cache()
    Dict(
        "t_potts" => t_potts,
        "t_network" => t_net,
        "t_solve" => stats.time,
        "alloc_gib" => stats.bytes / 2^30,
        "gc_time" => stats.gctime,
        "best_energy" => first(sol.energies),
        "n_states" => length(sol.energies),
    )
end

function main()
    set = "quick"
    tag = ""
    for a in ARGS
        startswith(a, "--set=") && (set = split(a, "=")[2])
        startswith(a, "--tag=") && (tag = split(a, "=")[2])
    end
    cases = set == "full" ? vcat(CASES["quick"], CASES["full"]) : CASES[set]
    onGPU = CUDA.functional()
    commit = strip(read(`git -C $REPO rev-parse --short HEAD`, String))

    results = Dict{String,Any}(
        "commit" => commit,
        "julia" => string(VERSION),
        "device" => onGPU ? name(CUDA.device()) : "CPU",
        "threads" => Threads.nthreads(),
        "cases" => Dict{String,Any}(),
    )
    for case in cases
        @info "benchmark" case.name
        cold = solve_once(case, onGPU)
        onGPU && CUDA.reclaim()
        warm = solve_once(case, onGPU)
        onGPU && CUDA.reclaim()
        results["cases"][case.name] = Dict("cold" => cold, "warm" => warm)
        @info "done" cold_s = round(cold["t_solve"], digits = 2) warm_s =
            round(warm["t_solve"], digits = 2) energy = warm["best_energy"]
    end

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    stamp = replace(string(round(time())), r"\.0$" => "")
    file = joinpath(outdir, "$(stamp)-$(commit)$(isempty(tag) ? "" : "-" * tag).json")
    open(file, "w") do io
        write(io, json(results))
    end
    @info "results written" file
end

main()
