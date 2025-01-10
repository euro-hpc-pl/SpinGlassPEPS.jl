using SpinGlassPEPS
using CUDA
using Pkg
using Logging

# for visualisation of results, we need following packages
try
    using Colors, Images
catch e
    if isa(e, ArgumentError) || isa(e, LoadError)
        Pkg.add("Colors")
        Pkg.add("Images")
    else
        rethrow(e)
    end
end

disable_logging(LogLevel(1))

instance = "$(@__DIR__)/instances/triplepoint4-plain-ring.h5"
onGPU = CUDA.has_cuda_gpu()

GEOMETRY = SquareSingleNode
LAYOUT = GaugesEnergy
SPARSITY = Sparse
STRATEGY = SVDTruncate
GAUGE =  NoUpdate

function bench_inpaining(::Type{T}, β::Real, max_states::Integer, bond_dim::Integer) where {T}
	potts_h= potts_hamiltonian(instance, 120, 120)

	params = MpsParameters{T}(; bond_dim = bond_dim, method = :svd)
	search_params = SearchParameters(; max_states = max_states)
	net = PEPSNetwork{GEOMETRY{LAYOUT}, SPARSITY, T}(120, 120, potts_h, rotation(0))
	ctr = MpsContractor{STRATEGY, GAUGE, T}(net, params; onGPU = onGPU, beta = convert(Float64, β), graduate_truncation = true)
    droplets = SingleLayerDroplets(; max_energy = 100, min_size = 100 , metric = :hamming, mode=:RMF)
	merge_strategy = merge_branches(ctr; merge_prob = :none, droplets_encoding = droplets)

	sol, info = low_energy_spectrum(ctr, search_params, merge_strategy)
    ground = sol.energies[begin]

    println("Best energy found: $(ground)")
    sol
end

function visualize_result(sol::Solution)
    solution = sol.states[begin]
    solution = reshape(solution, (120, 120))

    sol2 = unpack_droplets(sol, 6)
    droplet_state = sol2.states[2]
    droplet_state = reshape(droplet_state, (120, 120))

    droplet = findall(solution .!= droplet_state)

    color_map = Dict(
        1 => RGB(225/255, 0.0, 100/255),
        2 => RGB(100/255, 225/255, 0.0),
        3 => RGB(0/255, 100/255, 225/255),
        4 => RGB(1.0, 1.0, 1.0)
    )

    img = zeros(RGB, 120, 120)

    for i in 1:120
        for j in 1:120
            img[j, i] = color_map[solution[i, j]]
        end
    end

    save(joinpath("$(@__DIR__)/inpaining_solution.png"), img)
    println("Solution visualisation saved to $(@__DIR__)/inpaining_solution.png")
    
    for (i,j) in Tuple.(droplet)
        img[j, i] = RGB(255/255, 255/255, 0/255)
    end

    save(joinpath("$(@__DIR__)/inpaining_droplet.png"), img)
    println("Droplet visualisation saved to $(@__DIR__)/inpaining_droplet.png")
end


sol = bench_inpaining(Float64, 6, 64, 4)
visualize_result(sol)

