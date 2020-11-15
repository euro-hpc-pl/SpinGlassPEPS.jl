export ising_graph, energy
export gibbs_tensor
export GibbsControl
export brute_force
export brute_force_lazy

struct GibbsControl 
    β::Number
    β_schedule::Vector{<:Number}
end

function brute_force_lazy(ig::MetaGraph, k::Int=1)
    L = nv(ig)
    states = product(fill([-1, 1], L)...)
    energies = vec(energy.(states, Ref(ig)))
    perm = sortperm(energies)
    collect.(states)[perm][1:k], energies[perm][1:k]
end    

function brute_force(ig::MetaGraph, k::Int=1)
    L = nv(ig)
    states = ising.(digits.(0:2^L-1, base=2, pad=L))
    energies = energy.(states, Ref(ig))
    perm = sortperm(energies)
    states[perm][1:k], energies[perm][1:k]
end  

function gibbs_tensor(ig::MetaGraph, opts::GibbsControl)
    L = nv(ig)
    β = opts.β
    states = product(fill([-1, 1], L)...)
    ρ = exp.(-β .* energy.(states, Ref(ig)))
    ρ ./ sum(ρ)
end


"""
Calculate the Ising energy as E = -sum_<i,j> s_i * J_ij * s_j - sum_j h_i * s_j.
"""
function energy(σ::Union{Vector, NTuple}, ig::MetaGraph)

    energy = 0
    # quadratic
    for edge ∈ edges(ig)
        i, j = src(edge), dst(edge)         
        J = get_prop(ig, i, j, :J) 
        energy += σ[i] * J * σ[j]   
    end 

    # linear
    for i ∈ vertices(ig)
        h = get_prop(ig, i, :h)  
        energy += h * σ[i]
    end    
    -energy
end
    
"""
Create a graph that represents the Ising Hamiltonian.
"""
function ising_graph(instance::String, L::Int, β::Number=1)

    # load the Ising instance
    ising = CSV.File(instance, types=[Int, Int, Float64], comment = "#")
    ig = MetaGraph(L, 0.0)

    set_prop!(ig, :description, "The Ising model.")

    # setup the model (J_ij, h_i)
    for row ∈ ising 
        i, j, v = row
        if i == j
            set_prop!(ig, i, :h, v) || error("Node $i missing!")
        else
            add_edge!(ig, i, j) && 
            set_prop!(ig, i, j, :J, v) || error("Cannot add Egde ($i, $j)") 
        end    
    end   

    # by default h should be zero
    for i ∈ 1:nv(ig)
        if !has_prop(ig, i, :h) 
            set_prop!(ig, i, :h, 0.) || error("Cannot set bias at node $(i).")
        end 
    end
    
    # state and corresponding energy
    state = 2(rand(L) .< 0.5) .- 1

    set_prop!(ig, :state, state)
    set_prop!(ig, :energy, energy(state, ig)) || error("Unable to calculate the Ising energy!")

    # store extra information
    set_prop!(ig, :β, β)
    
    ig
end

function unique_neighbors(ig::MetaGraph, i::Int)
    nbrs = neighbors(ig::MetaGraph, i::Int)
    filter(j -> j > i, nbrs)
end