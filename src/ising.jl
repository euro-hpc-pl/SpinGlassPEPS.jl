export ising_graph, energy
export gibbs_tensor
export GibbsControl

struct GibbsControl 
    β::Number
    β_schedule::Vector{<:Number}
end

function gibbs_tensor(ig::MetaGraph, opts::GibbsControl)
    L = nv(ig)
    β = opts.β

    all_states = product(fill([-1, 1], L)...)
    rank = fill(2, L)

    r = exp.(-β * energy.(all_states, Ref(ig)))
    # r = [exp(-β * energy(ig, collect(σ))) for σ ∈ all_states]
    ρ = reshape(r, rank...)
    ρ / sum(ρ)
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
    ising = CSV.File(instance, types=[Int, Int, Float64])
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

    # state and corresponding energy
    state = 2(rand(L) .< 0.5) .- 1

    set_prop!(ig, :state, state)
    set_prop!(ig, :energy, energy(state, ig)) || error("Unable to calculate the Ising energy!")

    # store extra information
    set_prop!(ig, :β, β)
    
    ig
end