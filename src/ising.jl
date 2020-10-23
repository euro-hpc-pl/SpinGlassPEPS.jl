export ising_graph, energy

function energy(ig::MetaGraph, σ::Vector{<:Number})
    """
    Calculate the Ising energy as E = -sum_<i,j> s_i * J_ij * s_j - sum_j h_i * s_j.
    """

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
    return -energy
end
    
function ising_graph(instance::String, L::Int, β::Number=1)
    """
    Create a graph that represents the Ising Hamiltonian.
    """

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
    state = [rand() < 0.5 ? -1 : 1 for _ ∈ 1:L]

    set_prop!(ig, :state, state)
    set_prop!(ig, :energy, energy(ig, state)) || error("Unable to calculate the Ising energy!")

    # store extra information
    set_prop!(ig, :β, β)
    
    return ig
end