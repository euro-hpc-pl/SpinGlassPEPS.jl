export ising_graph, energy
export gibbs_tensor, brute_force
export State

const State = Union{Vector, NTuple}
const Instance = Union{String, Dict}
const EdgeIter = LightGraphs.SimpleGraphs.SimpleEdgeIter

"""
$(TYPEDSIGNATURES)

Return the low energy spectrum

# Details

Calculates \$k\$ lowest energy states 
together with the coresponding energies 
of a classical Ising Hamiltonian
"""

function brute_force(ig::MetaGraph, k::Int=1)
    states = all_states(get_prop(ig, :rank))
    energies = vec(energy.(states, Ref(ig)))
    perm = partialsortperm(energies, 1:k) 
    collect.(states)[perm], energies[perm]
end    

_ising(σ::State) = 2 .* σ .- 1

function _brute_force(ig::MetaGraph, k::Int=1)
    L = nv(ig)
    states = _ising.(digits.(0:2^L-1, base=2, pad=L))
    energies = energy.(states, Ref(ig))
    perm = partialsortperm(energies, 1:k) 
    states[perm], energies[perm]
end  


"""
$(TYPEDSIGNATURES)

Calculates Gibbs state of a classical Ising Hamiltonian

# Details

Calculates matrix elements (probabilities) of \$\\rho\$ 
```math
\$\\bra{\\σ}\\rho\\ket{\\sigma}\$
```
for all possible configurations \$\\σ\$.
"""
function gibbs_tensor(ig::MetaGraph)
    β = get_prop(ig, :β)
    rank = get_prop(ig, :rank)
    ρ = exp.(-β .* energy.(all_states(rank), Ref(ig)))
    ρ ./ sum(ρ)
end


function energy(σ::State, ig::MetaGraph, vertices, sgn::Float64=-1.0)
    energy::Float64 = 0
    for i ∈ vertices
        h = get_prop(ig, i, :h)  
        energy += h * σ[i]
    end
    sgn * energy   
end

function energy(σ::State, ig::MetaGraph, edges::EdgeIter, sgn::Float64=-1.0)
    energy::Float64 = 0
    for e ∈ edges
        i, j = src(e), dst(e)         
        J = get_prop(ig, e, :J) 
        energy += σ[i] * J * σ[j]   
    end 
    sgn * energy
end

"""
$(TYPEDSIGNATURES)

Calculate the Ising energy 
```math
E = -\\sum_<i,j> s_i J_{ij} * s_j - \\sum_j h_i s_j.
```
"""
function energy(σ::State, ig::MetaGraph, sgn::Float64=-1.0)
    energy = energy(σ, ig, edges(ig), sgn) 
    energy += energy(σ, ig, vertices(ig), sgn)
end
    
"""
$(TYPEDSIGNATURES)

Create the Ising spin glass model.

# Details

Store extra information
"""
function ising_graph(instance::Instance, L::Int, β::Number=1, sgn::Number=1)

    # load the Ising instance
    if typeof(instance) == String
        ising = CSV.File(instance, types = [Int, Int, Float64], header=0, comment = "#")
    else
        ising = [ (i, j, J) for ((i, j), J) ∈ instance ] 
    end

    ig = MetaGraph(L, 0.0)
    set_prop!(ig, :description, "The Ising model.")

    # setup the model (J_ij, h_i)
    for (i, j, v) ∈ ising 
        if i == j
            set_prop!(ig, i, :h, sgn * v) || error("Node $i missing!")
        else
            add_edge!(ig, i, j) && 
            set_prop!(ig, i, j, :J, sgn * v) || error("Cannot add Egde ($i, $j)") 
        end    
    end   

    # by default h should be zero
    for i ∈ 1:nv(ig)
        if !has_prop(ig, i, :h) 
            set_prop!(ig, i, :h, 0.) || error("Cannot set bias at node $(i).")
        end 
    end
    
    # state (random by default) and corresponding energy
    state = 2(rand(L) .< 0.5) .- 1

    set_prop!(ig, :state, state)
    #set_prop!(ig, :energy, energy(state, ig)) || error("Unable to calculate the Ising energy!")

    # store extra information 
    set_prop!(ig, :β, β)
    set_prop!(ig, :rank, fill(2, L))

    ig
end

"""
$(TYPEDSIGNATURES)

Calculate unique neighbors of node \$i\$

# Details

This is equivalent of taking the upper 
diagonal of the adjacency matrix
"""
function unique_neighbors(ig::MetaGraph, i::Int)
    nbrs = neighbors(ig::MetaGraph, i::Int)
    filter(j -> j > i, nbrs)
end