export ising_graph, energy
export gibbs_tensor
export State

const State = Union{Vector, NTuple}
const Instance = Union{String, Dict}


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
    states = collect.(all_states(rank))
    ρ = exp.(-β .* energy.(states, Ref(ig)))
    ρ ./ sum(ρ)
end

"""
$(TYPEDSIGNATURES)

Calculate the Ising energy 
```math
E = -\\sum_<i,j> s_i J_{ij} * s_j - \\sum_j h_i s_j.
```
"""

energy(σ::Vector, J::Matrix, η::Vector=σ) = dot(σ, J, η)
energy(σ::Vector, h::Vector) = dot(h, σ)
energy(σ::Vector, cl::Cluster, η::Vector=σ) = energy(σ, cl.J, η) + energy(cl.h, σ)

function energy(σ::Vector, ig::MetaGraph) 
    cl = Cluster(ig, 0, enum(vertices(ig)), edges(ig))
    energy(σ, cl) 
end
   
function energy(fg::MetaDiGraph, edge::Edge) 
    v, w = edge.tag
    vSp = get_prop(fg, v, :spectrum).states
    wSp = get_prop(fg, w, :spectrum).states

    m = prod(size(vSp))
    n = prod(size(wSp))

    en = zeros(m, n) 
    for (j, η) ∈ enumerate(wSp)
        en[:, j] = energy.(vec(vSp), Ref(edge.J), Ref(η)) 
    end
    en 
end

"""
$(TYPEDSIGNATURES)

Create the Ising spin glass model.

# Details

Store extra information
"""
function ising_graph(instance::Instance, L::Int, β::Number=1.0, sgn::Number=-1.0)

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
        v *= sgn
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

    # store extra information 
    set_prop!(ig, :β, β)
    set_prop!(ig, :rank, fill(2, L))

    # state (random by default) and corresponding energy
    σ = 2.0 * (rand(L) .< 0.5) .- 1.0

    set_prop!(ig, :state, σ)
    set_prop!(ig, :energy, energy(σ, ig)) || error("Unable to calculate the Ising energy!")

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
