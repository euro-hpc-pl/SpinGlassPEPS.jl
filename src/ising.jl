export ising_graph, energy
export gibbs_tensor, brute_force
export State, Cluster, Spectrum

const State = Union{Vector, NTuple}
const Instance = Union{String, Dict}
const EdgeIter = Union{LightGraphs.SimpleGraphs.SimpleEdgeIter, Base.Iterators.Filter}

struct Spectrum
    energies::Array{<:Number}
    states::Array{Vector{<:Number}}
end

mutable struct Cluster
    vertices::Dict{Int,Int}
    edges::EdgeIter
    rank::Vector
    J::Matrix{<:Number}
    h::Vector{<:Number}

    function Cluster(ig::MetaGraph, vertices::Dict, edges::EdgeIter)
        cl = new(vertices, edges)
        L = length(cl.vertices)

        cl.J = zeros(L, L)
        for e ∈ cl.edges
            i = cl.vertices[src(e)]
            j = cl.vertices[dst(e)] 
            cl.J[i, j] = get_prop(ig, e, :J)
        end

        rank = get_prop(ig, :rank)
        cl.rank = rank[1:L]

        cl.h = zeros(L)
        for (v, i) ∈ cl.vertices
            cl.h[i] = get_prop(ig, v, :h)
            cl.rank[i] = rank[v]
        end
        cl
    end
end

"""
$(TYPEDSIGNATURES)

Return the low energy spectrum

# Details

Calculates \$k\$ lowest energy states 
together with the coresponding energies 
of a classical Ising Hamiltonian
"""

function brute_force(ig::MetaGraph; num_states::Int=1)
    cl = Cluster(ig, enum(vertices(ig)), edges(ig))
    brute_force(cl, num_states=num_states)
end 

function brute_force(cl::Cluster; num_states::Int=1)
    σ = collect.(all_states(cl.rank))
    states = reshape(σ, prod(cl.rank))
    energies = energy.(states, Ref(cl))
    perm = partialsortperm(energies, 1:num_states) 
    Spectrum(energies[perm], states[perm])
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

function energy(σ::Vector, J::Matrix, η::Vector=σ; sgn::Float64=-1.0) 
    sgn * dot(σ, J, η)
end

function energy(σ::Vector, h::Vector; sgn::Float64=-1.0) 
    sgn * dot(h, σ)
end

function energy(σ::Vector, cl::Cluster, η::Vector=σ; sgn::Float64=-1.0) 
    energy(σ, cl.J, η, sgn=sgn) + energy(cl.h, σ, sgn=sgn)
end

function energy(σ::Vector, ig::MetaGraph; sgn::Float64=-1.0) 
    cl = Cluster(ig, enum(vertices(ig)), edges(ig))
    energy(σ, cl, sgn=sgn) 
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
