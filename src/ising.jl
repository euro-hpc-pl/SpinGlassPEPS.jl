export ising_graph, update_cells!
export energy, gibbs_tensor

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
function gibbs_tensor(ig::MetaGraph, β=Float64=1.0)
    states = collect.(all_states(rank_vec(ig)))
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
energy(σ::Vector, ig::MetaGraph) = energy(σ, get_prop(ig, :J)) + energy(σ, get_prop(ig, :h))

function energy(fg::MetaDiGraph, edge::Edge)
    v, w = edge.tag
    vSp = get_prop(fg, v, :spectrum).states
    wSp = get_prop(fg, w, :spectrum).states

    m = prod(size(vSp))
    n = prod(size(wSp))

    en = zeros(m, n)
    for (j, η) ∈ enumerate(vec(wSp))
        en[:, j] = energy.(vec(vSp), Ref(edge.J), Ref(η))
    end
    #=
    for (j, η) ∈ enumerate(vec(wSp))
        for (i, σ) ∈ enumerate(vec(vSp))
            en[i, j] = energy(σ, Ref(edge.J), η)
        end
    end
    =#
    en
end

"""
$(TYPEDSIGNATURES)

Create the Ising spin glass model.

# Details

Store extra information
"""
function ising_graph(
    instance::Instance,
    L::Int,
    sgn::Number=1.0,
    rank_override::Dict{Int, Int}=Dict{Int, Int}()
)

    # load the Ising instance
    if typeof(instance) == String
        ising = CSV.File(instance, types = [Int, Int, Float64], header=0, comment = "#")
    else
        ising = [ (i, j, J) for ((i, j), J) ∈ instance ]
    end

    ig = MetaGraph(L, 0.0)
    set_prop!(ig, :description, "The Ising model.")
    set_prop!(ig, :L, L)

    for v ∈ 1:L
        set_prop!(ig, v, :active, false)
        set_prop!(ig, v, :cell, v)
        set_prop!(ig, v, :h, 0.)
    end

    J = zeros(L, L)
    h = zeros(L)

    #r
    # setup the model (J_ij, h_i)
    for (i, j, v) ∈ ising
        v *= sgn

        if i == j
            set_prop!(ig, i, :h, v) || error("Node $i missing!")
            h[i] = v
        else
            if has_edge(ig, j, i)
                error("Cannot add ($i, $j) as ($j, $i) already exists!")
            end
            add_edge!(ig, i, j) &&
            set_prop!(ig, i, j, :J, v) || error("Cannot add Egde ($i, $j)")
            J[i, j] = v
        end

        set_prop!(ig, i, :active, true) || error("Cannot activate node $(i)!")
        set_prop!(ig, j, :active, true) || error("Cannot activate node $(j)!")
    end

    # store extra information
    rank = Dict{Int, Int}()
    for v in vertices(ig)
        if get_prop(ig, v, :active)
            rank[v] = get(rank_override, v, 2)
        end
    end
    
    set_prop!(ig, :rank, rank)

    set_prop!(ig, :J, J)
    set_prop!(ig, :h, h)

    σ = 2.0 * (rand(L) .< 0.5) .- 1.0

    set_prop!(ig, :state, σ)
    set_prop!(ig, :energy, energy(σ, ig))
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


function update_cells!(ig::MetaGraph; rule::Dict)
    for v ∈ vertices(ig)
         w = get_prop(ig, v, :cell)
        set_prop!(ig, v, :cell, rule[w])
    end
end
