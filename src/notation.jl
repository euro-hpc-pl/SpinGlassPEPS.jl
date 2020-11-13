using TensorOperations
using LightGraphs
using MetaGraphs

### the following are used to write and stopre J an h
# Interaction type are used to distinguish zero interactions
# strength and no interactions

struct Interaction{T<:AbstractFloat}
    ind::Tuple{Int, Int}
    coupling::T
    function(::Type{Interaction{T}})(ind::Tuple{Int, Int}, coupling::T1) where {T <: AbstractFloat, T1 <: AbstractFloat}
        new{T}(ind, T(coupling))
    end
    function(::Type{Interaction})(ind::Tuple{Int, Int}, coupling::Float64)
        new{Float64}(ind, coupling)
    end
end

#############    forming varous grids    #############
function nxmgrid(n::Int, m::Int)
    grid = zeros(Int, n, m)
    for i in 1:m
       for j in 1:n
           grid[j,i] = i+m*(j-1)
       end
   end
   grid
end


function grid_cel(i::Int, j::Int, block_s::Tuple{Int, Int}, size::Tuple{Int, Int})
    d1 = (i-1)*block_s[1]
    d2 = (j-1)*block_s[2]
    s1 = minimum([block_s[1], size[1]-d1])
    s2 = minimum([block_s[2], size[2]-d2])
    cel = zeros(Int, s1, s2)
    delta_i = (i-1)*block_s[1]*size[2]
    delta_j = (j-1)*block_s[2]
    for k in 1:s1
        for l in 1:s2
            cel[k,l] = delta_i+ l + delta_j + (k-1)*size[2]
        end
    end
    cel
end


function form_a_grid(block_size::Tuple{Int, Int}, size::Tuple{Int, Int})
    s1 = ceil(Int, size[1]/block_size[1])
    s2 = ceil(Int, size[2]/block_size[2])
    M = nxmgrid(s1,s2)
    grid1 = Array{Array{Int}}(undef, (s1,s2))
    for i in 1:s1
        for j in 1:s2
            grid1[i,j] = grid_cel(i,j, block_size, size)
        end
    end
    Array{Array{Int}}(grid1), M
end

function chimera_cell(i::Int, j::Int, size::Int)
    size = Int(sqrt(size/8))
    ofset = 8*(j-1)+8*size*(i-1)
    cel = zeros(Int, 4, 2)
    cel[:,1] = [k+ofset for k in 1:4]
    cel[:,2] = [k+ofset for k in 5:8]
    cel
end

# TODO this needs to be altered bor larger chimera cell
function form_a_chimera_grid(n::Int)
    problem_size = 8*n^2
    M = nxmgrid(n,n)

    grid = Array{Array{Int}}(undef, (n,n))

    for i in 1:n
        for j in 1:n
            grid[i,j] = chimera_cell(i,j, problem_size)
        end
    end
    Array{Array{Int}}(grid), M
end


#removes bonds that do not fit to the grid
# testing function
function fullM2grid!(M::Matrix{Float64}, s::Tuple{Int, Int})
    s1 = s[1]
    s2 = s[2]
    pairs = Vector{Int}[]
    for i in 1:s1*s2
        if (i%s2 > 0 && i < s1*s2)
            push!(pairs, [i, i+1])
        end
        if i <= s2*(s1-1)
            push!(pairs, [i, i+s2])
        end
    end

    for k in CartesianIndices(size(M))
        i1 = [k[1], k[2]]
        i2 = [k[2], k[1]]
        if !(i1 in pairs) && !(i2 in pairs) && (k[1] != k[2])
            M[i1...] = M[i2...] = 0.
        end
    end
end


function get_system_size(interactions::Vector{Interaction{T}}) where T <: AbstractFloat
    size = 0
    for q in interactions
        size = maximum([size, q.ind[1], q.ind[2]])
    end
    size
end


"""
    ind2spin(i::Int, s::Int = 2)

return a spin from the physical index, if no_spins is 1, returns zero.
"""
function ind2spin(i::Int, no_spins::Int = 1)
    s = [2^i for i in 1:no_spins]
    return [1-2*Int((i-1)%j < div(j,2)) for j in s]
end


function spins2ind(s::Vector{Int})
    s = [Int(el == 1) for el in s]
    v = [2^i for i in 0:1:length(s)-1]
    transpose(s)*v+1
end



function reindex(i::Int, no_spins::Int, subset_ind::Vector{Int})
    if length(subset_ind) == 0
        return 1
    end
    s = ind2spin(i, no_spins)
    spins2ind(s[subset_ind])
end


"""
    M2interactions(M::Matrix{T}) where T <:AbstractFloat

Convert the interaction matrix (must be symmetric) to the vector
of Interaction Type
"""
function M2interactions(M::Matrix{T}) where T <:AbstractFloat
    interactions = Interaction{T}[]
    s = size(M)
    for i in 1:s[1]
        for j in i:s[2]
            if (M[i,j] != 0.) | (i == j)
                x = M[i,j]
                interaction = Interaction{T}((i,j), x)
                push!(interactions, interaction)
            end
        end
    end
    interactions
end

"""
    interactions2M(ints::Vector{Interaction{T}}) where T <: AbstractFloat

inverse to M2interactions(M::Matrix{T})
"""

function interactions2M(ints::Vector{Interaction{T}}) where T <: AbstractFloat
    s = get_system_size(ints)
    Mat = zeros(T,s,s)
    for q in ints
        (i,j) = q.ind
        Mat[i,j] = Mat[j,i] = q.coupling
    end
    Mat
end

function interactions2graph(ints::Vector{Interaction{T}}) where T <: AbstractFloat
    L = get_system_size(ints)

    ig = MetaGraph(L, 0.0)

    set_prop!(ig, :description, "The Ising model.")

    # setup the model (J_ij, h_i)
    for q ∈ ints
        (i,j) = q.ind
        v = q.coupling
        if i == j
            set_prop!(ig, i, :h, v) || error("Node $i missing!")
        else
            add_edge!(ig, i, j) &&
            set_prop!(ig, i, j, :J, v) || error("Cannot add Egde ($i, $j)")
        end
    end
    ig
end

function graph4mps(ig::MetaGraph)
    for v in vertices(ig)
        h = props(ig, v)[:h]
        set_prop!(ig, v, :log_energy, [-h, h])
        set_prop!(ig, v, :spins, [v])
    end
    ig
end


"""
    getJ(interactions::Vector{Interaction{T}}, i::Int, j::Int) where T <: AbstractFloat

reades the coupling from the interactions, returns the number
"""
function getJ(interactions::Vector{Interaction{T}}, i::Int, j::Int) where T <: AbstractFloat
    try
        return filter(x->x.ind==(i,j), interactions)[1].coupling
    catch
        return filter(x->x.ind==(j,i), interactions)[1].coupling
    end
end

# TODO following are used to form a grid,
# they should be compleated and incorporated into the solver


function is_element(i::Int, j::Int, grid::Matrix{Int})
    try
        grid[i, j]
        return true
    catch
        return false
    end
end

function position_in_cluster(cluster::Vector{Int}, i::Int)
    # cluster is assumed to be unique
    return findall(x->x==i, cluster)[1]
end

function index_of_interacting_spins(all::Vector{Int}, part::Vector{Int})
    [position_in_cluster(all, i) for i in part]
end


struct Element_of_square_grid
    row::Int
    column::Int
    spins_inds::Vector{Int}
    intra_struct::Vector{Tuple{Int, Int}}
    left::Vector{Int}
    right::Vector{Int}
    up::Vector{Int}
    down::Vector{Int}

    #construction from the grid
    function(::Type{Element_of_square_grid})(i::Int, grid::Matrix{Int}, spins_inds::Matrix{Int})
        s = size(grid)
        intra_struct = Tuple{Int, Int}[]
        left = Int[]
        right = Int[]
        up = Int[]
        down = Int[]

        r = findall(x->x==i, grid)[1]
        row = r[1]
        column = r[2]

        internal_size = size(spins_inds)

        for k in 1:internal_size[1]
            for l in 2:internal_size[2]
                push!(intra_struct, (spins_inds[k,l-1], spins_inds[k,l]))
            end
        end
        for l in 1:internal_size[2]
            for k in 2:internal_size[1]
                push!(intra_struct, (spins_inds[k-1,l], spins_inds[k,l]))
            end
        end

        #TODO this representation will be changes
        spins = Vector{Int}(vec(transpose(spins_inds)))

        #if the node exist
        if column > 1
            left = index_of_interacting_spins(spins, spins_inds[:, 1])
        end
        if column < size(grid, 2)
            right = index_of_interacting_spins(spins, spins_inds[:, end])
        end
        if row > 1
            up = index_of_interacting_spins(spins, spins_inds[1, :])
        end
        if row < size(grid, 1)
            down = index_of_interacting_spins(spins, spins_inds[end,:])
        end

        new(row, column, spins, intra_struct, left, right, up, down)
    end
    function(::Type{Element_of_square_grid})(i::Int, grid::Matrix{Int})
        Element_of_square_grid(i, grid, reshape([i], (1,1)))
    end
    function(::Type{Element_of_square_grid})(i::Int, grid::Matrix{Int}, M::Array{Array{Int64,N} where N,2})
        r = findall(x->x==i, grid)[1]
        Element_of_square_grid(i, grid, M[r])
    end
end


struct Element_of_chimera_grid
    row::Int
    column::Int
    spins_inds::Vector{Int}
    intra_struct::Vector{Tuple{Int, Int}}
    left::Vector{Int}
    right::Vector{Int}
    up::Vector{Int}
    down::Vector{Int}

    #construction from the grid
    function(::Type{Element_of_chimera_grid})(i::Int, grid::Matrix{Int}, spins_inds::Matrix{Int})
        s = size(grid)
        intra_struct = Tuple{Int, Int}[]
        left = Int[]
        right = Int[]
        up = Int[]
        down = Int[]

        r = findall(x->x==i, grid)[1]
        row = r[1]
        column = r[2]

        internal_size = size(spins_inds)
        internal_size == (4,2) || error("size $(internal_size) does not fit chimera cell")

        for k1 in 1:4
            for k2 in 1:4
                push!(intra_struct, (spins_inds[k1,1], spins_inds[k2,2]))
            end
        end

        spins = Vector{Int}(vec(transpose(spins_inds)))
        if column > 1
            left = index_of_interacting_spins(spins, spins_inds[:, 2])
        end
        if column < size(grid, 2)
            right = index_of_interacting_spins(spins, spins_inds[:, 2])
        end
        if row > 1
            up = index_of_interacting_spins(spins, spins_inds[:, 1])
        end
        if row < size(grid, 1)
            down = index_of_interacting_spins(spins, spins_inds[:, 1])
        end

        new(row, column, spins, intra_struct, left, right, up, down)
    end
    function(::Type{Element_of_chimera_grid})(i::Int, grid::Matrix{Int}, M::Array{Array{Int64,N} where N,2})
        r = findall(x->x==i, grid)[1]
        Element_of_chimera_grid(i, grid, M[r])
    end
end

EE = Union{Element_of_square_grid, Element_of_chimera_grid}

function is_grid(ig::MetaGraph, g_elements)
    # TODO finish it
    for i in vertices(ig)
        a = g_elements[i]
        println(i)
        println(a.row, ",",  a.column)
        for j in all_neighbors(ig, i)
            b = g_elements[j]
            println(b.row, ",", b.column)
        end
    end
end


function interactions2grid_graph(ig::MetaGraph, ints::Vector{Interaction{T}}, cell_size::Tuple{Int, Int} = (1,1)) where T <: AbstractFloat
    L = nv(ig)

    M = zeros(1,1)
    g_elements = []
    if degree(ig, 1) == 2

        s2 = maximum(all_neighbors(ig, 1))-1
        # error will be rised if not Int
        s1 = Int(L/s2)
        #is_grid(ig, (s1, s2))
        grid, M = form_a_grid(cell_size, (s1, s2))
        g_elements = [Element_of_square_grid(i, M, grid) for i in 1:maximum(M)]
    elseif degree(ig, 1) == 5
        # error will be rised if not Int
        n = Int(sqrt(L/8))
        grid, M = form_a_chimera_grid(n)
        g_elements = [Element_of_chimera_grid(i, M, grid) for i in 1:maximum(M)]
    else
        error("degree of first node = $degree(ig, 1), neither grid nor chimera")
    end
    L1 = maximum(M)
    g = MetaGraph(L1, 0.0)

    set_prop!(g, :grid, M)

    for i in 1:L1
        g_element = g_elements[i]
        set_prop!(g, i, :row, g_element.row)
        set_prop!(g, i, :column, g_element.column)
        set_prop!(g, i, :spins, g_element.spins_inds)

        internal_e = log_internal_energy(g_element, ints)
        set_prop!(g, i, :log_energy, internal_e)

        if g_element.column < size(M, 2)
            ip = M[g_element.row, g_element.column+1]
            e = Edge(i, ip)
            add_edge!(g, e) || error("Not a grid - cannot add Egde $e")
            set_prop!(g, e, :inds, g_element.right)
        end

        if g_element.column > 1
            ip = M[g_element.row, g_element.column-1]
            e = Edge(i, ip)
            J = get_Js(g_element, g_elements[ip], ints, connection = "lr")
            M_left = M_of_interaction(g_element, J, g_element.left)
            set_prop!(g, e, :M, M_left)
        end

        if g_element.row < size(M, 1)
            ip = M[g_element.row+1, g_element.column]
            e = Edge(i, ip)
            add_edge!(g, e) || error("Not a grid - cannot add Egde $e")
            set_prop!(g, e, :inds, g_element.down)
        end

        if g_element.row > 1
            ip = M[g_element.row-1, g_element.column]
            e = Edge(i, ip)
            J = get_Js(g_element, g_elements[ip], ints, connection = "ud")
            M_up = M_of_interaction(g_element, J, g_element.up)
            set_prop!(g, e, :M, M_up)
        end
    end
    g
end

"""
    struct Node_of_grid

this structure is supposed to include all information about nodes on the grid.
necessary to crearte the corresponding tensor (e.g. the element of the peps).
"""
# TODO this for sure need to be clarified, however I would leave such
# approach. It allows for easy creation of various grids of interactions
# (e.g. Pegasusu) and its modification during computation (if necessary).




function log_internal_energy(g::EE, interactions::Vector{Interaction{T}}) where T <: AbstractFloat

    # TODO h and J this will be from a grid
    hs = [getJ(interactions, i, i) for i in g.spins_inds]
    no_spins = length(g.spins_inds)

    log_energy = zeros(T, 2^no_spins)

    for i in 1:2^no_spins
        σs = ind2spin(i, no_spins)
        e = sum(σs.*hs)
        for pair in g.intra_struct
            J = getJ(interactions, pair...)
            i1 = position_in_cluster(g.spins_inds, pair[1])
            i2 = position_in_cluster(g.spins_inds, pair[2])
            e = e + 2*σs[i1]*σs[i2]*J
        end
        log_energy[i] = e
    end
    log_energy
end

function M_of_interaction(g::EE, J::Vector{T}, spin_subset::Vector{Int}) where T <: AbstractFloat

    subset_size = length(spin_subset)
    no_spins = length(g.spins_inds)

    log_energies = zeros(T, 2^subset_size, 2^no_spins)

    for i in 1:2^no_spins
        σ_cluster = ind2spin(i, no_spins)
        for j in 1:2^subset_size
            σ = ind2spin(j, subset_size)
            log_energy = 2*sum(J.*σ.*σ_cluster[spin_subset])
            @inbounds log_energies[j,i] = log_energy
        end
    end
    log_energies
end



struct Node_of_grid
    spins_inds::Vector{Int}
    row::Int
    column::Int
    right::Vector{Int}
    down::Vector{Int}
    energy::Vector{Float64}
    energy_left::Array{Float64, 2}
    energy_up::Array{Float64, 2}

    function(::Type{Node_of_grid})(i::Int, grid::Matrix{Int}, interactions::Vector{Interaction{T}}) where T <: AbstractFloat
        s = size(grid)
        intra_struct = Vector{Int}[]

        g = Element_of_square_grid(i, grid)

        a = findall(x->x==i, grid)[1]
        j = a[1]
        k = a[2]

        el = zeros(1,2)
        eu = zeros(1,2)

        if g.column > 1
            ip = grid[j, k-1]
            left_J = [getJ(interactions, i, ip)]
            el = M_of_interaction(g, left_J, g.left)
        end

        if g.row > 1
            ip = grid[j-1, k]
            up_J = [getJ(interactions, i, ip)]
            eu = M_of_interaction(g, up_J, g.up)
        end

        h = getJ(interactions, i, i)
        log_energy = [-h, h]

        new([i], g.row, g.column, g.right, g.down, log_energy, el, eu)
    end
    #construction from matrix of matrices (a grid of many nodes)
    function(::Type{Node_of_grid})(i::Int, Mat::Matrix{Int},
                                grid::Array{Array{Int64,N} where N,2},
                                interactions::Vector{Interaction{T}};
                                chimera::Bool = false) where T <: AbstractFloat



        # TODO to below 10 lines is the working by-pass will be changed to graphs
        a = findall(x->x==i, Mat)[1]
        j = a[1]
        k = a[2]
        M = grid[j,k]


        if chimera
            g = Element_of_chimera_grid(i, Mat, M)
        else
            g = Element_of_square_grid(i, Mat, M)
        end

        el = zeros(1,2^length(g.spins_inds))
        eu = zeros(1,2^length(g.spins_inds))

        if is_element(j, k-1, Mat)
            ip = Mat[j, k-1]
            Mp = grid[j,k-1]

            g1 = typeof(g)(ip, Mat, Mp)

            left_J = get_Js(g, g1, interactions, connection = "lr")

            el = M_of_interaction(g, left_J, g.left)
        end

        if is_element(j-1, k, Mat)
            up_J = T[]
            ip = Mat[j-1, k]
            Mp = grid[j-1,k]
            g1 = typeof(g)(ip, Mat, Mp)

            up_J = get_Js(g, g1, interactions, connection = "ud")

            eu = M_of_interaction(g, up_J, g.up)
        end

        log_energy = log_internal_energy(g, interactions)

        new(g.spins_inds, g.row, g.column, g.right, g.down, log_energy, el, eu)
    end
end

function get_Js(g::EE, g1::EE, interactions::Vector{Interaction{T}}; connection::String = "") where T <: AbstractFloat
    J = T[]
    v = [0]
    v1 = [0]
    if connection == "lr"
        v = g.left
        v1 = g1.right
    elseif connection == "ud"
        v = g.up
        v1 = g1.down
    end

    for i in 1:length(v)
        i1 = v[i]
        i2 = v1[i]
        j = getJ(interactions, g.spins_inds[i1], g1.spins_inds[i2])
        push!(J, j)
    end
    J
end

###################### Axiliary functions mainly for the solver ########

function get_system_size(ns::Vector{Node_of_grid})
    mapreduce(x -> length(x.spins_inds), +, ns)
end

function get_system_size(g::MetaGraph)
    mapreduce(i -> length(props(g,i)[:spins]), +, vertices(g))
end

"""
    function sum_over_last(T::Array{T, N}) where N

    used to trace over the phisical index, treated as the last index

"""
function sum_over_last(tensor::Array{T, N}) where {T <: AbstractFloat, N}
    dropdims(sum(tensor, dims = N), dims = N)
end


"""
    last_m_els(vector::Vector{Int}, m::Int)

returns last m element of the vector{Int} or the whole vector if it has less than m elements

"""
function last_m_els(vector::Vector{Int}, m::Int)
    if length(vector) <= m
        return vector
    else
        return vector[end-m+1:end]
    end
end

spins2binary(spins::Vector{Int}) = [Int(i > 0) for i in spins]

binary2spins(spins::Vector{Int}) = [2*i-1 for i in spins]


# this is axiliary function for npz write

function vecvec2matrix(v::Vector{Vector{Int}})
    M = v[1]
    for i in 2:length(v)
        M = hcat(M, v[i])
    end
    M
end
