using TensorOperations

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

function get_system_size(interactions::Vector{Interaction{T}}) where T <: AbstractFloat
    size = 0
    for q in interactions
        size = maximum([size, q.ind[1], q.ind[2]])
    end
    size
end


"""
    ind2spin(i::Int, s::Int = 2)

return a spin from the physical index, if size is 1, returns zero.
"""
function ind2spin(i::Int, size::Int = 1)
    if size == 0
        return [0]
    else
        s = [2^i for i in 1:size]
        return [1-2*Int((i-1)%j < div(j,2)) for j in s]
    end
end


function spins2ind(s::Vector{Int})
    s = [Int(el == 1) for el in s]
    v = [2^i for i in 0:1:length(s)-1]
    transpose(s)*v+1
end



function reindex1(i::Int, n::Int, subset_ind::Vector{Int})
    s = ind2spin(i, n)
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

function nxmgrid(n::Int, m::Int)
    grid = zeros(Int, n, m)
    for i in 1:m
       for j in 1:n
           grid[j,i] = i+m*(j-1)
       end
   end
   grid
end

function chimera_cell(i::Int, j::Int, size::Int)
    size = Int(sqrt(size/8))
    ofset = 8*(j-1)+8*size*(i-1)
    cel = zeros(Int, 4, 2)
    cel[:,1] = [k+ofset for k in 1:4]
    cel[:,2] = [k+ofset for k in 5:8]
    cel
end


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
    i::Int
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
        #if the node exist

        if is_element(row, column-1, grid)
            left = spins_inds[:, 1]
        end
        if is_element(row, column+1, grid)
            right = spins_inds[:, end]
        end
        if is_element(row-1, column, grid)
            up = spins_inds[1, :]
        end
        if is_element(row+1, column, grid)
            down = spins_inds[end,:]
        end

        #TODO this representation will be changes
        #spins_inds = sort(vec(spins_inds))
        spins_inds = Vector{Int}(vec(transpose(spins_inds)))

        new(i, row, column, spins_inds, intra_struct, left, right, up, down)
    end
end


struct Element_of_chimera_grid
    i::Int
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
        #if the node exist

        if is_element(row, column-1, grid)
            left = spins_inds[:, 2]
        end
        if is_element(row, column+1, grid)
            right = spins_inds[:, 2]
        end
        if is_element(row-1, column, grid)
            up = spins_inds[:, 1]
        end
        if is_element(row+1, column, grid)
            down = spins_inds[:, 1]
        end

        #spins_inds = sort(vec(spins_inds))
        spins_inds = Vector{Int}(vec(transpose(spins_inds)))

        new(i, row, column, spins_inds, intra_struct, left, right, up, down)
    end
end

function compute_log_energy(g::Union{Element_of_square_grid, Element_of_chimera_grid},
                            interactions::Vector{Interaction{T}}) where T <: AbstractFloat

    # TODO h and J this will be tead from a grid
    hs = [getJ(interactions, i, i) for i in g.spins_inds]
    no_spins = length(g.spins_inds)
    #g.intra_struct
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


"""
    struct Node_of_grid

this structure is supposed to include all information about nodes on the grid.
necessary to crearte the corresponding tensor (e.g. the element of the peps).
"""
# TODO this for sure need to be clarified, however I would leave such
# approach. It allows for easy creation of various grids of interactions
# (e.g. Pegasusu) and its modification during computation (if necessary).

struct Node_of_grid
    i::Int
    spins_inds::Vector{Int}
    #intra_struct::Vector{Vector{Int}}
    left::Vector{Int}
    left_J::Vector{Float64}
    right::Vector{Int}
    up::Vector{Int}
    up_J::Vector{Float64}
    down::Vector{Int}
    energy::Vector{Float64}
    connected_nodes::Vector{Int}
    connected_spins::Vector{Matrix{Int}}
    connected_J::Vector{Float64}

    #construction from the grid
    function(::Type{Node_of_grid})(i::Int, grid::Matrix{Int}, interactions::Vector{Interaction{T}}) where T <: AbstractFloat
        s = size(grid)
        intra_struct = Vector{Int}[]
        connected_nodes = Int[]

        l = Int[]
        r = Int[]
        u = Int[]
        d = Int[]
        left_J = Float64[]
        up_J = Float64[]
        connected_J = Float64[]

        a = findall(x->x==i, grid)[1]
        j = a[1]
        k = a[2]

        if is_element(j, k-1, grid)
            l = [1]
            ip = grid[j, k-1]
            left_J = [getJ(interactions, i, ip)]
            push!(connected_nodes, ip)
            push!(connected_J, getJ(interactions, i, ip))
        end
        if is_element(j, k+1, grid)
            r = [1]
            ip = grid[j, k+1]
            push!(connected_nodes, ip)
            push!(connected_J, getJ(interactions, i, ip))
        end
        if is_element(j-1, k, grid)
            u = [1]
            ip = grid[j-1, k]
            up_J = [getJ(interactions, i, ip)]
            push!(connected_nodes, ip)
            push!(connected_J, getJ(interactions, i, ip))
        end
        if is_element(j+1, k, grid)
            d = [1]
            ip = grid[j+1, k]
            push!(connected_nodes, ip)
            push!(connected_J, getJ(interactions, i, ip))
        end

        connected_spins = [ones(Int, 1,2) for _ in connected_nodes]
        for j in 1:length(connected_nodes)
            connected_spins[j] = reshape([i, connected_nodes[j]], (1,2))
        end

        h = getJ(interactions, i, i)
        log_energy = [-h, h]

        new(i, [i], l, left_J, r, u, up_J, d, log_energy, connected_nodes, connected_spins, connected_J)
    end
    #construction from matrix of matrices (a grid of many nodes)
    function(::Type{Node_of_grid})(i::Int, Mat::Matrix{Int},
                                grid::Array{Array{Int64,N} where N,2},
                                interactions::Vector{Interaction{T}};
                                chimera::Bool = false) where T <: AbstractFloat


        connected_nodes = Int[]

        # TODO to below 10 lines is the working by-pass will be changed to graphs
        a = findall(x->x==i, Mat)[1]
        j = a[1]
        k = a[2]
        M = grid[j,k]

        g = 0.
        if chimera
            g = Element_of_chimera_grid(i, Mat, M)
        else
            g = Element_of_square_grid(i, Mat, M)
        end


        # TODO below will be done on a graph
        connected_spins = Matrix{Int}[]
        left_J = Float64[]
        up_J = Float64[]
        connected_J = Float64[]

        if is_element(j, k-1, Mat)
            ip = Mat[j, k-1]
            Mp = grid[j,k-1]

            push!(connected_nodes, ip)

            g1 = 0.
            if chimera
                g1 = Element_of_chimera_grid(ip, Mat, Mp)
            else
                g1 = Element_of_square_grid(ip, Mat, Mp)
            end

            for i in 1:length(g.left)
                J = getJ(interactions, g.left[i], g1.right[i])
                push!(left_J, J)
                push!(connected_J, J)
            end
            push!(connected_spins, hcat(g.left, g1.right))

        end
        if is_element(j, k+1, Mat)

            ip = Mat[j, k+1]
            Mp = grid[j,k+1]

            push!(connected_nodes, ip)

            g1 = 0.
            if chimera
                g1 = Element_of_chimera_grid(ip, Mat, Mp)
            else
                g1 = Element_of_square_grid(ip, Mat, Mp)
            end

            for i in 1:length(g.right)
                J = getJ(interactions, g.right[i], g1.left[i])
                push!(connected_J, J)
            end
            push!(connected_spins, hcat(g.right, g1.left))
        end

        if is_element(j-1, k, Mat)

            ip = Mat[j-1, k]
            Mp = grid[j-1,k]

            push!(connected_nodes, ip)

            g1 = 0.
            if chimera
                g1 = Element_of_chimera_grid(ip, Mat, Mp)
            else
                g1 = Element_of_square_grid(ip, Mat, Mp)
            end

            for i in 1:length(g.up)
                J = getJ(interactions, g.up[i], g1.down[i])
                push!(connected_J, J)
                push!(up_J, J)
            end
            push!(connected_spins, hcat(g.up, g1.down))

        end
        if is_element(j+1, k, Mat)

            ip = Mat[j+1, k]
            Mp = grid[j+1,k]

            push!(connected_nodes, ip)

            g1 = 0.
            if chimera
                g1 = Element_of_chimera_grid(ip, Mat, Mp)
            else
                g1 = Element_of_square_grid(ip, Mat, Mp)
            end

            for i in 1:length(g.down)
                J = getJ(interactions, g.down[i], g1.up[i])
                push!(connected_J, J)
            end
            push!(connected_spins, hcat(g.down, g1.up))
        end

        l = index_of_interacting_spins(g.spins_inds, g.left)
        r = index_of_interacting_spins(g.spins_inds, g.right)
        u = index_of_interacting_spins(g.spins_inds, g.up)
        d = index_of_interacting_spins(g.spins_inds, g.down)

        log_energy = compute_log_energy(g, interactions)

        new(i, g.spins_inds, l, left_J, r, u, up_J, d, log_energy, connected_nodes, connected_spins, connected_J)
    end
    # construction from interactions directly, It will not check if interactions fits grid
    function(::Type{Node_of_grid})(i::Int, interactions::Vector{Interaction{T}}) where T <: AbstractFloat
        x = Vector{Int}[]
        connected_nodes = Int[]
        connected_J = Float64[]

        for q in interactions
            if (q.ind[1] == i && q.ind[2] != i)
                push!(connected_nodes, q.ind[2])
                push!(connected_J, q.coupling)
            end
            if (q.ind[2] == i && q.ind[1] != i)
                push!(connected_nodes, q.ind[1])
                push!(connected_J, q.coupling)
            end
        end
        connected_spins = [ones(Int, 1,2) for _ in connected_nodes]
        for j in 1:length(connected_nodes)
            connected_spins[j] = reshape([i, connected_nodes[j]], (1,2))
        end

        h = getJ(interactions, i, i)
        log_energy = [-h, h]

        new(i, [i], Int[], Float64[], Int[], Int[], Float64[], Int[], log_energy, connected_nodes, connected_spins, connected_J)
    end
end

###################### Axiliary functions mainly for the solver ########

function get_system_size(ns::Vector{Node_of_grid})
    mapreduce(x -> length(x.spins_inds), +, ns)
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
