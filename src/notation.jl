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

#spins2ind(s::Int) = spins2ind([s])


function spins2ind(s::Vector{Int})
    s = [Int(el == 1) for el in s]
    v = [2^i for i in 0:1:length(s)-1]
    transpose(s)*v+1
end


#function reindex(i::Int, all_spins::Vector{Int}, subset_spins::Vector{Int})
#    s = ind2spin(i, length(all_spins))
#    k = [findall(x->x==j, all_spins)[1] for j in subset_spins]
#    spins2ind(s[k])
#end

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


function is_node(j::Int, k::Int, grid::Matrix{Int})
    try
        grid[j, k]
        return true
    catch
        return false
    end
end

function index_of_interacting_spins(all::Vector{Int}, part::Vector{Int})
    spins = Int[]

    for el in part
        index = findall(x->x==el, all)[1]
        push!(spins, index)
    end
    spins
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
    spin_inds::Vector{Int}
    intra_struct::Vector{Vector{Int}}
    left::Vector{Int}
    left_J::Vector{Float64}
    right::Vector{Int}
    up::Vector{Int}
    up_J::Vector{Float64}
    down::Vector{Int}
    energy::Vector{Float64}
    connected_nodes::Vector{Int}
    connected_spins::Vector{Matrix{Int}}
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

        a = findall(x->x==i, grid)[1]
        j = a[1]
        k = a[2]

        if is_node(j, k-1, grid)
            l = [1]
            ip = grid[j, k-1]
            left_J = [getJ(interactions, i, ip)]
            push!(connected_nodes, ip)
        end
        if is_node(j, k+1, grid)
            r = [1]
            push!(connected_nodes, grid[j, k+1])
        end
        if is_node(j-1, k, grid)
            u = [1]
            ip = grid[j-1, k]
            up_J = [getJ(interactions, i, ip)]
            push!(connected_nodes, ip)
        end
        if is_node(j+1, k, grid)
            d = [1]
            push!(connected_nodes, grid[j+1, k])
        end

        connected_spins = [ones(Int, 1,2) for _ in connected_nodes]
        for j in 1:length(connected_nodes)
            connected_spins[j] = reshape([i, connected_nodes[j]], (1,2))
        end

        h = getJ(interactions, i, i)
        log_energy = [-h, h]

        new(i, [i], intra_struct, l, left_J, r, u, up_J, d, log_energy, connected_nodes, connected_spins)
    end
    #construction from matrix of matrices (a grid of many nodes)
    function(::Type{Node_of_grid})(i::Int, Mat::Matrix{Int},
                                grid::Array{Array{Int64,N} where N,2},
                                interactions::Vector{Interaction{T}};
                                chimera::Bool = false) where T <: AbstractFloat


        connected_nodes = Int[]

        a = findall(x->x==i, Mat)[1]
        j = a[1]
        k = a[2]
        M = grid[j,k]
        # this transpose makes better ordering
        spin_inds = Vector{Int}(vec(transpose(M)))
        intra_struct = Vector[]
        s = size(M)
        no_spins = length(spin_inds)
        log_energy = zeros(T, 2^no_spins)

        first_spin_index = Int[]
        second_spin_index = Int[]
        Js = T[]

        if ! chimera
            for k in 1:s[1]
                if length(M[k,:]) > 1
                    push!(intra_struct, M[k,:])
                    for j in 2:s[2]
                        push!(first_spin_index, M[k, j-1])
                        push!(second_spin_index, M[k, j])
                        push!(Js, getJ(interactions, M[k, j-1], M[k, j]))
                    end
                end
            end
            for k in 1:s[2]
                if length(M[:,k]) > 1
                    push!(intra_struct, M[:,k])
                    for j in 2:s[1]
                        push!(first_spin_index, M[j-1, k])
                        push!(second_spin_index, M[j, k])
                        push!(Js, getJ(interactions, M[j-1, k], M[j, k]))
                    end
                end
            end
        else
            for k in 1:s[1]
                for j in 1:s[1]
                    push!(intra_struct, [M[k, 1], M[j,end]])
                    push!(first_spin_index, M[k,1])
                    push!(second_spin_index, M[j,end])
                    push!(Js, getJ(interactions, M[k,1], M[j,end]))
                end
            end
        end

        i1 = index_of_interacting_spins(spin_inds, first_spin_index)
        i2 = index_of_interacting_spins(spin_inds, second_spin_index)
        hs = [getJ(interactions, a, a) for a in spin_inds]

        for i in 1:2^no_spins
            s = ind2spin(i, no_spins)
            e = sum(s.*hs)
            for j in 1:length(i1)
                e = e + 2*s[i1[j]]*s[i2[j]]*Js[j]
            end
            log_energy[i] = e
        end

        #l,r,u,d = read_connecting_pairs(Mat, i)
        connected_spins = Matrix{Int}[]

        l = Int[]
        r = Int[]
        u = Int[]
        d = Int[]
        left_J = Float64[]
        up_J = Float64[]

        if is_node(j, k-1, Mat)
            push!(connected_nodes, Mat[j, k-1])

            v1 = 0
            if chimera
                v1 = M[:,end]
            else
                v1 = M[:,1]
            end
            l = index_of_interacting_spins(spin_inds, v1)

            Mp = grid[j, k-1]
            v2 = Mp[:,end]

            for i in 1:length(v1)
                push!(left_J, getJ(interactions, v1[i], v2[i]))
            end

            push!(connected_spins, hcat(v1, v2))

        end
        if is_node(j, k+1, Mat)
            push!(connected_nodes, Mat[j, k+1])

            v1 = M[:,end]
            r = index_of_interacting_spins(spin_inds, v1)

            v2 = 0
            Mp = grid[j, k+1]
            if chimera
                v2 = Mp[:,end]
            else
                v2 = Mp[:,1]
            end

            push!(connected_spins, hcat(v1, v2))
        end

        if is_node(j-1, k, Mat)

            push!(connected_nodes, Mat[j-1, k])
            v1 = 0
            v2 = 0
            Mp = grid[j-1, k]
            if chimera
                v1 = M[:,1]
                v2 = Mp[:,1]
            else
                v1 = M[1,:]
                v2 = Mp[end,:]
            end
            u = index_of_interacting_spins(spin_inds, v1)

            for i in 1:length(v1)
                push!(up_J, getJ(interactions, v1[i], v2[i]))
            end

            push!(connected_spins, hcat(v1, v2))
        end
        if is_node(j+1, k, Mat)
            push!(connected_nodes, Mat[j+1, k])

            v1 = 0
            v2 = 0

            Mp = grid[j+1, k]
            if chimera
                v1 = M[:,1]
                v2 = Mp[:,1]
            else
                v1 = M[end,:]
                v2 = Mp[1,:]
            end
            d = index_of_interacting_spins(spin_inds, v1)

            push!(connected_spins, hcat(v1, v2))
        end

        new(i, spin_inds, intra_struct, l, left_J, r, u, up_J, d, log_energy, connected_nodes, connected_spins)
    end
    # construction from interactions directly, It will not check if interactions fits grid
    function(::Type{Node_of_grid})(i::Int, interactions::Vector{Interaction{T}}) where T <: AbstractFloat
        x = Vector{Int}[]
        connected_nodes = Int[]
        for q in interactions
            if (q.ind[1] == i && q.ind[2] != i)
                push!(connected_nodes, q.ind[2])
            end
            if (q.ind[2] == i && q.ind[1] != i)
                push!(connected_nodes, q.ind[1])
            end
        end
        connected_spins = [ones(Int, 1,2) for _ in connected_nodes]
        for j in 1:length(connected_nodes)
            connected_spins[j] = reshape([i, connected_nodes[j]], (1,2))
        end

        h = getJ(interactions, i, i)
        log_energy = [-h, h]

        new(i, [i], x, Int[], Float64[], Int[], Int[], Float64[], Int[], log_energy, connected_nodes, connected_spins)
    end
end

###################### Axiliary functions mainly for the solver ########

function get_system_size(ns::Vector{Node_of_grid})
    mapreduce(x -> length(x.spin_inds), +, ns)
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
