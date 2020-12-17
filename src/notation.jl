

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

function chimera_cells(i::Int, j::Int, size::Int, cell_size::Tuple{Int, Int})
    cels = zeros(Int, 4*cell_size[1], 2*cell_size[2])
    for k1 in 1:cell_size[1]
        for k2 in 1:cell_size[2]
            ip = k1+(i-1)*cell_size[1]
            jp = k2+(j-1)*cell_size[2]
            cels[(k1-1)*4+1:k1*4, (k2-1)*2+1:k2*2] = chimera_cell(ip, jp, size)
        end
    end
    cels
end

# TODO this needs to be altered for larger chimera cell
function form_a_chimera_grid(n::Int, cell_size::Tuple{Int, Int})
    problem_size = 8*n^2

    n1 = ceil(Int, n/cell_size[1])
    n2 = ceil(Int, n/cell_size[2])
    M = nxmgrid(n1,n2)

    grid = Array{Array{Int}}(undef, (n1,n2))

    for i in 1:n1
        for j in 1:n2

            grid[i,j] = chimera_cells(i,j, problem_size, cell_size)
        end
    end
    Array{Array{Int}}(grid), M
end


#removes bonds that do not fit to the grid, testing function
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

#### constructors of the grid structure ######

function position_in_cluster(cluster::Vector{Int}, i::Int)
    # cluster is assumed to be unique
    return findall(x->x==i, cluster)[1]
end

function positions_in_cluster(all::Vector{Int}, part::Vector{Int})
    [position_in_cluster(all, i) for i in part]
end


struct Element_of_square_grid
    row::Int
    column::Int
    spins_inds::Vector{Int}

    left::Vector{Int}
    right::Vector{Int}
    up::Vector{Int}
    down::Vector{Int}

    #construction from the grid
    function(::Type{Element_of_square_grid})(i::Int, grid::Matrix{Int}, spins_inds::Matrix{Int})
        #s = size(grid)

        left = Int[]
        right = Int[]
        up = Int[]
        down = Int[]

        r = findall(x->x==i, grid)[1]
        row = r[1]
        column = r[2]

        internal_size = size(spins_inds)

        spins = vec(spins_inds)

        #if the node exist
        if column > 1
            left = positions_in_cluster(spins, spins_inds[:, 1])
        end
        if column < size(grid, 2)
            right = positions_in_cluster(spins, spins_inds[:, end])
        end
        if row > 1
            up = positions_in_cluster(spins, spins_inds[1, :])
        end
        if row < size(grid, 1)
            down = positions_in_cluster(spins, spins_inds[end,:])
        end

        new(row, column, spins, left, right, up, down)
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

    left::Vector{Int}
    right::Vector{Int}
    up::Vector{Int}
    down::Vector{Int}

    #construction from the grid
    function(::Type{Element_of_chimera_grid})(i::Int, grid::Matrix{Int}, spins_inds::Matrix{Int})
        #s = size(grid)
        left = Int[]
        right = Int[]
        up = Int[]
        down = Int[]

        r = findall(x->x==i, grid)[1]
        row = r[1]
        column = r[2]

        internal_size = size(spins_inds)
        internal_size[1] % 4 == 0 || error("size $(internal_size) does not fit chimera cell")
        internal_size[2] % 2 == 0 || error("size $(internal_size) does not fit chimera cell")

        cell_columns = div(internal_size[2], 2)
        cell_rows = div(internal_size[1], 4)


        spins = vec(spins_inds)
        if column > 1
            left = positions_in_cluster(spins, spins_inds[:, 2])
        end
        if column < size(grid, 2)*cell_columns
            right = positions_in_cluster(spins, spins_inds[:, end])
        end
        if row > 1
            up = Int[]
            for i in 1:cell_columns
                 up = vcat(up, positions_in_cluster(spins, spins_inds[1:4, 2*i-1]))
            end
        end
        if row < size(grid, 1)*cell_rows
            down = Int[]
            for i in 1:cell_columns
                 down = vcat(down, positions_in_cluster(spins, spins_inds[end-3:end, 2*i-1]))
            end
        end

        new(row, column, spins, left, right, up, down)
    end
    function(::Type{Element_of_chimera_grid})(i::Int, grid::Matrix{Int}, M::Array{Array{Int64,N} where N,2})
        r = findall(x->x==i, grid)[1]
        Element_of_chimera_grid(i, grid, M[r])
    end
end


#####  graph representation #########


function M2graph(M::Matrix{Float64}, sgn::Int = 1)
    size(M,1) == size(M,2) || error("matrix not squared")
    L = size(M,1)
    #TODO we do not require symmetric, is it ok?

    D = Dict{Tuple{Int64,Int64},Float64}()
    for j in 1:size(M, 1)
        for i in 1:j
            if (i == j)
                push!(D, (i,j) => M[j,i])
            elseif M[j,i] != 0.
                push!(D, (i,j) => M[i,j]+M[j,i])
            end
        end
    end
    ising_graph(D, L, 1, sgn)
end


function graph4mps(ig::MetaGraph)
    for v in vertices(ig)
        h = props(ig, v)[:h]
        # -∑hs convention
        set_prop!(ig, v, :energy, [h, -h])
        set_prop!(ig, v, :spectrum, [[-1], [1]])
        set_prop!(ig, v, :spins, [v])
    end
    ig
end

EE = Union{Element_of_square_grid, Element_of_chimera_grid}


# TODO this is temporal
function make_inner_graph(ig::MetaGraph, g_element::EE)
    LL = length(g_element.spins_inds)

    gg = MetaGraph(LL, 0.0)

    p = props(ig)
    set_props!(gg, p)
    set_prop!(gg, :rank, fill(2, LL))

    for i in 1:LL
        v = g_element.spins_inds[i]
        p = props(ig, v)
        set_props!(gg, i, p)
    end

    for i in 1:LL
        for j in i+1:LL
            v1 =  g_element.spins_inds[i]
            v2 =  g_element.spins_inds[j]

            if v2 in all_neighbors(ig, v1)

                p = props(ig, v1, v2)
                add_edge!(gg, i,j)
                set_props!(gg, i,j, p)

            end
        end
    end
    gg
end


#TODO this will be factor_graph
function graph4peps(ig::MetaGraph, cell_size::Tuple{Int, Int} = (1,1)) where T <: AbstractFloat
    L = nv(ig)

    M = zeros(1,1)
    g_elements = []
    if degree(ig, 1) == 2

        s2 = maximum(all_neighbors(ig, 1))-1
        # error will be rised if not Int
        s1 = Int(L/s2)

        grid, M = form_a_grid(cell_size, (s1, s2))
        g_elements = [Element_of_square_grid(i, M, grid) for i in 1:maximum(M)]
    elseif degree(ig, 1) == 5
        # error will be rised if not Int
        n = Int(sqrt(L/8))
        grid, M = form_a_chimera_grid(n, cell_size)
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

        gg = make_inner_graph(ig, g_element)

        no_conf = 2^length(g_element.spins_inds)
        # TODO no_conf can be reduced for approximate spectrum

        spectrum = brute_force(gg; num_states = no_conf)

        # sorting is required for indexing
        p = sortperm([spins2ind(e) for e in spectrum.states], rev=true)

        e = spectrum.energies[p]
        s = spectrum.states[p]

        set_prop!(g, i, :energy, e)
        set_prop!(g, i, :spectrum, s)

        if g_element.column < size(M, 2)
            ip = M[g_element.row, g_element.column+1]

            add_edge!(g, i, ip) || error("Not a grid - cannot add Egde $e")
            set_prop!(g, i, ip, :inds, g_element.right)
        end

        if g_element.column > 1
            ip = M[g_element.row, g_element.column-1]

            M_left = M_of_interaction(g_element, g_elements[ip], ig, s)
            #println(M_left)
            set_prop!(g, i, ip, :M, M_left)
        end

        if g_element.row < size(M, 1)
            ip = M[g_element.row+1, g_element.column]

            add_edge!(g, i, ip) || error("Not a grid - cannot add Egde $e")
            set_prop!(g, i, ip, :inds, g_element.down)
        end

        if g_element.row > 1
            ip = M[g_element.row-1, g_element.column]
            M_up = M_of_interaction(g_element, g_elements[ip], ig, s)
            #println(M_up)
            set_prop!(g, i, ip, :M, M_up)
        end
    end
    g
end


function get_Js(g::EE, g1::EE, ig::MetaGraph)
    J = Float64[]
    v = [0]
    v1 = [0]
    if g.row == g1.row
        v = g.left
        v1 = g1.right
    elseif g.column == g1.column
        v = g.up
        v1 = g1.down
    end

    for i in 1:length(v)
        i1 = v[i]
        i2 = v1[i]
        j = props(ig, g.spins_inds[i1], g1.spins_inds[i2])[:J]
        push!(J, j)
    end
    J
end

function M_of_interaction(g::EE, g1::EE, ig::MetaGraph, spectrum)
    spin_subset = []
    if g.row == g1.row
        spin_subset = g.left
    elseif g.column == g1.column
        spin_subset = g.up
    end
    J = get_Js(g, g1, ig)

    subset_size = length(spin_subset)
    no_spins = length(g.spins_inds)

    energy = zeros(2^subset_size, 2^no_spins)

    for i in 1:2^no_spins

        σ_cluster = spectrum[i]

        k = [1]
        if spin_subset != Int[]
            k = unique([e[spin_subset] for e in spectrum])
        end
        for j in 1:2^subset_size
            #iiii = ind2spin(j, subset_size)
            σ = k[j]

            #println(σ == iiii)
            @inbounds energy[j,i] = -sum(J.*σ.*σ_cluster[spin_subset])
        end
    end
    energy
end


###################### Axiliary functions on spins ########


function ind2spin(i::Int, no_spins::Int = 1)
    s = [2^i for i in 1:no_spins]
    return [1-2*Int((i-1)%j < div(j,2)) for j in s]
end

function ind2spin(i::Int, no_spins::Int, removed_indexes::Vector{Int})
    # TODO make to more efficient and check
    for _ in 1:length(removed_indexes)
        i = i + count(i .> removed_indexes)
    end
    spins = ind2spin(i::Int, no_spins)
end

function spins2ind(s::Vector{Int})
    s = [Int(el == 1) for el in s]
    v = [2^i for i in 0:1:length(s)-1]
    transpose(s)*v+1
end

function spins2ind(s::Vector{Int}, removed_indexes::Vector{Int})
    i = spins2ind(s)
    !(i in removed_indexes) || error("index $i has been removed")
    i - count(i .> removed_indexes)
end


function reindex(i::Int, no_spins::Int, subset_ind::Vector{Int}, removed_indexes::Vector{Int} = Int[])
    if length(subset_ind) == 0
        return 1
    end
    s = ind2spin(i, no_spins, removed_indexes)
    spins2ind(s[subset_ind], removed_indexes)
end

spins2binary(spins::Vector{Int}) = [Int(i > 0) for i in spins]

binary2spins(spins::Vector{Int}) = [2*i-1 for i in spins]


function get_system_size(g::MetaGraph)
    mapreduce(i -> length(props(g,i)[:spins]), +, vertices(g))
end

"""
    function sum_over_last(tensor::Array{T, N}) where {T <: AbstractFloat, N}

sum over last index phisical index
returns Array{T, N-1}

"""
function sum_over_last(tensor::Array{T, N}) where {T <: AbstractFloat, N}
    dropdims(sum(tensor, dims = N), dims = N)
end

"""
    last_m_els(vector::Vector{Int}, m::Int)

returns last m element of the Vector{Int} or the whole vector if it has less than m elements

"""
function last_m_els(vector::Vector{Int}, m::Int)
    if length(vector) <= m
        return vector
    else
        return vector[end-m+1:end]
    end
end

function s2i(a)
    if Int[] in a
        return ones(Int, length(a))
    else
        ret = zeros(Int, length(a))
        k = 1
        for u in unique(a)
           ret = ret + (a .== [u]).*k
           k = k+1
        end
        return ret
    end
end
