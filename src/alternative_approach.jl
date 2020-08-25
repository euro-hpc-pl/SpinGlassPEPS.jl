using LightGraphs
using MetaGraphs
using TikzGraphs
using TensorOperations
using LinearAlgebra

# tensor operations


"""
    contract_tensors(A::Array{Float64, N1} where N1, C::Array{Float64, N2} where N2, mode_a::Int, mode_c::Int)

contracts tensor A with C in the given modes
"""
function contract_tensors(A::Array{Float64, N1} where N1, C::Array{Float64, N2} where N2, mode_a::Int, mode_c::Int)

    iA = collect(1:ndims(A))
    iC = collect(ndims(A)+1:ndims(C)+ndims(A))
    iA[mode_a] = -1
    iC[mode_c] = -1

    tensorcontract(A, iA, C, iC)
end

"""
    perm_moving_mode(N::Int, old_i::Int, new_i::Int)

returns a vector being a permutation that moves a mode of N mode array
from old_i to new_i
"""
function perm_moving_mode(N::Int, old_i::Int, new_i::Int)
    p = collect(1:N)
    filter!(e -> e != old_i, p)
    insert!(p, new_i, old_i)
    return p
end

"""
    function join_modes(A::Array{Float64, N} where N, m1::Int, m2::Int)

Changes N mode array to N-1 one by joining two modes (not necessarly lying one by one).
The new mode in on the position m1.
"""
function join_modes(A::Array{Float64, N} where N, m1::Int, m2::Int)
    s = size(A)
    m1 < m2 <= ndims(A) || error("we expect m1 < m2 ≤ N")
    p = perm_moving_mode(ndims(A), m2, m1+1)
    A = permutedims(A, p)

    siz = [s...]
    siz[m1] = s[m1]*s[m2]
    deleteat!(siz, m2)
    reshape(A, (siz...))
end

# from here the qubo is introduced




function add_qubo_el!(graph::MetaGraph, q::Qubo_el)
    i = q.ind
    if i[1] == i[2]
        set_prop!(graph, i[1], :h, q.coupling) || error("vertex not in graph")
    else
        set_prop!(graph, Edge(i...), :J, q.coupling) || error("edge not in graph")
    end
end

"""
    make_grid3x3()

returns the 3x3 grid the light graph
"""
function make_grid3x3()
    g = path_graph(9)

    add_edge!(g, 1, 6)
    add_edge!(g, 2, 5)
    add_edge!(g, 5, 8)
    add_edge!(g, 4, 9)

    for i in 1:9
        add_edge!(g, i, i+1)
    end
    return g
end

"""
    add_locations3x3!(graph::MetaGraph)

to each link of the grid meta graph add :side the pair
of strings saying if the bound is to the left, right up or down from the vertex.

it is given in the increasing order of vertices numeration.


"""
function add_locations3x3!(graph::MetaGraph)
    # e.g. from 1 a link egde to right and from 2 the same to lest
    a = set_prop!(graph, Edge(1,2), :side, ["r","l"])
    b = set_prop!(graph, Edge(2,3), :side, ["r","l"])

    c = set_prop!(graph, Edge(4,5), :side, ["l","r"])
    d = set_prop!(graph, Edge(5,6), :side, ["l","r"])

    e = set_prop!(graph, Edge(7,8), :side, ["r","l"])
    f = set_prop!(graph, Edge(8,9), :side, ["r","l"])

    g = set_prop!(graph, Edge(1,6), :side, ["d","u"])
    h = set_prop!(graph, Edge(6,7), :side, ["d","u"])

    i = set_prop!(graph, Edge(2,5), :side, ["d","u"])
    j = set_prop!(graph, Edge(5,8), :side, ["d","u"])

    k = set_prop!(graph, Edge(3,4), :side, ["d","u"])
    l = set_prop!(graph, Edge(4,9), :side, ["d","u"])

    # also the "empty" modes is initialised
    for e in edges(graph)
        set_prop!(graph, e, :modes, [0,0])
    end

    a*b*c*d*e*f*g*h*i*j*k*l || error("vertex not in graph")
end

function make_graph3x3()
    g = make_grid3x3()
    # this will be a meta graph
    mg = MetaGraph(g)
    add_locations3x3!(mg)
    mg
end

function add_qubo2graph!(mg::MetaGraph, qubo::Vector{Qubo_el})
    for q in qubo
        add_qubo_el!(mg, q)
    end
end



# tensors to vertices.
"""
    function index2physical(i::Int)

changes index i.e. i ∈ [1,2] to spin ∈ [-1,1]
"""
function index2physical(i::Int)
    i in [1,2] || error("array index should be 1 or 2, we have $i")
    2*i-3
end

"""
    function read_pair_from_edge(mg::MetaGraph, v1::Int, v2::Int, s::Symbol)

Returns the 2 elements vector of features tied to the egde between vertices.
As features are always attached in the increasing order of vertices, the permutation is used.

"""
function read_pair_from_edge(mg::MetaGraph, v1::Int, v2::Int, s::Symbol)
    # each pair is given in the increasing order of vertices
    e = Edge(v1, v2)
    has_prop(mg, e, s) || error("there is no direct link or $s between $(v1) and $(v2)")
    p = sortperm([v1, v2])
    pair = props(mg, e)[s]
    pair[p]
end

function write_pair2edge!(mg::MetaGraph, v1::Int, v2::Int, s::Symbol, pair::Vector)
    # each pair is given in the increasing order of vertices
    e = Edge(v1, v2)
    p = sortperm([v1, v2])
    length(pair) == 2 || error("the pair has more than 2 lements")
    set_prop!(mg, e, s, pair[p]) || error("there is no direct link or $s between $(v1) and $(v2)")
end
"""
    readJs(mg::MetaGraph, vertex::Int)

read couplings from give vertex, if there is no returns 0.
"""
function readJs(mg::MetaGraph, vertex::Int)
    # linear trem coefficient
    h = props(mg, vertex)[:h]

    # quadratic
    Jir = 0.
    Jid = 0.

    for v in neighbors(mg, vertex)
        directions = read_pair_from_edge(mg, vertex, v, :side)

        if directions[1] == "r"
            Jir = props(mg, Edge(vertex,v))[:J]
        elseif directions[1] == "d"
            Jid = props(mg, Edge(vertex,v))[:J]
        end
    end
    Jir, Jid, h
end

"""
    sort2lrud(vec_of_directions::Vector{String})

sort a vector of strings with elements from ["l","r","u","d"]
in the order as in ["l","r","u","d"]
"""
function sort2lrud(vec_of_directions::Vector{String})
    ret = Vector{String}()
    for el in ["l","r","u","d"]
        if el in vec_of_directions
            push!(ret, el)
        end
    end
    ret
end

"""
    bond_directions(mg::MetaGraph, vertex::Int)

returns the vector of string of the directions of bonds for the given vertex (in the sorted order)
"""
function bond_directions(mg::MetaGraph, vertex::Int)
    bond_directions = Vector{String}()
    for v in neighbors(mg, vertex)
        directions = read_pair_from_edge(mg, vertex, v, :side)
        push!(bond_directions, directions[1])
    end
    sort2lrud(bond_directions)
end

"""
    get_modes(mg::MetaGraph, vertex::Int)

returns the 4 elements array of either the mode that would correspond with the
full tensor or zero if there is no conecction in the given direction
"""
function get_modes(mg::MetaGraph, vertex::Int)
    bd = bond_directions(mg, vertex)
    modes = zeros(Int, 0)
    j = 0
    for d in ["l", "r", "u", "d"]
        j = j+1
        if d in bd
            push!(modes, j)
        end
    end
    modes
end

"""
    function makeTensor(mg::MetaGraph, vertex::Int)

given the vertex of the MetaGraph generates the full
tensor with both virtual dimentions and physical  dimention
"""
function makeTensor(mg::MetaGraph, vertex::Int)
    modes = get_modes(mg, vertex)
    Js = readJs(mg, vertex)
    virtual_dims = length(modes)

    A = zeros(fill(2, virtual_dims+1)...)
    lrud = [0,0,0,0]
    for i in CartesianIndices(size(A))

        for l in 1:virtual_dims
            lrud[modes[l]] = index2physical(i[l])
        end
        s = index2physical(i[virtual_dims+1])
        A[i] = Tgen(lrud..., s, Js...)
    end
    A
end

function add_tensor2vertex(mg::MetaGraph, vertex::Int, s::Int = 0)
    T = makeTensor(mg, vertex)
    if s == 0
        T = sum_over_last(T)
    else
        T = set_last(T, s)
    end
    set_prop!(mg, vertex, :tensor, T)
    bd = bond_directions(mg, vertex)
    for v in neighbors(mg, vertex)
        e = Edge(vertex,v)
        direction_on_graph = read_pair_from_edge(mg, vertex, v, :side)[1]
        mode = findall(x->x==direction_on_graph, bd)[1]
        m = props(mg, e)[:modes]
        p = sortperm([vertex,v])
        m[p[1]] = mode
        set_prop!(mg, e, :modes, m)
    end
end


function contract_vertices!(mg::MetaGraph, v1::Int, v2::Int)
    tg1 = props(mg, v1)[:tensor]
    N1 = ndims(tg1)
    tg2 = props(mg, v2)[:tensor]

    modes = read_pair_from_edge(mg, v1, v2, :modes)
    tg = contract_tensors(tg1, tg2, modes[1], modes[2])
    set_prop!(mg, v1, :tensor, tg)
    rem_edge!(mg, Edge(v1, v2))

    for v in collect(neighbors(mg, v2))
        m = read_pair_from_edge(mg, v, v2, :modes)
        m_v = m[1]
        m_v1new = m[2] + N1 - 1
        rem_edge!(mg, Edge(v, v2))

        p1 = sortperm([v, v1])
        m_new = [m_v, m_v1new][p1]

        if has_edge(mg, Edge(v, v1))

            m_all = props(mg, Edge(v, v1))[:modes]
            m_new = vcat(m_all, m_new)
            set_prop!(mg, Edge(v, v1), :modes, m_new)
        else
            add_edge!(mg, Edge(v, v1))
            set_prop!(mg, Edge(v, v1), :modes, m_new)
        end
    end
    clear_props!(mg, v2)
end

function move_modes!(mg::MetaGraph, v1::Int, mode_deleted::Int)
    for v in neighbors(mg, v1)
        p = sortperm([v1, v])
        modes = props(mg, Edge(v, v1))[:modes]

        if length(modes) > 2
            # this in the first element of the second pair
            i = [3,4][p[1]]
            if modes[i] > mode_deleted
                modes[i] = modes[i] - 1
            end
        end
    end
end

function combine_legs_exact!(mg::MetaGraph, v1::Int, v2::Int)
    p = sortperm([v1, v2])
    e = Edge(v1, v2)
    all_modes = props(mg, e)[:modes]
    length(all_modes) == 4 || error("no double legs to be joint")
    first_pair = all_modes[1:2][p]
    second_pair = all_modes[3:4][p]

    t = props(mg, v1)[:tensor]
    t1 = join_modes(t, first_pair[1], second_pair[1])

    set_prop!(mg, v1, :tensor, t1)

    write_pair2edge!(mg, v1, v2, :modes, first_pair)

    t = props(mg, v2)[:tensor]

    t2 = join_modes(t, first_pair[2], second_pair[2])
    set_prop!(mg, v2, :tensor, t2)

    # changes elements in the array of modes
    move_modes!(mg, v1, second_pair[1])
    move_modes!(mg, v2, second_pair[1])
end

function reduce_bond_size_svd!(mg::MetaGraph, v1::Int, v2::Int, threshold::Float64 = 1e-12)

    modes = read_pair_from_edge(mg, v1, v2, :modes)
    t1 = props(mg, v1)[:tensor]
    t2 = props(mg, v2)[:tensor]
    s1 = size(t1)
    N1 = length(s1)
    s2 = size(t2)
    N2 = length(s2)

    p1 = perm_moving_mode(N1, modes[1], 1)
    p1inv = invperm(p1)
    t1 = permutedims(t1, p1)
    pi = prod(s1[p1[2:end]])
    A1 = reshape(t1, (s1[p1[1]], pi))

    p2 = perm_moving_mode(N2, modes[2], 1)
    p2inv = invperm(p2)

    t2 = permutedims(t2, p2)
    pi = prod(s2[p2[2:end]])
    A2 = reshape(t2, (s2[p2[1]], pi))

    U,Σ,V = svd(A1)
    k = length(filter(e -> e > threshold, Σ))
    proj = transpose(U)[1:k,:]

    A1_red = proj*A1
    T1_red = reshape(A1_red, (k, s1[p1[2:end]]...))
    T1_red = permutedims(T1_red, p1inv)

    A2_red = proj*A2
    T2_red = reshape(A2_red, (k, s2[p2[2:end]]...))
    T2_red = permutedims(T2_red, p2inv)

    set_correctly1 = set_prop!(mg, v1, :tensor, T1_red)
    set_correctly2 = set_prop!(mg, v2, :tensor, T2_red)
    set_correctly1*set_correctly2 || error("features of nodes not set properly")
end

function merge_lines!(mg::MetaGraph, v_line1::Vector{Int}, v_line2::Vector{Int}, approx_svd::Bool = true)
    length(v_line1) == length(v_line2) || error("linse needs to be the same size")
    for i in 1:length(v_line1)
        contract_vertices!(mg, v_line1[i],v_line2[i])
    end
    for i in 1:(length(v_line1)-1)
        combine_legs_exact!(mg, v_line1[i], v_line1[i+1])
    end
    if approx_svd
        for i in 1:(length(v_line1)-1)
            reduce_bond_size_svd!(mg, v_line1[i], v_line1[i+1])
        end
        for i in (length(v_line1)-1):-1:1
            reduce_bond_size_svd!(mg, v_line1[i+1], v_line1[i])
        end
    end
end

# computation of probabilities

"""
    set_spins2firs_k!(mg::MetaGraph, s::Vector{Int})

    trace over all spins but first k=length(s)
"""
function set_spins2firs_k!(mg::MetaGraph, s::Vector{Int})
    for i in 1:length(s)
        add_tensor2vertex(mg, i, s[i])
    end
    l = (length(s)+1)
    for i in l:nv(mg)
        add_tensor2vertex(mg, i)
    end
end

"""
    set_spins2firs_k!(mg::MetaGraph, s::Int)

    trace over all spins but first that is s
"""
set_spins2firs_k!(mg::MetaGraph, s::Int) = set_spins2firs_k!(mg, [s])

"""
    set_spins2firs_k!(mg::MetaGraph)

    trace over all spins
"""
set_spins2firs_k!(mg::MetaGraph) = set_spins2firs_k!(mg, Int[])


function compute_marginal_prob(mg::MetaGraph, ses::Vector{Int}, svd_approx::Bool = true)
    set_spins2firs_k!(mg, ses)
    # v2d is the configuration of the grid
    v2d = [1 2 3; 6 5 4; 7 8 9]
    for i in size(v2d,1)-1:-1:1
        merge_lines!(mg, v2d[i,:], v2d[i+1,:], svd_approx)
    end
    for i in size(v2d,2)-1:-1:1
        contract_vertices!(mg, v2d[1,i], v2d[1,i+1])
    end
    props(mg, 1)[:tensor][1]
end


# this solver is called naive since we contract wha whole
# grit at each itteration, and do not take advantage from
# breaking bonds between vertices with set spins.

# it is aimed for testing the final solver

function naive_solve(qubo::Vector{Qubo_el}, M::Int, approx::Bool = true)
    problem_size = 9
    a = zeros(Int, 2,1)
    a[1,1] = 1
    a[2,1] = -1
    for j in 1:problem_size
        objective = Float64[]
        for i in 1:size(a,1)
            r = optimisation_step_naive(qubo, a[i,:], approx)
            push!(objective, r)
        end
        p = sortperm(objective)
        p1, k = get_last_m(p, M)
        a_temp = zeros(Int, k, j)
        for i in 1:k
            a_temp[i,:] = a[p1[i],:]
        end
        if j == problem_size
            return a_temp, objective[p1]
        else
            a = add_another_spin2configs(a)
        end
    end
    0
end

"""
    function optimisation_step_naive(qubo::Vector{Qubo_el}, ses::Vector{Int}, approx::Bool = true)

returns the non-normalised marginal probability of the given partial configuration
in ses::Vector{Int}

Naive since each step contracts the whole grid to the single value.
"""
function optimisation_step_naive(qubo::Vector{Qubo_el}, ses::Vector{Int}, use_svd_approx::Bool = true)
    mg = make_graph3x3();
    add_qubo2graph!(mg, qubo)
    compute_marginal_prob(mg, ses, use_svd_approx)
end

# BG algorithm.
# 1. compute(P(s1)) by contractiong whole network

#2. P_{cond}(s_k | s_1 , ...., s_{k-1})

# for conditional probability only e^(-J_{i1,i2)
# from the boundary
