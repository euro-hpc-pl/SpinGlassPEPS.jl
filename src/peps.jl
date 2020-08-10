using LightGraphs
using MetaGraphs
using TikzGraphs
using TensorOperations

β = 1.
# tensor operations
"""
    function sum_over_last(T::Array{Float64, N}) where N

    used to trace over the phisical index, treated as the last index

"""
function sum_over_last(T::Array{Float64, N}) where N
    tensorcontract(T, collect(1:N), ones(size(T,N)), [N])
end

"""
    set_last(T::Array{Float64, N}, s::Int) where N

set value of the physical index, s ∈ {-1,1} are supported
"""
function set_last(T::Array{Float64, N}, s::Int) where N
    if s == -1
        B = [1.,0.]
    elseif s == 1
        B = [0.,1.]
    else
        error("spin value $s ∉ {-1, 1}")
    end
    tensorcontract(T, collect(1:N), B, [N])
end


function contract_ts1(A::Array{Float64, N1} where N1, C::Array{Float64, N2} where N2, mode_a::Int, mode_c::Int)

    iA = collect(1:ndims(A))
    iC = collect(ndims(A)+1:ndims(C)+ndims(A))
    iA[mode_a] = -1
    iC[mode_c] = -1

    tensorcontract(A, iA, C, iC)
end


struct Qubo_el
    ind::Tuple{Int, Int}
    coupling::Float64
end

function add_qubo_el!(graph::MetaGraph, q::Qubo_el)
    i = q.ind
    if i[1] == i[2]
        set_prop!(graph, i[1], :h, q.coupling) || error("vertex not in graph")
    else
        set_prop!(graph, Edge(i...), :J, q.coupling) || error("edge not in graph")
    end
end

# 3x3 grid

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

function add_locations3x3_1!(graph::MetaGraph)
    # e.g. from 1 a link egde to right and from 2 the same to lest
    a = set_prop!(graph, Edge(1,2), :type, ["r","l"])
    b = set_prop!(graph, Edge(2,3), :type, ["r","l"])

    c = set_prop!(graph, Edge(4,5), :type, ["l","r"])
    d = set_prop!(graph, Edge(5,6), :type, ["l","r"])

    e = set_prop!(graph, Edge(7,8), :type, ["r","l"])
    f = set_prop!(graph, Edge(8,9), :type, ["r","l"])

    g = set_prop!(graph, Edge(1,6), :type, ["d","u"])
    h = set_prop!(graph, Edge(6,7), :type, ["d","u"])

    i = set_prop!(graph, Edge(2,5), :type, ["d","u"])
    j = set_prop!(graph, Edge(5,8), :type, ["d","u"])

    k = set_prop!(graph, Edge(3,4), :type, ["d","u"])
    l = set_prop!(graph, Edge(4,9), :type, ["d","u"])

    for e in edges(graph)
        set_prop!(graph, e, :modes, [0,0])
    end

    a*b*c*d*e*f*g*h*i*j*k*l || error("vertex not in graph")
end

function make_graph3x3()
    g = make_grid3x3()
    # this will be a meta graph
    mg = MetaGraph(g)
    add_locations3x3_1!(mg)
    mg
end

function add_qubo2graph(mg::MetaGraph, qubo::Vector{Qubo_el})
    for q in qubo
        add_qubo_el!(mg, q)
    end
    mg
end


# generation of tensors

"""
    delta(a::Int, b::Int)

Dirac delta, additionally return 1 if first arguments is zero for
implementation cause
"""
function delta(γ::Int, s::Int)
    if γ != 0
        return Int(γ == s)
    end
    1
end

"""
    c(γ::Int, J::Float64, s::Int)

c building block
"""
c(γ::Int, J::Float64, s::Int) =  exp(-β*J*γ*s)


function Tgen(l::Int, r::Int, u::Int, d::Int, s::Int, Jir::Float64, Jid::Float64, Jii::Float64)
    delta(l, s)*delta(u, s)*c(r, Jir, s)*c(d, Jid, s)*exp(-β*Jii*s)
end


# tensors to vertices.

function index2physical(i::Int)
    i in [1,2] || error("array index should be 1 or 2, we have $i")
    2*i-3
end

function getJs(mg::MetaGraph, i::Int)
    vertex_props = props(mg, i)

    # linear trem coefficient
    h = vertex_props[:h]

    # quadratic

    Jir = 0.
    Jid = 0.

    for v in neighbors(mg, i)
        e = Edge(i,v)
        # bonds types are given increasing order of vertices
        p = sortperm([i,v])

        if props(mg, e)[:type][p[1]] == "r"
            Jir = props(mg, e)[:J]
        elseif props(mg, e)[:type][p[1]] == "d"
            Jid = props(mg, e)[:J]
        end
    end
    Jir, Jid, h
end

function sort2lrud(x::Vector{String})
    ret = Vector{String}()
    for i in ["l","r","u","d"]
        if i in x
            push!(ret, i)
        end
    end
    ret
end


function bond_dirs(mg::MetaGraph, i::Int)
    bond_dirs = Vector{String}()
    for v in neighbors(mg, i)
        e = Edge(i,v)
        p = sortperm([i,v])
        push!(bond_dirs, props(mg, e)[:type][p[1]])
    end
    sort2lrud(bond_dirs)
end

function get_modes(mg::MetaGraph, i::Int)
    bd = bond_dirs(mg, i)
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


function makeTensor(mg::MetaGraph, i::Int)
    modes = get_modes(mg, i)
    Js = getJs(mg, i)
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
    bd = bond_dirs(mg, vertex)
    for v in neighbors(mg, vertex)
        e = Edge(vertex,v)
        p = sortperm([vertex,v])
        dir = props(mg, e)[:type][p[1]]
        mode = findall(x->x==dir, bd)[1]
        m = props(mg, e)[:modes]
        m[p[1]] = mode
        set_prop!(mg, e, :modes, m)
    end
end


function contract_vertices(mg::MetaGraph, v1::Int, v2::Int)
    e = Edge(v1,v2)

    has_prop(mg, e, :J) || error("there is no direct link between $(v1) and $(v2)")
    tg1 = props(mg, v1)[:tensor]
    tg2 = props(mg, v2)[:tensor]

    p = sortperm([v1, v2])
    modes = props(mg, e)[:modes][p]

    tg = contract_ts1(tg1, tg2, modes[1], modes[2])

    set_prop!(mg, v1, :tensor, tg)

    rem_edge!(mg, Edge(v1, v2))
    n = collect(neighbors(mg, v2))
    for v in n

        p = sortperm([v, v2])
        m = props(mg, Edge(v, v2))[:modes]
        m_v = m[p[1]]
        m_v1new = m[p[2]] + ndims(tg1) - 1
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
    0
end
