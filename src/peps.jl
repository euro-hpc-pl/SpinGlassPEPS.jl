using LightGraphs
using MetaGraphs
using TikzGraphs
using TensorOperations

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


VS = Vector{String}

function contract_ts(A::Array{Float64, N1} where N1, C::Array{Float64, N2} where N2, modesA::VS, modesC::VS, scheme::VS)
    mode_a = findall(x -> x == scheme[1], modesA)[1]
    mode_c = findall(x -> x == scheme[2], modesC)[1]
    iA = collect(1:ndims(A))
    iC = collect(ndims(A)+1:ndims(C)+ndims(A))
    iA[mode_a] = -1
    iC[mode_c] = -1

    modes_a = copy(modesA)
    modes_c = copy(modesC)

    deleteat!(modes_a, mode_a)
    deleteat!(modes_c, mode_c)

    T = tensorcontract(A, iA, C, iC)

    for d in ["l", "r", "u", "d"]
        if (d in modes_a) & (d in modes_c)
            modes_a = replace(modes_a, d => d*"1")
            modes_c = replace(modes_c, d => d*"2")
        end
    end

    T, vcat(modes_a, modes_c)
end

# graphical representation

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

function add_locations3x3!(graph::MetaGraph)
    a = set_prop!(graph, 1, :bonds, Dict("l" => 0, "r" => 2, "u" => 0, "d" => 6))
    b = set_prop!(graph, 2, :bonds, Dict("l" => 1, "r" => 3, "u" => 0, "d" => 5))
    c = set_prop!(graph, 3, :bonds, Dict("l" => 2, "r" => 0, "u" => 0, "d" => 4))
    d = set_prop!(graph, 4, :bonds, Dict("l" => 5, "r" => 0, "u" => 3, "d" => 9))
    e = set_prop!(graph, 5, :bonds, Dict("l" => 6, "r" => 4, "u" => 2, "d" => 8))
    f = set_prop!(graph, 6, :bonds, Dict("l" => 0, "r" => 5, "u" => 1, "d" => 7))
    g = set_prop!(graph, 7, :bonds, Dict("l" => 0, "r" => 8, "u" => 6, "d" => 0))
    h = set_prop!(graph, 8, :bonds, Dict("l" => 7, "r" => 9, "u" => 5, "d" => 0))
    i = set_prop!(graph, 9, :bonds, Dict("l" => 8, "r" => 0, "u" => 4, "d" => 0))
    a*b*c*d*e*f*g*h*i || error("vertex not in graph")
end


function make_graph3x3()
    g = make_grid3x3()
    # this will be a meta graph
    mg = MetaGraph(g)
    add_locations3x3!(mg)
    mg
end

function add_qubo2graph(mg::MetaGraph, qubo::Vector{Qubo_el})
    for q in qubo
        add_qubo_el!(mg, q)
    end
    mg
end


# generation of tensors
β = 1.

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

    # tle linear trem coefficient
    h = vertex_props[:h]

    # quadratic
    dirs_of_bonds = vertex_props[:bonds]
    Jir = 0.
    Jid = 0.

    if dirs_of_bonds["r"] != 0
        e = Edge(i, dirs_of_bonds["r"])
        Jir = props(mg, e)[:J]
    end
    if dirs_of_bonds["d"] != 0
        e = Edge(i, dirs_of_bonds["d"])
        Jid = props(mg, e)[:J]
    end
    Jir, Jid, h
end


function get_modes(mg::MetaGraph, i::Int)
    b = props(mg, i)[:bonds]
    modes = zeros(Int, 0)
    bonds_dirs =  Vector{String}()
    j = 0
    for d in ["l", "r", "u", "d"]
        j = j+1
        if b[d] != 0
            push!(modes, j)
            push!(bonds_dirs, d)
        end
    end
    bonds_dirs, modes
end


function makeTensor(mg::MetaGraph, i::Int)
    bonds_dirs, modes = get_modes(mg, i)
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
    A, vcat(bonds_dirs, "s")
end


struct TensorOnGraph
    A::Array{Float64, N} where N
    bonds_dirs::Vector{String}
    function(::Type{TensorOnGraph})(mg::MetaGraph, i::Int)
        A, bonds_dirs = makeTensor(mg, i)
        new(A, bonds_dirs)
    end
    function(::Type{TensorOnGraph})(A::Array{Float64, N} where N, bonds_dirs::Vector{String})
        ndims(A) == length(bonds_dirs) || error("not all modes described")
        new(A, bonds_dirs)
    end
end

function trace_physical_dim(tensor::TensorOnGraph)
    A = tensor.A
    bonds_dirs = tensor.bonds_dirs
    if occursin("s", bonds_dirs[end])
        A = sum_over_last(A)
        return TensorOnGraph(A, bonds_dirs[1:end-1])
    else
        error("last mode is not physical")
    end
end

function set_physical_dim(tensor::TensorOnGraph, s::Int)
    A = tensor.A
    bonds_dirs = tensor.bonds_dirs
    if occursin("s", bonds_dirs[end])
        A = set_last(A, s)
        return TensorOnGraph(A, bonds_dirs[1:end-1])
    else
        error("last mode is not physical")
    end
end

# add tensors to the graph

function add_tensor2vertex(mg::MetaGraph, vertex::Int)
    T = TensorOnGraph(mg, vertex)
    T = trace_physical_dim(T)
    set_prop!(mg, vertex, :tensor, T)
end

function add_tensor2vertex(mg::MetaGraph, vertex::Int, s::Int)
    T = TensorOnGraph(mg, vertex)
    T = set_physical_dim(T, s)
    set_prop!(mg, vertex, :tensor, T)
end
