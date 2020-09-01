using CSV
using LightGraphs
using MetaGraphs
using GraphPlot
using TensorOperations
using LinearAlgebra

function get_energy(ising::MetaGraph)::Float64
    """
    Calculate the Ising energy as E = -sum_<i,j> s_i * J_ij * s_j - sum_j h_i * s_j.
    """
    energy = 0.0
    # quadratic
    for edge in edges(ising)
        i, j = src(edge), dst(edge)   
        s = get_prop(ising, i, :s)
        r = get_prop(ising, j, :s)        
        J = get_prop(ising, i, j, :J) 
        energy += s*J*r   
    end 
    # linear
    for i in vertices(ising)
        s = get_prop(ising, i, :s)
        h = get_prop(ising, i, :h)   
        energy += s*h     
    end    
    return -energy
end
    
function create_ising(instance::String, L::Int)::MetaGraph
    """
    Create a graph that represents the Ising Hamiltonian.
    """
    # load
    df = CSV.read(instance, types=[Int, Int, Float64])
    g = MetaGraph(L^2, 0.0)
    set_prop!(g, :description, "The Ising model.")

    # setup the model (J_ij, h_i on the lattice)
    for row in eachrow(df)
        i, j, v = row
        if i == j
            set_prop!(g, i, :h, v) || error("Node $i missing!")
            set_prop!(g, i, :s, rand() < 0.5 ? -1 : 1) || error("Cannot set spin $i")
        else
            add_edge!(g, i, j) && 
            set_prop!(g, i, j, :J, v) || error("Cannot add Egde ($i, $j)") 
        end    
    end   
    # energy
    set_prop!(g, :energy, get_energy(g)) || error("Unable to calculate the Ising energy!")
    return g
end

function update_ising!(ising::MetaGraph)
    return
end

function make_clusters(model::MetaGraph)::MetaGraph
    return
end

L = 3
instance = "./lattice_$L.txt"    
ising = create_ising(instance, L)

E = get_prop(ising, :energy)
println(ising)
println("energy: $E")