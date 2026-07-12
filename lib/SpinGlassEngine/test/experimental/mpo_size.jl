using SpinGlassExhaustive
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using Graphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics
using LowRankApprox
using CUDA

disable_logging(LogLevel(1))
CUDA.allowscalar(false)


function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end


onGPU = true

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end

m, n, t = 7, 7, 3

Dcut = 8
β = 1.0
tolV = 1E-16
tolS = 1E-16
max_sweeps = 1
ITERS_SVD = 1
ITERS_VAR = 1
DTEMP_MULT = 2
MAX_STATES = 128
METHOD = :psvd_sparse #:psvd_sparse #:svd
DE = 16.0
δp = 1E-5 * exp(-β * DE)
cluster_states = [2^4, 2^8, 2^12, 2^16, 2^20]

ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P8/CBFM-P/SpinGlass/001_sg.txt")

for cl_states in cluster_states
    println("====================")

    println("cluster states: ", cl_states)
    # potts_h = potts_hamiltonian(
    #     ig,
    #     cl_states,
    #     spectrum=my_brute_force, #rm _gpu to use CPU
    #     cluster_assignment_rule=pegasus_lattice((m, n, t))
    # )

    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum, #rm _gpu to use CPU
        cluster_assignment_rule = pegasus_lattice((m, n, t)),
    )

    potts_h = truncate_potts_hamiltonian_2site_energy(potts_h, cl_states)

    params = MpsParameters{Float64}(
        Dcut,
        tolV,
        max_sweeps,
        tolS,
        ITERS_SVD,
        ITERS_VAR,
        DTEMP_MULT,
        METHOD,
    )
    search_params = SearchParameters(MAX_STATES, δp)

    Strategy = Zipper
    tran = LatticeTransformation((1, 2, 3, 4), false)
    Layout = GaugesEnergy
    Gauge = NoUpdate

    i = div(m, 2)

    net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparse,Float64}(m, n, potts_h, tran)
    ctr = MpsContractor{Strategy,Gauge,Float64}(
        net,
        params;
        onGPU = onGPU,
        beta = β,
        graduate_truncation = true,
    )
    Ws = SpinGlassEngine.mpo(ctr, ctr.layers.main, i)
    # println(" Ws -> ", which_device(Ws), " ", format_bytes.(measure_memory(Ws)))
    # println(ctr.layers.main)
    site = Ws[3].ctr
    virtual = Ws[5//2].ctr
    println("SITE TENSOR ")
    println("loc exp ", size(site.loc_exp))
    println("dims ", site.dims)
    # println("projs ", size(site.projs))

    println("VIRTUAL TENSOR ")
    # println("con ", size(virtual.con))
    println("dims ", virtual.dims)

    # println("projs ", size(virtual.projs))
end
