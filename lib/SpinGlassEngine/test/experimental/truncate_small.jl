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

onGPU = true

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end

function run_test(instance, m, n, t)
    β = 2
    bond_dim = 16
    δp = 1e-10
    num_states = 512

    tolV = 1E-16
    tolS = 1E-16
    max_sweeps = 1
    ITERS_SVD = 1
    ITERS_VAR = 1
    DTEMP_MULT = 2
    METHOD = :psvd_sparse

    ig = ising_graph(instance)

    params = MpsParameters{Float64}(
        bond_dim,
        tolV,
        max_sweeps,
        tolS,
        ITERS_SVD,
        ITERS_VAR,
        DTEMP_MULT,
        METHOD,
    )

    # params = MpsParameters{Float64}(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)
    energies = []
    Gauge = NoUpdate
    Strategy = Zipper
    Sparsity = Sparse
    Layout = GaugesEnergy
    cl_states = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]


    for cl in cl_states
        println("===================================")
        println("Cluster states ", cl)
        println("===================================")

        for tran ∈ all_lattice_transformations #[LatticeTransformation((1, 2, 3, 4), false),]
            println("===============")
            println("Transform ", tran)

            potts_h = potts_hamiltonian(
                ig,
                spectrum = full_spectrum, #_gpu, # rm _gpu to use CPU
                cluster_assignment_rule = pegasus_lattice((m, n, t)),
            )
            potts_h = truncate_potts_hamiltonian_2site_energy(potts_h, cl)

            net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparsity,Float64}(
                m,
                n,
                potts_h,
                tran,
            )

            ctr = MpsContractor{Strategy,Gauge,Float64}(
                net,
                params;
                onGPU = onGPU,
                beta = β,
                graduate_truncation = true,
            )

            sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr)) #, merge_branches(ctr))

            println("sol ", sol)
            clear_memoize_cache()
        end
    end
end


instance = "$(@__DIR__)/../instances/pathological/pegasus_3_4_1.txt"
m, n, t = 3, 4, 1
run_test(instance, m, n, t)
