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

m, n, t = 8, 8, 8

β = 1.0
Dcut = 7
tolV = 0.01
tolS = 0.0
max_sweeps = 4

ig = ising_graph("$(@__DIR__)/../instances/chimera_droplets/512power/001.txt")

potts_h = potts_hamiltonian(
    ig,
    spectrum = my_brute_force, #rm _gpu to use CPU
    cluster_assignment_rule = super_square_lattice((m, n, t)),
)

params = MpsParameters{Float64}(Dcut, tolV, max_sweeps)

Strategy = SVDTruncate
tran = LatticeTransformation((1, 2, 3, 4), false)
Layout = EnergyGauges
Gauge = NoUpdate

i = div(m, 2)

net = PEPSNetwork{SquareSingleNode{Layout},Sparse,Float64}(m, n, potts_h, tran)
ctr = MpsContractor{Strategy,Gauge,Float64}(
    net,
    params;
    onGPU = onGPU,
    beta = β,
    graduate_truncation = true,
)
Ws = SpinGlassEngine.mpo(ctr, ctr.layers.main, i)
println(" Ws -> ", which_device(Ws), " ", format_bytes.(measure_memory(Ws)))

net = PEPSNetwork{SquareSingleNode{Layout},Dense,Float64}(m, n, potts_h, tran)
ctr = MpsContractor{Strategy,Gauge,Float64}(
    net,
    params;
    onGPU = onGPU,
    beta = β,
    graduate_truncation = true,
)
Wd = SpinGlassEngine.mpo(ctr, ctr.layers.main, i)
println(" Wd -> ", which_device(Wd), " ", format_bytes.(measure_memory(Wd)))

println(
    "Dcut = ",
    Dcut,
    " tolV = ",
    tolV,
    " tolS = ",
    tolS,
    " max_sweeps = ",
    max_sweeps,
    " i = ",
    i,
)

ψ = rand(QMps{Float64}, local_dims(Wd, :down), Dcut; onGPU = onGPU) # F64 for now
println(" ψ -> ", which_device(ψ), " ", format_bytes.(measure_memory(ψ)))
canonise!(ψ, :left)

for (W, msg) ∈ [(Ws, "SPARSE"), (Wd, "DENSE")] #
    println(msg)
    println("dot")
    ψ0 = dot(W, ψ)
    println(" ψ0 -> ", which_device(ψ0), " ", format_bytes.(measure_memory(ψ0)))
    println(bond_dimensions(ψ0))

    println("canonize_truncate!")
    ψ1 = dot(W, ψ)
    canonise!(ψ1, :left)
    canonise_truncate!(ψ1, :right, Dcut, tolS)
    println(" ψ0 -> ", which_device(ψ1), " ", format_bytes.(measure_memory(ψ)))
    println(dot(ψ0, ψ1) / (norm(ψ0) * norm(ψ1)), "  ", dot(ψ0, ψ1) / norm(ψ0))
    canonise!(ψ1, :left)

    println("zipper dense svd")
    ψ2 = zipper(
        W,
        ψ,
        method = :svd,
        Dcut = Dcut,
        tol = tolS,
        iters_svd = 1,
        iters_var = 1,
        Dtemp_multiplier = 2,
    )
    println(dot(ψ0, ψ2) / (norm(ψ0) * norm(ψ2)), "  ", dot(ψ0, ψ2) / norm(ψ0))
    canonise!(ψ2, :left)

    println("zipper psvd")
    ψ3 = zipper(
        W,
        ψ,
        method = :psvd,
        Dcut = Dcut,
        tol = tolS,
        iters_svd = 1,
        iters_var = 1,
        Dtemp_multiplier = 2,
    )
    println(dot(ψ0, ψ3) / (norm(ψ0) * norm(ψ3)), "  ", dot(ψ0, ψ3) / norm(ψ0))
    canonise!(ψ3, :left)

    println("zipper psvd_sparse")
    ψ3 = zipper(
        W,
        ψ,
        method = :psvd_sparse,
        Dcut = Dcut,
        tol = tolS,
        iters_svd = 1,
        iters_var = 1,
        Dtemp_multiplier = 2,
    )
    println(dot(ψ0, ψ3) / (norm(ψ0) * norm(ψ3)), "  ", dot(ψ0, ψ3) / norm(ψ0))
    canonise!(ψ3, :left)

    println("variational compressions")
    overlap, env = variational_compress!(ψ1, W, ψ, tolV, max_sweeps)
    println(dot(ψ0, ψ1) / (norm(ψ0) * norm(ψ1)))
    println(dot(ψ0, ψ1), "   vs   ", exp(overlap))

    overlap, env = variational_compress!(ψ2, W, ψ, tolV, max_sweeps)
    println(dot(ψ0, ψ2) / (norm(ψ0) * norm(ψ2)))
    println(dot(ψ0, ψ2), "   vs   ", exp(overlap))

    overlap, env = variational_compress!(ψ3, W, ψ, tolV, max_sweeps)
    println(dot(ψ0, ψ3) / (norm(ψ0) * norm(ψ3)))
    println(dot(ψ0, ψ3), "   vs   ", exp(overlap))

    # overlap, env = variational_compress!(ψ4, W, ψ, tolV, max_sweeps)
    # println(dot(ψ0, ψ4) / (norm(ψ0) * norm(ψ4)))
end

# @time ψ4 = zipper(Ws, ψ, method=:psvd, Dcut=Dcut, tol=tolS)
# @time ψ7 = zipper(Wd, ψ, method=:tsvd, Dcut=Dcut, tol=tolS) #, maxiter=Dcut+1, tolconv=100, tolreorth=100)
# @time ψ8 = zipper(Ws, ψ, method=:tsvd, Dcut=Dcut, tol=tolS) #, maxiter=Dcut+1, tolconv=100, tolreorth=100)
# @time ψ9 = zipper(Wd, ψ, method=:tsvd_sparse, Dcut=Dcut, tol=tolS) #, maxiter=Dcut+1, tolconv=100, tolreorth=100)
# @time ψ10 = zipper(Ws, ψ, method=:tsvd_sparse, Dcut=Dcut, tol=tolS) #, maxiter=Dcut+1, tolconv=100, tolreorth=100)

# println(dot(ψ0, ψ0))
# println(dot(ψ1, ψ1))
# println(dot(ψ2, ψ2))

# println(dot(ψ0, ψ1) / (norm(ψ0) * norm(ψ1)))
# println(dot(ψ0, ψ2) / (norm(ψ0) * norm(ψ2)))
# println(dot(ψ1, ψ2) / (norm(ψ1) * norm(ψ2)))
# println(dot(ψ1, ψ3) / (norm(ψ1) * norm(ψ3)))
# println(dot(ψ1, ψ4) / (norm(ψ1) * norm(ψ4)))
# println(dot(ψ1, ψ5) / (norm(ψ1) * norm(ψ5)))
# println(dot(ψ1, ψ6) / (norm(ψ1) * norm(ψ6)))
# # println(dot(ψ1, ψ7) / (norm(ψ1) * norm(ψ7)))
# # println(dot(ψ1, ψ8) / (norm(ψ1) * norm(ψ8)))
# # println(dot(ψ1, ψ9) / (norm(ψ1) * norm(ψ9)))
# # println(dot(ψ1, ψ10) / (norm(ψ1) * norm(ψ10)))


# println(" ψ0 -> ", format_bytes.(measure_memory(ψ0)))
