using SpinGlassNetworks
using Graphs
using SpinGlassExhaustive
using LinearAlgebra
using CUDA
using Bits

function bench_cpu(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], num_states=max_states)
    Spectrum
    sp
end

function bench_gpu(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], :GPU; num_states=max_states)
    sp
end

println("*** CPU ***")
sp_cpu = bench_cpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
sp_cpu = bench_cpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")

println("*** GPU ***")
sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")

@assert sp_gpu.energies ≈ sp_cpu.energies
@show sp_gpu.states
@show sp_cpu.states[1]
@assert sp_gpu.states == sp_cpu.states[1]

"""
$(SIGNATURES)
- `instance::String`: path to instance of Ising model.
It generates the running time of algorithms that search the solution space of the Ising model.
"""
function bench(instance::String)
    ig = SpinGlassEngine.ising_graph(instance)
    graph = couplings(ig) + SpinGlassEngine.Diagonal(biases(ig))
    cu_graph = graph |> cu 	

    println("Graph size ", size(cu_graph,1))

    # Setting SpinGlassNetworks - brute force
    println("Benchmark time of Brute force")
    @btime begin
        res_naive = SpinGlassNetworks.brute_force(ig)
    end
    
    # Setting SpinGlassEnginge
    nrows = 4
    ncols = 4

    println("Benchmark time of PEPSNetwork")
    @btime begin
        fg = factor_graph(
            ig, cluster_assignment_rule=super_square_lattice((nrows, ncols, 14))
        )
        β = 1

        network = SpinGlassEngine.PEPSNetwork{true}(
            nrows,
            ncols,
            fg,
            rotation(180),
            β=β,
            bond_dim=24
        )
        
        res_peps = low_energy_spectrum(network, 24)
    end
    
    println("Benchmark time of Matrix Product states")
    max_states = size(ig)[1]*size(ig)[1]
    to_show = size(ig)[1]*size(ig)[1]

    β = 2.
    dβ = β/8.0

    Dcut = 256
    var_ϵ = 1E-8
    max_sweeps = 4

    @btime begin
        igp = prune(ig) 
        schedule = fill(dβ, Int(ceil(β/dβ)))


        ψ = SpinGlassEngine.MPS(igp, Dcut, var_ϵ, max_sweeps, schedule)

        states, lprob, _ = solve(ψ, max_states)

        sort(SpinGlassEngine.energy.(states[1:to_show], Ref(igp)))
    end
    
    println("Benchmark time of Coherent Ising Machine simulator")
    @btime begin
        L = size(ig)[1]

        scale = 0.7
        noise = Normal(0.1, 0.3)

        x0 = 2.0 .* rand(L) .- 1.0
        sat = 1.0
        time = 1000.
        pi, pf, α = -5.0, 0.0, 2.0
        momentum = 0.9

        pump = [ramp(t, time, α, pi, pf) for t ∈ 1:time]

        opo = OpticalOscillators{Float64}(ig, scale, noise)
        dyn = OPODynamics{Float64}(x0, sat, pump, momentum)

        pump_nmfa = [ramp(t, time, α, 10., 0.01) for t ∈ 1:time]

        opo_nmfa = OpticalOscillators{Float64}(ig, scale, noise)
        dyn_nmfa = OPODynamics{Float64}(x0, sat, pump_nmfa, momentum)

        N = 500
        states = Vector{Vector{Int}}(undef, N)
        states_nmfa = copy(states)
        Threads.@threads for i ∈ 1:N
            states[i] = evolve_optical_oscillators(opo, dyn)
            states_nmfa[i] = noisy_mean_field_annealing(opo_nmfa, dyn_nmfa)
        end


        en = minimum(SpinGlassDynamics.energy.(states, Ref(ig)))
        en_nmfa = minimum(SpinGlassDynamics.energy.(states_nmfa, Ref(ig)))
    end

    println("Benchmark time of Spin Glass Exhaustive")
    @btime begin
        res_sge = SpinGlassExhaustive.exhaustive_search(ig)
    end

    
    println("Benchmark time of Spin Glass Exhaustive with bucket selection")
    @btime begin
        res_sge = SpinGlassExhaustive.exhaustive_search_bucket(ig)
    end
end