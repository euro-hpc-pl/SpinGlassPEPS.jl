using SpinGlassPEPS
using CUDA

function bench_cpu(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], num_states=max_states)
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

if CUDA.functional()
    println("*** GPU ***")
    sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
    sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")

    @assert sp_gpu.energies ≈ sp_cpu.energies
    @show sp_gpu.states
    @show sp_cpu.states[1]
    @assert sp_gpu.states == sp_cpu.states[1]
else
    @info "CUDA is not functional; skipping the GPU exhaustive benchmark"
end
