function bench_cpu(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], num_states=max_states)
    sp
end


# function bench_gpu(instance::String, max_states::Int=100)
#     m = 2
#     n = 2
#     t = 24

#     ig = ising_graph(instance)
#     cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
#     @time sp = brute_force_gpu(cl[1, 1], num_states=max_states)
#     sp
# end

sp_cpu = bench_cpu("$(@__DIR__)/2_2_3_00.txt");
# sp_gpu = bench_gpu("$(@__DIR__)/2_2_3_00.txt");

# @testset "Brute Force search: CPU vs GPU" begin

#     @test sp_gpu.energies ≈ sp_cpu.energies
#     @test all(
#         sp_gpu.states[:, i] == sp_cpu.states[i]
#         for i ∈ 1:size(sp_cpu.states, 1)
#     )
# end