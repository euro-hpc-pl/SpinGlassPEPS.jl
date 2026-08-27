instance_dir = "$(@__DIR__)/instances/pegasus/"
instances = ["P2"] #, "P4", "P8", "P16"]

@testset verbose = true "Renumerated instances generate correct Potts Hamiltonian" begin
    size = [2, 4, 8, 16]

    @testset "$instance" for (i, instance) ∈ enumerate(instances)
        instance = instance * ".txt"
        s = size[i] - 1
        m, n, t = s, s, 24
        max_cl_states = 2

        ig = ising_graph(joinpath(instance_dir, instance))
        potts_h = potts_hamiltonian(
            ig,
            max_cl_states,
            spectrum = brute_force,
            cluster_assignment_rule = super_square_lattice((m, n, t)),
        )
        @test nv(potts_h) == s^2

        if s > 1
            @test all(has_edge(potts_h, (l, k), (l + 1, k - 1)) for l ∈ 1:s-1, k ∈ 2:s)
        end
    end
end
