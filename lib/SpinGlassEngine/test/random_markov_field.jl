@testset "Example image is correcly loaded and computed" begin
    instance_dir = "$(@__DIR__)/instances/rmf/n4/penguin-small.h5"
    onGPU = true
    β = 1.0
    bond_dim = 8
    δp = 1E-4
    num_states = 64
    potts_h = potts_hamiltonian(instance_dir)
    Nx, Ny = get_prop(potts_h, :Nx), get_prop(potts_h, :Ny)
    params = MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = δp)
    Gauge = NoUpdate
    graduate_truncation = true
    energies = Vector{Float64}[]
    Strategy = Zipper
    Layout = GaugesEnergy
    Sparsity = Sparse
    transform = rotation(0)
    net = PEPSNetwork{KingSingleNode{Layout},Sparsity}(Nx, Ny, potts_h, transform)
    ctr = MpsContractor{Strategy,Gauge,Float64}(
        net,
        params;
        onGPU = onGPU,
        beta = β,
        graduate_truncation = true,
        mode = :RMF,
    )
    sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
end
