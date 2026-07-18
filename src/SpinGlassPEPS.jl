module SpinGlassPEPS

using Reexport

include("tensors/SpinGlassTensors.jl")
include("networks/SpinGlassNetworks.jl")
include("exhaustive/SpinGlassExhaustive.jl")
include("engine/SpinGlassEngine.jl")

@reexport using .SpinGlassTensors
@reexport using .SpinGlassNetworks
@reexport using .SpinGlassExhaustive
@reexport using .SpinGlassEngine

using PrecompileTools: @setup_workload, @compile_workload

# Precompile the end-to-end CPU solve path so the first real solve in a fresh
# session does not pay full compilation for the whole type stack (the cold-start
# cost measured after the type-parameter consolidation). Uses a 3-spin instance
# (the smallest tested case) on the CPU path — no GPU, no external files.
@setup_workload begin
    inst = Dict((1, 1) => 1.0, (2, 2) => 0.75, (3, 3) => -0.5, (1, 2) => 0.25, (2, 3) => 0.6)
    @compile_workload begin
        ig = ising_graph(inst)
        ph = potts_hamiltonian(
            ig;
            spectrum = full_spectrum,
            cluster_assignment_rule = super_square_lattice((3, 1, 1)),
        )
        net = PEPSNetwork{SquareSingleNode{EnergyGauges},Dense,Float64}(
            3, 1, ph, rotation(0),
        )
        ctr = MpsContractor{SVDTruncate,NoUpdate,Float64}(
            net,
            MpsParameters{Float64}(; bond_dim = 8, num_sweeps = 1);
            onGPU = false, beta = 1.0, graduate_truncation = true,
        )
        low_energy_spectrum(ctr, SearchParameters(; max_states = 8, cutoff_prob = 0.0))
    end
end

end
