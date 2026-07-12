using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end

function overlap_states(state1::Vector{Int}, state2::Vector{Int})
    s1 = reshape(state1, :, length(state1))
    s2 = reshape(state2, :, length(state2))
    n1 = sqrt(dot(s1, s1))
    n2 = sqrt(dot(s2, s2))
    dot(s1, s2) / (n1 * n2)
end

m, n, t = 12, 12, 1

β = 1.0
bond_dim = 2
num_states = 7 #22

ig = ising_graph("$(@__DIR__)/../instances/square_gauss/S12/001.txt")
potts_h = potts_hamiltonian(
    ig,
    spectrum = my_brute_force,
    cluster_assignment_rule = super_square_lattice((m, n, t)),
)
params = MpsParameters{Float64}(bond_dim, 1E-8, 4)
search_params = SearchParameters(num_states, 0.0)
Gauge = NoUpdate

energies = Vector{Float64}[]
Strategy = Zipper
Sparsity = Sparse
Layout = EnergyGauges
Lattice = SquareSingleNode
transform = rotation(0)

net = PEPSNetwork{Lattice{Layout},Sparsity,Float64}(m, n, potts_h, transform)
ctr = MpsContractor{Strategy,Gauge,Float64}(
    net,
    params;
    onGPU = onGPU,
    beta = β,
    graduate_truncation = true,
)
sol = gibbs_sampling(ctr, search_params)

for i = 1:num_states-1
    o = overlap_states(sol.states[i], sol.states[i+1])
    println("Overlap between states ", i, " and ", i + 1, " is ", o)
end

println(sol.energies)
#println(sol.states)
