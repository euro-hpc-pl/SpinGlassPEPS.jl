using SpinGlassExhaustive

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end

function check_ground_state(potts_h)
    ground_state = [
        982,
        751,
        433,
        421,
        659,
        290,
        577,
        144,
        531,
        554,
        217,
        432,
        875,
        964,
        434,
        180,
        167,
        303,
    ]
    gs = Dict(
        56 => 1,
        35 => 1,
        60 => -1,
        67 => 1,
        215 => 1,
        73 => 1,
        115 => 1,
        112 => 1,
        185 => 1,
        86 => 1,
        168 => 1,
        207 => 1,
        183 => -1,
        177 => -1,
        12 => 1,
        75 => 1,
        23 => 1,
        111 => 1,
        41 => 1,
        68 => 1,
        82 => -1,
        130 => -1,
        125 => 1,
        77 => 1,
        172 => 1,
        71 => 1,
        66 => -1,
        103 => -1,
        59 => -1,
        208 => 1,
        26 => -1,
        211 => 1,
        127 => 1,
        116 => -1,
        100 => 1,
        79 => -1,
        195 => 1,
        141 => -1,
        135 => 1,
        138 => -1,
        107 => 1,
        46 => 1,
        57 => 1,
        152 => -1,
        170 => 1,
        129 => 1,
        78 => -1,
        133 => 1,
        72 => -1,
        184 => -1,
        1 => -1,
        137 => 1,
        22 => -1,
        154 => 1,
        206 => 1,
        33 => -1,
        40 => -1,
        113 => 1,
        165 => -1,
        142 => 1,
        5 => -1,
        55 => -1,
        114 => 1,
        136 => -1,
        117 => 1,
        45 => -1,
        145 => -1,
        158 => 1,
        176 => 1,
        28 => -1,
        148 => -1,
        92 => 1,
        36 => -1,
        118 => -1,
        162 => -1,
        84 => -1,
        7 => -1,
        25 => 1,
        95 => -1,
        203 => 1,
        93 => -1,
        18 => 1,
        147 => 1,
        157 => 1,
        16 => 1,
        19 => -1,
        44 => 1,
        31 => 1,
        146 => -1,
        74 => -1,
        61 => 1,
        29 => -1,
        212 => -1,
        159 => -1,
        193 => -1,
        101 => -1,
        105 => 1,
        17 => 1,
        166 => 1,
        89 => 1,
        198 => 1,
        214 => -1,
        80 => -1,
        51 => 1,
        143 => 1,
        48 => -1,
        15 => -1,
        97 => -1,
        134 => -1,
        110 => -1,
        30 => -1,
        6 => 1,
        182 => -1,
        164 => 1,
        153 => 1,
        186 => 1,
        64 => 1,
        90 => -1,
        139 => -1,
        4 => -1,
        13 => -1,
        104 => 1,
        52 => 1,
        179 => 1,
        43 => -1,
        11 => 1,
        69 => -1,
        171 => -1,
        85 => -1,
        119 => -1,
        39 => -1,
        216 => -1,
        126 => -1,
        108 => -1,
        156 => -1,
        2 => -1,
        10 => -1,
        27 => 1,
        124 => 1,
        144 => -1,
        200 => -1,
        20 => 1,
        81 => 1,
        187 => 1,
        213 => -1,
        9 => 1,
        189 => -1,
        109 => 1,
        161 => 1,
        88 => -1,
        209 => -1,
        120 => 1,
        24 => 1,
        8 => -1,
        37 => -1,
        83 => -1,
        190 => 1,
        201 => -1,
        99 => 1,
        121 => 1,
        14 => 1,
        174 => -1,
        123 => 1,
        32 => 1,
        197 => 1,
        196 => -1,
        210 => -1,
        151 => 1,
        54 => -1,
        63 => 1,
        191 => -1,
        91 => -1,
        62 => -1,
        205 => 1,
        150 => 1,
        122 => 1,
        58 => -1,
        199 => 1,
        173 => 1,
        188 => -1,
        98 => -1,
        204 => -1,
        76 => -1,
        34 => 1,
        50 => 1,
        194 => -1,
        167 => 1,
        42 => -1,
        87 => 1,
        132 => 1,
        140 => -1,
        202 => -1,
        169 => 1,
        180 => -1,
        160 => 1,
        49 => -1,
        106 => 1,
        94 => 1,
        102 => 1,
        128 => -1,
        70 => -1,
        21 => 1,
        38 => 1,
        163 => 1,
        131 => -1,
        192 => -1,
        53 => 1,
        47 => 1,
        175 => 1,
        3 => -1,
        178 => -1,
        96 => -1,
        149 => 1,
        155 => 1,
        181 => -1,
        65 => -1,
    )
    decoded_states = zeros(18)
    for (i, node) in enumerate(vertices(potts_h))
        node_states = get_prop(potts_h, node, :spectrum).states
        spins = get_prop(potts_h, node, :cluster).labels
        decoded_from_ig = [gs[key] for key in spins]

        if decoded_from_ig in node_states
            decoded_states[i] = 1
        end
    end
    println(decoded_states)
end


m = 3
n = 3
t = 3

β = 0.5
bond_dim = 12
DE = 16.0
δp = 0 #1E-5*exp(-β * DE)
num_states = 256

VAR_TOL = 1E-16
MS = 0
TOL_SVD = 1E-16
ITERS_SVD = 1
ITERS_VAR = 1
DTEMP_MULT = 2
iter = 2
cs = 2^16
eng = 7
hamming_dist = 14
inst = "006"
ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P4/RAU/SpinGlass/006_sg.txt")
results_folder = "$(@__DIR__)/../instances/pegasus_random/P4/RAU/SpinGlass/BP_idx"
if !Base.Filesystem.isdir(results_folder)
    Base.Filesystem.mkpath(results_folder)
end

potts_h = potts_hamiltonian(
    ig,
    spectrum = full_spectrum, #rm _gpu to use CPU
    cluster_assignment_rule = pegasus_lattice((m, n, t)),
)
potts_h = truncate_potts_hamiltonian(
    potts_h,
    β,
    cs,
    results_folder,
    inst;
    tol = 1e-6,
    iter = iter,
)
check_ground_state(potts_h)
params =
    MpsParameters{Float64}(bond_dim, VAR_TOL, MS, TOL_SVD, ITERS_SVD, ITERS_VAR, DTEMP_MULT)
search_params = SearchParameters(num_states, δp)
energies = Vector{Float64}[]
Strategy = Zipper # SVDTruncate
Sparsity = Sparse #Dense
Layout = GaugesEnergy
Gauge = NoUpdate

for tran ∈ all_lattice_transformations #[LatticeTransformation((1, 2, 3, 4), false), ]
    net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparsity,Float64}(m, n, potts_h, tran)
    ctr = MpsContractor{Strategy,Gauge,Float64}(
        net,
        params;
        onGPU = onGPU,
        beta = β,
        graduate_truncation = true,
    )
    sol, s = low_energy_spectrum(
        ctr,
        search_params,
        merge_branches(
            ctr;
            merge_prob = :none,
            droplets_encoding = SingleLayerDroplets(;
                max_energy = eng,
                min_size = hamming_dist,
                metric = :hamming,
            ),
        ),
    )
    println(sol.energies)
    # println(sol.states)
    ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
    # println(ig_states)
    clear_memoize_cache()
end
