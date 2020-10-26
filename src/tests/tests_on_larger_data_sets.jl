using NPZ
using Plots
using Test
using ArgParse

include("../notation.jl")
include("../brute_force.jl")
include("../compression.jl")
include("../peps_no_types.jl")
include("../mps_implementation.jl")


s = ArgParseSettings("description")
  @add_arg_table! s begin
    "file"
    help = "the file name"
    arg_type = String
  end
file = parse_args(s)["file"]
println(file)

data = npzread(file)
number_of_all_states = length(data["energies"][1,:])
examples = length(data["energies"][:,1])
println("examples = ", examples)


β = 3.

number_of_states = 10
#examples = 100
more_states_for_mps = 2
#degeneracy = false
if number_of_all_states > 150
    number_of_states = 150
    more_states_for_mps = 20
end

if false
if occursin("examples.npz", file)
    number_of_states = 10
    more_states_for_mps = 0
    examples = 100
    # here only energies will be tested
    println("degeneracy")
    #degeneracy = true
end


if occursin("examples2.npz", file)
    number_of_states = 10
    more_states_for_mps = 0
    examples = 100
end

if occursin("energies_and_matrix_only.npz", file)
    number_of_states = 150
    more_states_for_mps = 20
    examples = 1
end

if occursin("examples_full_L=6.npz", file)
    number_of_states = 150
    more_states_for_mps = 20
    examples = 1
end
end

# the type of data can be changed if someone wishes
T = Float64

for k in 1:examples

    println(" ..................... SAMPLE =  ", k, "  ...................")
    QM = T.(data["Js"][k,:,:])
    degeneracy = (0. in diag(QM))
    println(degeneracy)

    states_given = 0
    try
        if !degeneracy
            states_given = data["states"][k,:,:]
        end
    catch
        0
    end


    energies_given = data["energies"][k,:,:]

    interactions = M2interactions(QM)

    grid = nxmgrid(5,5)
    if size(QM, 1) == 36
        grid = nxmgrid(6,6)
    end
    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]

    ################ exact method ###################

    print("peps time  = ")
    @time spins, objective = solve(interactions, ns, grid, number_of_states ; β = T(β), χ = 0, threshold = T(0.))

    for i in 1:number_of_states

        @test v2energy(QM, spins[i]) ≈ -energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins[i]
        end
    end

    ############### approximated methods  ################
    χ = 2
    print("approx peps  ")

    @time spins_approx, objective_approx = solve(interactions, ns, grid, number_of_states; β = T(β), χ = χ, threshold = T(1e-12))

    for i in 1:number_of_states

        @test v2energy(QM, spins_approx[i]) ≈ -energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_approx[i]
        end
    end

    @test objective ≈ objective_approx atol = 1.e-10

    nodes_numbers = nxmgrid(3,3)

    grid = Array{Array{Int}}(undef, (3,3))
    grid[1,1] = [1 2; 6 7]
    grid[1,2] = [3 4; 8 9]
    grid[1,3] = reshape([5; 10], (2,1))
    grid[2,1] = [11 12; 16 17]
    grid[2,2] = [13 14; 18 19]
    grid[2,3] = reshape([15; 20], (2,1))
    grid[3,1] = reshape([21; 22], (1,2))
    grid[3,2] = reshape([23; 24], (1,2))
    grid[3,3] = reshape([25], (1,1))

    if size(QM, 1) == 36
        grid[1,1] = [1 2; 7 8]
        grid[1,2] = [3 4; 9 10]
        grid[1,3] = [5 6; 11 12]
        grid[2,1] = [13 14; 19 20]
        grid[2,2] = [15 16; 21 22]
        grid[2,3] = [17 18; 23 24]
        grid[3,1] = [25 26; 31 32]
        grid[3,2] = [27 28; 33 34]
        grid[3,3] = [29 30; 35 36]
    end

    grid = Array{Array{Int}}(grid)

    ns_l = [Node_of_grid(i, nodes_numbers, grid) for i in 1:9]

    print("peps larger T")

    @time spins_larger_nodes, objective_larger_nodes = solve(interactions, ns_l, nodes_numbers, number_of_states; β = T(β), χ = χ, threshold = T(1e-12))

    for i in 1:number_of_states

        @test v2energy(QM, spins_larger_nodes[i]) ≈ -energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_larger_nodes[i]
        end
    end

    @test objective ≈ objective_larger_nodes atol = 1.e-10

    ############ MPO - MPS #########
    χ = 14

    β_step = 4

    print("mps time  =  ")
    ns = [Node_of_grid(i, interactions) for i in 1:get_system_size(interactions)]

    number = number_of_states + more_states_for_mps
    @time spins_mps, objective_mps = solve_mps(interactions, ns, number ; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

    # sorting improves the oputput
    energies_mps = [-v2energy(QM, spins) for spins in spins_mps]
    p = sortperm(energies_mps)

    spins_mps = spins_mps[p]
    energies_mps = energies_mps[p]

    for i in 1:number_of_states

        @test v2energy(QM, spins_mps[i]) ≈ -energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_mps[i]
        end
    end


end
