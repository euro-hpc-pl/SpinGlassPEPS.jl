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


β = 4.

number_of_states = 10

more_states_for_peps = 2
more_states_for_mps = 10

if number_of_all_states > 150
    number_of_states = 150
    more_states_for_mps = 20
end


# the type of data can be changed if someone wishes
T = Float64

for k in 1:examples

    println(" ..................... SAMPLE =  ", k, "  ...................")
    Mat_of_interactions = data["Js"][k,:,:]
    degeneracy = (0. in diag(Mat_of_interactions))

    states_given = 0
    try
        if !degeneracy
            states_given = data["states"][k,:,:]
        end
    catch
        0
    end

    energies_given = data["energies"][k,:,:]

    g = M2graph(Mat_of_interactions)

    ################ exact method ###################

    print("peps time  = ")


    number = number_of_states + more_states_for_peps
    @time spins, objective = solve(g, number ; β = T(β), χ = 0, threshold = 0.)

    for i in 1:number_of_states

        @test v2energy(Mat_of_interactions, spins[i]) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins[i]
        end
    end

    ############### approximated methods  ################
    χ = 2
    print("approx peps  ")

    number = number_of_states + more_states_for_peps
    @time spins_approx, objective_approx = solve(g, number; β = T(β), χ = χ, threshold = 1e-12)

    for i in 1:number_of_states

        @test v2energy(Mat_of_interactions, spins_approx[i]) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_approx[i]
        end
    end

    @test objective ≈ objective_approx atol = 1.e-7

    print("peps larger T")
    number = number_of_states + more_states_for_peps
    @time spins_larger_nodes, objective_larger_nodes = solve(g, number; node_size = (2,2), β = T(β), χ = χ, threshold = 1e-12)

    for i in 1:number_of_states

        @test v2energy(Mat_of_interactions, spins_larger_nodes[i]) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_larger_nodes[i]
        end
    end

    @test objective ≈ objective_larger_nodes atol = 1.e-7

    ############ MPO - MPS #########
    χ = 14

    β_step = 4

    print("mps time  =  ")

    number = number_of_states + more_states_for_mps

    @time spins_mps, objective_mps = solve_mps(g, number ; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

    # sorting improves the oputput
    energies_mps = [v2energy(Mat_of_interactions, spins) for spins in spins_mps]
    p = sortperm(energies_mps)

    spins_mps = spins_mps[p]
    energies_mps = energies_mps[p]

    for i in 1:number_of_states

        @test v2energy(Mat_of_interactions, spins_mps[i]) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_mps[i]
        end
    end
end
