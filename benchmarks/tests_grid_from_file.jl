using LinearAlgebra
using LightGraphs
using MetaGraphs
using NPZ

using SpinGlassPEPS

using Logging
using ArgParse
using CSV
using Test

import SpinGlassPEPS: solve, solve_mps, M2graph

disable_logging(LogLevel(0))


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
δH = 0.

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

    g = M2graph(Mat_of_interactions, -1)
    si = Int(sqrt(size(Mat_of_interactions, 1)))

    ################ exact method ###################
    fg = factor_graph(
        g,
        2,
        energy=energy,
        spectrum=brute_force,
    )

    peps = PepsNetwork(si, si, fg, β, :NW)
    println("size of peps = ", peps.size)
    print("peps time  = ")


    number = number_of_states + more_states_for_peps
    @time sols = solve(peps, number ; β = T(β), threshold = 0., δH = δH)
    objective = [e.objective for e in sols]
    spins = return_solution(g, fg, sols)

    for i in 1:number_of_states

        @test energy(spins[i], g) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins[i]
        end
    end

    ############### approximated methods  ################
    χ = 2
    print("approx peps  ")

    number = number_of_states + more_states_for_peps
    @time sols = solve(peps, number; β = T(β), χ = χ, threshold = 1e-12, δH = δH)

    objective_approx = [e.objective for e in sols]
    spins_approx = return_solution(g, fg, sols)

    for i in 1:number_of_states

        @test energy(spins_approx[i], g) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_approx[i]
        end
    end

    @test objective ≈ objective_approx atol = 1.e-7

    print("peps larger T")
    s1 = ceil(Int, si/2)
    s2 = floor(Int, si/2)
    println(s1)

    if s1 == s2
        rule = square_lattice((s1, 2, s1, 2, 1))
    else
        D1 = Dict{Any,Any}(1 => 1, 2 => 1, 6 => 1, 7 => 1)
        D2 = Dict{Any,Any}(3 => 2, 4 => 2, 8 => 2, 9 => 2)
        D3 = Dict{Any,Any}(5 => 3, 10 => 3)
        D4 = Dict{Any,Any}(11 => 4, 12 => 4, 16 => 4, 17 => 4)
        D5 = Dict{Any,Any}(13 => 5, 14 => 5, 18 => 5, 19 => 5)
        D6 = Dict{Any,Any}(15 => 6, 20 => 6)
        D7 = Dict{Any,Any}(21 => 7, 22 => 7)
        D8 = Dict{Any,Any}(23 => 8, 24 => 8)
        D9 = Dict{Any,Any}(25 => 9)

        rule = merge(D1, D2, D3, D4, D5, D6, D7, D8, D9)
    end

    update_cells!(
      g,
      rule = rule,
    )

    fg = factor_graph(
        g,
        16,
        energy=energy,
        spectrum=brute_force,
    )

    peps = PepsNetwork(s1, s1, fg, β, :NW)
    println("size of peps = ", peps.size)

    number = number_of_states + more_states_for_peps
    @time sols = solve(peps, number; node_size = (2,2), β = T(β), χ = χ, threshold = 1e-12, δH = δH)
    objective_larger_nodes = [e.objective for e in sols]
    spins_larger_nodes = return_solution(g, fg, sols)

    for i in 1:number_of_states

        @test energy(spins_larger_nodes[i], g) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_larger_nodes[i]
        end
    end

    @test objective ≈ objective_larger_nodes atol = 1.e-7

    print("peps larger T, limited spectrum")
    fg = factor_graph(
        g,
        15,
        energy=energy,
        spectrum=brute_force,
    )

    peps = PepsNetwork(s1, s1, fg, β, :NW)
    number = number_of_states + more_states_for_peps
    @time sols = solve(peps, number; node_size = (2,2), β = T(β), χ = χ, threshold = 1e-12, δH = δH)

    objective_spec = [e.objective for e in sols]
    spins_spec = return_solution(g, fg, sols)

    for i in 1:minimum([number_of_states, 60])
        @test energy(spins_spec[i], g) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_spec[i]
        end
    end

    ############ MPO - MPS #########
    χ = 15

    β_step = 4

    print("mps time  =  ")

    number = number_of_states + more_states_for_mps

    @time spins_mps, objective_mps = solve_mps(g, number ; β=β, β_step=β_step, χ=χ, threshold = 1.e-14)

    # sorting improves the oputput
    energies_mps = [energy(spins, g) for spins in spins_mps]
    p = sortperm(energies_mps)

    spins_mps = spins_mps[p]
    energies_mps = energies_mps[p]

    for i in 1:number_of_states

        @test energy(spins_mps[i], g) ≈ energies_given[i]

        if states_given != 0
            @test states_given[i,:] == spins_mps[i]
        end
    end
end
