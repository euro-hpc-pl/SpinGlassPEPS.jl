using LinearAlgebra
using LightGraphs
using NPZ
using SpinGlassPEPS
using Logging
using ArgParse
using CSV
using Test

function M2graph(M::Matrix{Float64}, sgn::Int = 1)
    size(M,1) == size(M,2) || error("matrix not squared")
    L = size(M,1)

    D = Dict{Tuple{Int64,Int64},Float64}()
    for j ∈ 1:size(M, 1)
        for i ∈ 1:j
            if (i == j)
                push!(D, (i,j) => M[j,i])
            elseif M[j,i] != 0.
                push!(D, (i,j) => M[i,j]+M[j,i])
            end
        end
    end
    ising_graph(D, sgn)
end

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
number_of_states = minimum([length(data["energies"][1,:]), 150])
examples = length(data["energies"][:,1])
println("examples = ", examples)


β = 3.

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
    grid_size = isqrt(size(Mat_of_interactions, 1))

    ################ exact method ###################


    fg = factor_graph(
        g,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((grid_size, grid_size, 1))
    )


    control_params = Dict(
       "bond_dim" => typemax(Int),
       "var_tol" => 1E-12,
       "sweeps" => 4.
   )

    peps = PEPSNetwork(grid_size, grid_size, fg, β, :NW, control_params)
    println("size of peps = ", peps.size)
    print("peps time  = ")

    @time sols = low_energy_spectrum(peps, number_of_states + 2)

    for i in 1:number_of_states

        @test sols.energies[i] ≈ energies_given[i]
        if states_given != 0
            f(i) = (i == 2) ? -1 : 1
            @test map(f, sols.states[i]) == states_given[i,:]
        end
    end

    ############### approximated methods  ################
    χ = 2
    χ = typemax(Int)

    control_params = Dict(
       "bond_dim" => χ,
       "var_tol" => 1E-7,
       "sweeps" => 4.
   )

    print("approx peps  ")

    peps = PEPSNetwork(grid_size, grid_size, fg, β, :NW, control_params)

    @time sols_approx = low_energy_spectrum(peps, number_of_states + 2)


    for i in 1:number_of_states

        @test sols_approx.energies[i] ≈ energies_given[i]
        if states_given != 0
            f(i) = (i == 2) ? -1 : 1
            @test map(f, sols_approx.states[i]) == states_given[i,:]
        end
    end

    print("peps larger T")
    s1 = ceil(Int, grid_size/2)

    if grid_size % 2 == 0
        rule = super_square_lattice((s1, 2, s1, 2, 1))
    else
        rule = Dict{Int, Int}(1 => 1, 2 => 1, 6 => 1, 7 => 1)
        push!(rule, (3 => 2), (4 => 2), (8 => 2), (9 => 2), (5 => 3), (10 => 3))
        push!(rule, (11 => 4), (12 => 4), (16 => 4), (17 => 4), (13 => 5), (14 => 5))
        push!(rule, (18 => 5), (19 => 5), (15 => 6), (20 => 6), (21 => 7), (22 => 7))
        push!(rule, (23 => 8), (24 => 8), (25 => 9))
    end


    fg = factor_graph(
        g,
        cluster_assignment_rule=rule,
        spectrum=brute_force,
    )
    peps = PEPSNetwork(s1, s1, fg, β, :NW, control_params)

    println("size of peps = ", peps.size)

    @time sols_large = low_energy_spectrum(peps, number_of_states + 2)

    for i in 1:number_of_states

        @test sols_large.energies[i] ≈ energies_given[i]
        #TODO we need to know how to read states from clusters
        #if states_given != 0
        #    @test sols_large.states[i] == states_given[i,:]
        #end
    end

    print("peps larger T, limited spectrum")
    D = Dict{Int, Int}()
    for v in vertices(g)
        push!(D, (v => 15))
    end
    fg = factor_graph(
        g,
        D,
        cluster_assignment_rule=rule,
        spectrum=brute_force,
    )

    peps = PEPSNetwork(s1, s1, fg, β, :NW, control_params)

    @time sols_spec = low_energy_spectrum(peps, number_of_states+2)

    for i in 1:minimum([number_of_states, 65])
        @test sols_spec.energies[i] ≈ energies_given[i]
        #TODO we need to know how to read states from clusters
        #if states_given != 0
        #    @test sols_spec.states[i] == states_given[i,:]
        #end
    end
end
