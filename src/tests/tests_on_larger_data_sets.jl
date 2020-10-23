using NPZ
using Plots
using Test

include("../notation.jl")
include("../brute_force.jl")
include("../compression.jl")
include("../peps_no_types.jl")
include("../mps_implementation.jl")


β = 3.
file = "examples.npz"
j = 10
examples = 100
# calculates r-times more solutions to avoid the droplet problem
r=2

if false
    file = "examples2.npz"
    j = 10
    examples = 100
end

if false
file = "energies_and_matrix_only.npz"
    j = 25
    examples = 1
end

# the type can be changed if someone wishes
T = Float64

data = npzread("./data/"*file)
println(file)


for k in 1:examples
    println(" ..................... SAMPLE =  ", k, "  ...................")
    QM = T.(data["Js"][k,:,:])

    states = 0
    try
        states = data["states"][k,:,:]
    catch
        0
    end
    ens = data["energies"][k,:,:]

    interactions = M2interactions(QM)

    grid = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25]
    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]

    print("peps time  = ")
    @time spins, objective = solve(interactions, ns, grid, r*j; β = T(β), χ = 0, threshold = T(0.))

    count = copy(j)
    for i in 1:j

        if !(v2energy(QM, spins[i]) ≈ -ens[i])
            println("... peps exact ...")
            println("n.o. state = ", i)
            println("energies (peps, bf)", (v2energy(QM, spins[i]), -ens[i]))
            count = count - 1

            try
                println(Int.(states[i,:]))
                println(spins[i])
            catch
                0
            end
        end
    end

    if count != j
        println(" xxxxxxx Peps exact No. matching energies $(count), should be $j xxxxxxxx")
    end

    # plotting spectrum if requires
    plot_spectrum = false

    if (k == 1) && plot_spectrum
        energies_exact = ens

        ps = exp.(-energies_exact*β)

        y = [v2energy(QM, spins[i]) for i in 1:j]
        y = exp.(y*β)
        p_theoretical = ps./sum(ps)
        A = [p_theoretical, y./sum(y), objective]

        plot(A, label = ["bf" "peps M" "peps"], yaxis = :log)
        savefig("./pics/$(file)_$(k)_myplot.pdf")
    end

    χ = 2
    print("approx peps  ")
    @time spins_a, objective_a = solve(interactions, ns, grid, r*j; β = T(β), χ = χ, threshold = T(1e-10))

    count_a = copy(j)
    for i in 1:j

        if !(v2energy(QM, spins_a[i]) ≈ -ens[i])
            println("... pepse approx ...")
            println("n.o. state = ", i)
            println("energies (peps,bf)", (v2energy(QM, spins_a[i]), -ens[i]))
            count_a = count_a - 1
            try
                println(Int.(states[i,:]))
                println(spins_a[i])
            catch
                0
            end
        end
    end

    if count_a != j
        println("xxxxxxxx peps approx No. matching energies = $(count_a), should be $j xxxxxxxx")
    end
    if true
    count_a = 0
    spins_a = 0

    M = [1 2 3; 4 5 6; 7 8 9]
    grid1 = Array{Array{Int}}(undef, (3,3))

    grid1[1,1] = [1 2; 6 7]
    grid1[1,2] = [3 4; 8 9]
    grid1[1,3] = reshape([5; 10], (2,1))
    grid1[2,1] = [11 12; 16 17]
    grid1[2,2] = [13 14; 18 19]
    grid1[2,3] = reshape([15; 20], (2,1))

    grid1[3,1] = reshape([21; 22], (1,2))
    grid1[3,2] = reshape([23; 24], (1,2))
    grid1[3,3] = reshape([25], (1,1))

    grid1 = Array{Array{Int}}(grid1)

    ns_l = [Node_of_grid(i, M, grid1) for i in 1:maximum(M)]


    χ = 2
    print("peps larger T")

    @time spins_l, objective_l = solve(interactions, ns_l, M, r*j; β = T(β), χ = χ, threshold = T(1e-10))

    count_l = copy(j)
    for i in 1:j

        if !(v2energy(QM, spins_l[i]) ≈ -ens[i])
            println("... pepse larger ...")
            println("n.o. state = ", i)
            println("energies (peps,bf)", (v2energy(QM, spins_l[i]), -ens[i]))
            count_l = count_l - 1
            try
                println(Int.(states[i,:]))
                println(spins_l[i])
            catch
                0
            end
        end
    end

    if count_l != j
        println("xxxxxxxx peps larger_tensors matching energies = $(count_l), should be $j xxxxxxxx")
    end
    end

    χ = 10
    β_step = 2

    print("mps time  =  ")
    ns = [Node_of_grid(i, interactions) for i in 1:get_system_size(interactions)]
    @time spins_mps, objective_mps = solve_mps(interactions, ns, r*j; β=β, β_step=β_step, χ=χ, threshold = 1.e-8)

    count_mps = copy(j)
    for i in 1:j

        if !(v2energy(QM, spins_mps[i]) ≈ -ens[i])
            println(".... mps .....")
            println("n.o. state = ", i)
            println("energies (mps,bf)", (v2energy(QM, spins_mps[i]), -ens[i]))
            count_mps = count_mps - 1
            try
                println( Int.(states[i,:]))
                println(spins_mps[i])
            catch
                0
            end
        end
    end

    if count_mps != j
        println("xxxxxxxxx  mps No. matching energies = $(count_mps), should be $j xxxxxxxxxxx")
    end

    println("peps, approx error = ", maximum(abs.(objective .- objective_a)))


end
