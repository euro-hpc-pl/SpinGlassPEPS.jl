using NPZ
using Plots
using Test

include("../notation.jl")
include("../compression.jl")
include("../peps.jl")
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

T = Float64

data = npzread("./data/"*file)
println(file)

function v2energy(M::Matrix{T}, v::Vector{Int}) where T <: AbstractFloat
    d =  diag(M)
    M = M .- diagm(d)

    transpose(v)*M*v + transpose(v)*d
end


for k in 1:examples
    println(" ..................... SAMPLE =  ", k, "  ...................")
    QM = data["Js"][k,:,:]

    states = 0
    try
        states = data["states"][k,:,:]
    catch
        0
    end
    ens = data["energies"][k,:,:]

    function M2Qubbo_els(M::Matrix{Float64}, T::Type = Float64)
        qubo = Qubo_el{T}[]
        s = size(M)
        for i in 1:s[1]
            for j in i:s[2]
                if (M[i,j] != 0.) | (i == j)
                    x = T(M[i,j])
                    q = Qubo_el{T}((i,j), x)
                    push!(qubo, q)
                end
            end
        end
        qubo
    end

    qubo = M2Qubbo_els(QM, T)

    grid = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25]

    print("peps time  = ")
    @time spins, objective = solve(qubo, grid, r*j; β = T(β), χ = 0, threshold = T(0.))

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

    if k == 1
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
    print("approx peps t")
    @time spins_a, objective_a = solve(qubo, grid, r*j; β = T(β), χ = χ, threshold = T(1e-10))

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

    χ = 10
    β_step = 2

    is,js = connections_for_mps(Array(transpose(grid)))
    all_is, all_js = cluster_conncetions(is,js)

    #all_is = [[1,8,20], [3,10,17], [5,13], [2, 14], [4, 12, 18], [6, 22], [7,21], [11, 24]]

    #all_js = [[[2,6],[7,9,13],[15,19,25]],[[2,4,8], [9,15], [16,18,22]], [[4,10],[12,14,18]]]
    #all_js = vcat(all_js, [[[7], [9,15,19]], [[9], [11,17], [19,23]], [[7,11], [23]], [[12], [16,22]], [[16], [19,23,25]]])

    print("mps time  =  ")
    @time spins_mps, objective_mps = solve_mps(qubo, grid, 25, r*j; β=β, β_step=β_step, χ=χ, threshold = 1.e-8)

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
