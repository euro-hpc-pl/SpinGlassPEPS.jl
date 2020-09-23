using NPZ

include("../notation.jl")
include("../peps.jl")

file = "examples.npz"
no_diag_degeneracy = true
j = 10
β = 4.

#file = "examples2.npz"
#no_diag_degeneracy = false
#j = 1
#β = 3.


data = npzread(file)

println(file)


for k in 1:100
    println("SAMPLE = ", k)
    QM = data["Js"][k,:,:]
    states = data["states"][k,:,:]
    ens = data["energies"][k,:,:]

    function M2Qubbo_els(M::Matrix{Float64})
        qubo = Qubo_el{Float64}[]
        s = size(M)
        for i in 1:s[1]
            for j in i:s[2]
                if (M[i,j] != 0.) | (i == j)
                    x = -1*M[i,j]
                     q = Qubo_el((i,j), x)
                    push!(qubo, q)
                end
            end
        end
        qubo
    end


    qubo = M2Qubbo_els(QM)

    grid = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25]


    @time ses = solve(qubo, grid, j; β = β, χ = 0, threshold = 0.)


    for i in 1:j
        testv = (Int.(states[i,:]) == ses[j-i+1].spins) | (Int.(states[i,:])*.-1 == ses[j-i+1].spins)
        if !testv

            println("exact")
            println("n.o. state = ", i)

            println(Int.(states[i,:]))
            println(ses[j-i+1].spins)

            println(ens[i])
            println(ses[j-i+1].objective)
        end

    end

    χ = 2

    @time ses = solve(qubo, grid, j; β = β, χ = χ, threshold = 1e-6)



    for i in 1:j
        testv = false
        if no_diag_degeneracy
            testv = (Int.(states[i,:]) == ses[j-i+1].spins) | (Int.(states[i,:])*.-1 == ses[j-i+1].spins)
        else
            testv = Int.(states[i,:]) == ses[j-i+1].spins
        end
        if !testv
            println("approx, chi = ", χ)
            println("n.o. state = ", i)

            println(Int.(states[i,:]))
            println(ses[j-i+1].spins)

            println(ens[i])
            println(ses[j-i+1].objective)
        end
    end
end
