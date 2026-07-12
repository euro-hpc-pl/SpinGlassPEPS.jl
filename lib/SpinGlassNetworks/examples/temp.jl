using HDF5
using Graphs
using LinearAlgebra
using LabelledGraphs
using MetaGraphs
using SpinGlassNetworks


function load_openGM(fname::String, Nx::Integer, Ny::Integer)
    file = h5open(fname, "r")

    file_keys = collect(keys(read(file)))
    data = read(file[file_keys[1]])
    H = collect(Int64, data["header"])
    F = Array{Int64}(data["factors"])
    J = Array{Int64}(data["function-id-16000"]["indices"])
    V = Array{Real}(data["function-id-16000"]["values"])
    N = Array{Int64}(data["numbers-of-states"])

    F = reverse(F)
    factors = Dict()

    while length(F) > 0
        f1 = pop!(F)
        z1 = pop!(F)
        nn = pop!(F)
        n = []

        for _ = 1:nn
            tt = pop!(F)
            ny, nx = divrem(tt, Nx)
            push!(n, ny, nx)
        end

        if length(n) == 4
            if abs(n[1] - n[3]) + abs(n[2] - n[4]) != 1
                throw(Exception("Not nearest neighbour"))
            end
        end

        if length(n) == 2
            if (n[1] >= Ny) || (n[2] >= Nx)
                throw(Exception("Wrong size"))
            end
        end

        factors[tuple(n...)] = f1

        if z1 != 0
            throw(Exception("Something wrong with the expected convention."))
        end
    end

    J = reverse(J)
    functions = Dict()
    ii = -1
    lower = 0

    while length(J) > 0
        ii += 1
        nn = pop!(J)
        n = []

        for _ = 1:nn
            push!(n, pop!(J))
        end

        upper = lower + prod(n)
        functions[ii] = reshape(V[lower+1:upper], reverse(n)...)'

        lower = upper
    end

    result = Dict(
        "fun" => functions,
        "fac" => factors,
        "N" => reshape(N, (Ny, Nx)),
        "Nx" => Nx,
        "Ny" => Ny,
    )
    result
end

function potts_hamiltonian(fname::String, Nx::Integer = 240, Ny::Integer = 320)
    loaded_rmf = load_openGM(fname, Nx, Ny)
    functions = loaded_rmf["fun"]
    factors = loaded_rmf["fac"]
    N = loaded_rmf["N"]

    clusters = super_square_lattice((Nx, Ny, 1))
    potts_h = LabelledGraph{MetaDiGraph}(sort(collect(values(clusters))))
    for v ∈ potts_h.labels
        x, y = v
        sp = Spectrum(
            Vector{Real}(undef, 1),
            Array{Vector{Int}}(undef, 1, 1),
            Vector{Int}(undef, 1),
        )
        set_props!(potts_h, v, Dict(:cluster => v, :spectrum => sp))
    end
    for (index, value) in factors
        if length(index) == 2
            y, x = index
            Eng = sum(functions[value])
            set_props!(potts_h, (x + 1, y + 1), Dict(:eng => Eng))
        elseif length(index) == 4
            y1, x1, y2, x2 = index
            add_edge!(potts_h, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1))
            Eng = sum(functions[value], dims = 2)
            set_props!(
                potts_h,
                (x1 + 1, y1 + 1),
                (x2 + 1, y2 + 1),
                Dict(
                    :outer_edges => ((x1 + 1, y1 + 1), (x2 + 1, y2 + 1)),
                    :eng => Eng,
                    :pl => I,
                    :pr => I,
                ),
            )
        else
            throw(
                ErrorException(
                    "Something is wrong with factor index, it has length $(length(index))",
                ),
            )
        end
    end

    potts_h
end


x, y = 240, 320
filename = "/home/tsmierzchalski/.julia/dev/SpinGlassNetworks/examples/penguin-small.h5"


cf = potts_hamiltonian(filename, x, y)
