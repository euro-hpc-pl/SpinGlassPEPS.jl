# SpinGlassPEPSOpenGMExt.jl
#
# Extension providing openGM/RMF loading (SpinGlassNetworks.load_openGM), which
# requires HDF5. HDF5 is a weak dependency so the heavy HDF5_jll binary is only
# pulled in for users who actually load openGM/RMF instances; `using HDF5`
# alongside SpinGlassPEPS activates this method.
module SpinGlassPEPSOpenGMExt

using HDF5
using SpinGlassPEPS.SpinGlassNetworks: benchmark_names
import SpinGlassPEPS.SpinGlassNetworks: load_openGM

    function load_openGM(
        fname::String,
        Nx::Union{Integer,Nothing} = nothing,
        Ny::Union{Integer,Nothing} = nothing,
    )
        file = h5open(fname, "r")

        file_keys = collect(keys(read(file)))
        data = read(file[file_keys[1]])
        H = collect(Int64, data["header"])
        F = Array{Int64}(data["factors"])
        J = Array{Int64}(data["function-id-16000"]["indices"])
        V = Array{Float64}(data["function-id-16000"]["values"])
        N = Array{Int64}(data["numbers-of-states"])

        if isnothing(Nx) || isnothing(Ny)
            filename, _ = splitext(basename(fname))
            Nx, Ny = benchmark_names[filename]
        end

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
                if abs(n[1] - n[3]) + abs(n[2] - n[4]) ∉ [1, 2] || (
                    abs(n[1] - n[3]) + abs(n[2] - n[4]) == 2 &&
                    (abs(n[1] - n[3]) == 2 || abs(n[2] - n[4]) == 2)
                )
                    throw(ErrorException("Not nearest neighbour or diagonal neighbors"))
                end
            end

            if length(n) == 2
                if (n[1] >= Ny) || (n[2] >= Nx)
                    throw(ErrorException("Wrong size"))
                end
            end

            factors[tuple(n...)] = f1

            if z1 != 0
                throw(ErrorException("Something wrong with the expected convention."))
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

end
