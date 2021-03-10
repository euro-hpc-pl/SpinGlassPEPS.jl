function proj(state, dims::Union{Vector, NTuple})
    P = Matrix{Float64}[]
    for (σ, r) ∈ zip(state, dims)
        v = zeros(r)
        v[idx(σ)...] = 1.
        push!(P, v * v')
    end
    P
end

function tensor(ψ::AbstractMPS, state::State)
    C = I
    for (A, σ) ∈ zip(ψ, state)
        C *= A[:, idx(σ), :]
    end
    tr(C)
end

function tensor(ψ::MPS)
    dims = rank(ψ)
    Θ = Array{eltype(ψ)}(undef, dims)

    for σ ∈ all_states(dims)
        Θ[idx.(σ)...] = tensor(ψ, σ)
    end
    Θ
end

#removes bonds that do not fit to the grid, testing function
function fullM2grid!(M::Matrix{Float64}, s::Tuple{Int, Int})
    s1 = s[1]
    s2 = s[2]
    pairs = Vector{Int}[]
    for i ∈ 1:s1*s2
        if (i%s2 > 0 && i < s1*s2)
            push!(pairs, [i, i+1])
        end
        if i <= s2*(s1-1)
            push!(pairs, [i, i+s2])
        end
    end

    for k ∈ CartesianIndices(size(M))
        i1 = [k[1], k[2]]
        i2 = [k[2], k[1]]
        if !(i1 ∈ pairs) && !(i2 ∈ pairs) && (k[1] != k[2])
            M[i1...] = M[i2...] = 0.
        end
    end
end


function make_interactions_case1()
    L = 9

    D = Dict{Tuple{Int64,Int64},Float64}()
    push!(
    D,
    (1,1) => 1.0,
    (2,2) => -2.0,
    (3,3) => 4.0,
    (4,4) => 0.0,
    (5,5) => 1.5,
    (6,6) =>  0.1,
    (7,7) => 0.7,
    (8,8) => -0.16,
    (9,9) => 0.66,

    (1, 2) => -1.0,
    (1, 4) => -3,
    (2, 3) => -3.0,
    (2, 5) => -2.0,
    (3,6) => 3.0,
    (4,5) => 1.0,
    (4,7) => -0.02,
    (5,6) => -0.5,
    (5,8) => 1.0,
    (6,9) => -1.04,
    (7,8) => 1.7,
    (8,9) => -0.1)


    ising_graph(D)#, ising_graph(D, L)
end


function make_interactions_case2(T::Type = Float64)
    f = 1
    f1 = 1
    L = 16
    D = Dict{Tuple{Int64,Int64},T}()
    push!(
    D,
    (1, 1) => T(-2.8),
    (2, 2) => T(2.7),
    (3, 3) => T(-2.6),
    (4, 4) => T(2.5),
    (5, 5) => T(-2.4),
    (6, 6) => T(2.3),
    (7, 7) => T(-2.2),
    (8, 8) => T(2.1),
    (9, 9) => T(-2.0),
    (10, 10) => T(1.9),
    (11, 11) => T(-1.8),
    (12, 12) => T(1.70),
    (13, 13) => T(-1.6),
    (14, 14) => T(1.5),
    (15, 15) => T(-1.4),
    (16, 16) => T(1.3),

    (1, 2) => T(0.30),
    (1, 5) => T(0.2),
    (2, 3) => T(0.255),
    (2, 6) => T(0.21),
    (3, 4) => T(0.222),
    (3, 7) => T(0.213),

    (4, 8) => T(0.2),
    (5, 6) => T(0.15),
    (5, 9) => T(0.211),
    (6, 7) => T(0.2),
    (6, 10) => T(0.15),
    (7, 8) => T(0.11),
    (7, 11) => T(0.35),
    (8, 12) => T(0.19),
    (9, 10) => T(0.222),
    (9, 13) => T(0.15),
    (10, 11) => T(0.28),
    (10, 14) => T(0.21),
    (11, 12) => T(0.19),
    (11, 15) => T(0.18),
    (12, 16) => T(0.27),
    (13, 14) => T(0.32),
    (14, 15) => T(0.19),
    (15, 16) => T(0.21)
    )

    ising_graph(D)
end

enum(vec) = Dict(v => i for (i, v) ∈ enumerate(vec))
