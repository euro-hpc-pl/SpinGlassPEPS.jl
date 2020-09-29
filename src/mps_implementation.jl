include("peps.jl")
include("notation.jl")

function initialize_mps(l::Int, physical_dims::Int =  2, T::Type = Float64)
    [ones(T, 1,1,physical_dims) for _ in 1:l]
end


function make_ones(T::Type = Float64)
    d = 2
    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for j in 1:d
            ret[i,i,j,j] = T(1.)
        end
    end
    ret
end

function T_with_B(T::Type = Float64)
    d = 2
    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for j in 1:d
            for k in 1:d
                for l in 1:d
                    ret[i,j,k,l] = T(i==j)*T(k==l)*T(j==l)
                end
            end
        end
    end
    ret
end

function T_with_C(Jb::T) where T <: AbstractFloat
    d = 2
    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for j in 1:d
            for k in 1:d
                for l in 1:d
                    ret[i,j,k,l] = T(i==j)*T(k==l)*exp(Jb*ind2spin(i)*ind2spin(k))
                end
            end
        end
    end
    ret
end

function add_MPO!(mpo::Vector{Array{T, 4}}, i::Int, i_n::Vector{Int}, qubo::Vector{Qubo_el{T}}, β::T) where T<: AbstractFloat
    mpo[i] = T_with_B(T)
    for j in i_n
        J = JfromQubo_el(qubo, i,j)
        println(i,",", j,",", J)
        mpo[j] = T_with_C(J*β)
    end
    mpo
end

function reduce_first_and_last!(mpo::Vector{Array{T, 4}}) where T <:AbstractFloat
    mpo[1] = sum(mpo[1], dims = 1)
    mpo[end] = sum(mpo[end], dims = 2)
end
