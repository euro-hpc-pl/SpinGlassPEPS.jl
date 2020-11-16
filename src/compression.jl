import LinearAlgebra: QRIteration
import LinearAlgebra: DivideAndConquer
using LowRankApprox
using Random

# TODO some of functions from LP-BG can be used there                                         |
#                                             |
# TODO permute dimetnions to ---viatual - physical - virtual---

function make_left_canonical(t1::Array{T, 3}, t2::Array{T, 3}) where T <: AbstractFloat
    s = size(t1)

    p1 = [1,3,2]
    A1 = permutedims(t1, p1)
    A1 = reshape(A1, (s[1]*s[3], s[2]))
    # TODO this need to be secured for singular matrix case
    # or a case as if does not converge, LAPACKException(15)
    #nor temporary solution noised the matrix
    U = 0.
    Σ = 0.
    V = 0.
    try
        U,Σ,V = svd(A1; alg = QRIteration())
    catch
        println("save problematic matrix")
        npzwrite("problematic_matrix.npz", A1)
        U,Σ,V = svd(A1; alg = QRIteration())
    end
    T2 = diagm(Σ)*transpose(V)
    k = length(Σ)

    Anew = reshape(U, (s[1], s[3], k))
    Anew = permutedims(Anew, invperm(p1))

    @tensor begin
        t2[a,b,c] := T2[a,x]*t2[x,b,c]
    end
    Anew, t2
end


# TODO perhaps some reshapes can be reduced
function make_right_canonical(t1::Array{T, 3}, t2::Array{T, 3}) where T <: AbstractFloat

    s = size(t2)

    B2 = reshape(t2, (s[1], s[2]*s[3]))

    U,Σ,V = svd(B2)
    k = length(Σ)
    T1 = U*diagm(Σ)
    V = transpose(V)

    Bnew = reshape(V, (k, s[2], s[3]))

    @tensor begin
        t1[a,b,c] := T1[x,b]*t1[a,x,c]
    end

    t1, Bnew
end

function vec_of_right_canonical(mps::Vector{Array{T, 3}}) where T <: AbstractFloat
    for i in length(mps)-1:-1:1
        mps[i], mps[i+1] = make_right_canonical(mps[i], mps[i+1])
    end
    mps
end


function vec_of_left_canonical(mps::Vector{Array{T, 3}}) where T <: AbstractFloat
    for i in 1:length(mps)-1
        mps[i], mps[i+1] = make_left_canonical(mps[i], mps[i+1])
    end
    mps
end

# TODO χ may be leveraged up to the svd()
function left_canonical_approx(mps::Vector{Array{T, 3}}, χ::Int) where T <: AbstractFloat

    mps = vec_of_left_canonical(copy(mps))
    if χ == 0
        return mps
    else
        for i in 1:length(mps)-1
            s = size(mps[i], 2)
            χ1 = min(s, χ)

            mps[i] = mps[i][:,1:χ1,:]
            mps[i+1] = mps[i+1][1:χ1,:,:]
        end
    end
    mps
end


# TODO it is noe used right now
function right_canonical_approx(mps::Vector{Array{T, 3}}, χ::Int) where T <: AbstractFloat

    mps = vec_of_right_canonical(copy(mps))
    if χ == 0
        return mps
    else
        for i in 2:length(mps)
            s = size(mps[i], 1)
            χ1 = min(s, χ)

            mps[i-1] = mps[i-1][:,1:χ1,:]
            mps[i] = mps[i][1:χ1,:,:]
        end
    end
    mps
end


function QR_make_right_canonical(t2::Array{T, 3}) where T <: AbstractFloat

    s = size(t2)
    p = [2,3,1]
    t2 = permutedims(t2, p)

    B2 = reshape(t2, (s[2]*s[3], s[1]))
    Q,R = qr(B2)
    Q = Q[:,1:size(R,1)]

    l = min(size(Q,2), s[1])

    Bnew = reshape(Q, (s[2], s[3], l))
    Bnew = permutedims(Bnew, invperm(p))

    Bnew, R
end

# TODO if we permute dimentions on the start the permutation here would not
# be necessary
function QR_make_left_canonical(t2::Array{T, 3}) where T <: AbstractFloat

    s = size(t2)
    p = [1,3,2]
    t2 = permutedims(t2, p)

    B2 = reshape(t2, (s[1]*s[3], s[2]))
    Q,R = qr(B2)
    Q = Q[:,1:size(R,1)]
    l = min(size(Q,2), s[2])

    Bnew = reshape(Q, (s[1], s[3], l))
    Bnew = permutedims(Bnew, invperm(p))

    Bnew, R
end

"""
    function R_update(U::Array{T, 3}, U_exact::Array{T, 3}, R::Array{T, 2}) where T <: AbstractFloat

update the right enviroment in the approximation,
Return matrix, the updated enviroment
"""
function R_update(U::Array{T, 3}, U_exact::Array{T, 3}, R::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(U, 1), size(U_exact, 1))
    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b] = U[a,x,v]*U_exact[b,y,v]*R[x,y]
    end
    C
end

"""
    L_update(U::Array{T, 3}, U_exact::Array{T, 3}, R::Array{T, 2}) where T <: AbstractFloat

update the left enviroment in the approximation,
Return matrix, the updated enviroment
"""
function L_update(U::Array{T, 3}, U_exact::Array{T, 3}, R::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(U, 2), size(U_exact, 2))
    @tensor begin
        C[a,b] = U[x,a,v]*U_exact[y,b,v]*R[x,y]
    end
    C
end

function compress_mps_itterativelly(mps::Vector{Array{T,3}}, mps_anzatz::Vector{Array{T,3}}, threshold::Float64) where T <: AbstractFloat

    mps_centr = [zeros(T, 1,1,1) for _ in 1:length(mps)]

    mps_ret = [zeros(T, 1,1,1) for _ in 1:length(mps)]
    mps_ret1 = [zeros(T, 1,1,1) for _ in 1:length(mps)]
    maxsweeps = 300
    maxsweeps = 5

    # initialize R and L
    all_L = [ones(T,1,1) for _ in 1:length(mps)]
    for i in 1:length(mps)-1
        all_L[i+1] = L_update(mps_anzatz[i], mps[i], all_L[i])
    end
    all_R = [ones(T,1,1) for _ in 1:length(mps)]

    s = size(mps[end],2)
    R_exact = Matrix{T}(I, s,s)

    for sweep = 1:maxsweeps
        n = 0.
        ϵ = 0.
        for i in length(mps):-1:1
            # transform to canonical centre
            @tensor begin
                mps_c[a,b,c] := mps[i][a,x,c]*R_exact[b,x]
            end
            mps_centr[i] = mps_c

            @tensor begin
                #v concers contracting modes of size 1 in C
                M[a,b,c] := all_L[i][a,y]*mps_c[y,z,c]*all_R[i][b,z]
            end

            Q, TD = QR_make_right_canonical(M)
            Q_exact, R_exact = QR_make_right_canonical(mps_c)

            # compute ϵ
            @tensor begin
                X[x,y] := M[x,a,b]*M[y,a,b]
            end
            if n == 0.
                n = norm(R_exact)
            end
            ϵ = ϵ + 1-tr(X./n^2)

            mps_ret[i] = Q
            if i > 1
                all_R[i-1] = R_update(Q, Q_exact, all_R[i])
            end
        end
        if false
            println("ϵ l2r = ", ϵ)
        end

        ϵ = 0.

        s = size(mps[1],1)
        R_exact = Matrix{T}(I, s,s)
        for i in 1:length(mps)

            mps_c = mps_centr[i]

            @tensor begin
                #v concers contracting modes of size 1 in C
                M[a,b,c] := all_L[i][a,y]*mps_c[y,z,c]*all_R[i][b,z]
            end

            Q, TD = QR_make_left_canonical(M)
            mps_ret1[i] = Q

            @tensor begin
                X[x,y] := M[x,a,b]*M[y,a,b]
            end
            ϵ = ϵ + 1-tr(X./n^2)

            if i < length(mps)
                A = L_update(Q, mps[i], all_L[i])
                all_L[i+1] = A
            end
        end
        if false
            println("ϵ r2l = ", ϵ)
        end
        if abs(ϵ) < threshold
            return mps_ret1
        end
    end
    mps_ret1
end

# various compressing schemes

function compress_iter(mps::Vector{Array{T,3}}, χ::Int, threshold::Float64) where T <: AbstractFloat
    mps_lc = left_canonical_approx(mps, 0)
    mps_anzatz = left_canonical_approx(mps, χ)
    compress_mps_itterativelly(mps_lc, mps_anzatz, threshold)
end


# TODO I made this simple implementation for the intra-step compression

function compress_iter(mpo::Vector{Array{T,4}}, χ::Int, threshold::Float64) where T <: AbstractFloat
    s = [size(el) for el in mpo]
    mps = [reshape(mpo[i], (s[i][1], s[i][2], s[i][3]*s[i][4])) for i in 1:length(mpo)]
    mps = compress_iter(mps, χ, threshold)

    s1 =  [size(el) for el in mps]
    mps = [reshape(mps[i], (s1[i][1], s1[i][2], s[i][3], s[i][4])) for i in 1:length(mps)]
    return mps
end

# TODO for testing - comparison only
function compress_svd(mps::Vector{Array{T,3}}, χ::Int) where T <: AbstractFloat
    left_canonical_approx(mps, χ)
end
