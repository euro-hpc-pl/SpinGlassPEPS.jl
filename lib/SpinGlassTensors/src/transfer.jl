
# transfer.jl: This file provides rules of how to transfer tensors to GPU. Note, NOT all of
#              tensor's coponents are moved from CPU to GPU and most tensors are generated
#              on CPU due to the size of clustered Hamiltonian.
export which_device, move_to_CUDA!, move_to_CPU!

move_to_CUDA!(ten::Array{T,N}) where {T,N} = CuArray(ten) #cu(ten, unified=true)


move_to_CUDA!(ten::Union{CuArray{T,N},Nothing}) where {T,N} = ten
move_to_CUDA!(ten::Diagonal) = Diagonal(move_to_CUDA!(diag(ten)))

function move_to_CUDA!(ten::CentralTensor)
    ten.e11 = move_to_CUDA!(ten.e11)
    ten.e12 = move_to_CUDA!(ten.e12)
    ten.e21 = move_to_CUDA!(ten.e21)
    ten.e22 = move_to_CUDA!(ten.e22)
    ten
end

function move_to_CUDA!(ten::DiagonalTensor)
    ten.e1 = move_to_CUDA!(ten.e1)
    ten.e2 = move_to_CUDA!(ten.e2)
    ten
end

function move_to_CUDA!(ten::VirtualTensor)
    ten.con = move_to_CUDA!(ten.con)
    # ten.projs = move_to_CUDA!.(ten.projs) # TODO 1) is this necessary ?
    ten
end

function move_to_CUDA!(ten::SiteTensor)
    ten.loc_exp = move_to_CUDA!(ten.loc_exp)
    # ten.projs = move_to_CUDA!.(ten.projs) # TODO 2) is this necessary ?
    ten
end

function move_to_CUDA!(ten::MpoTensor)
    for i ∈ 1:length(ten.top)
        ten.top[i] = move_to_CUDA!(ten.top[i])
    end
    for i ∈ 1:length(ten.bot)
        ten.bot[i] = move_to_CUDA!(ten.bot[i])
    end
    ten.ctr = move_to_CUDA!(ten.ctr)
    ten
end

function move_to_CUDA!(ψ::Union{QMpo{T},QMps{T}}) where {T}
    for k ∈ keys(ψ.tensors)
        ψ[k] = move_to_CUDA!(ψ[k])
    end
    ψ.onGPU = true
    ψ
end

move_to_CPU!(ten::CuArray{T,N}) where {T,N} = Array(ten)
move_to_CPU!(ten::Union{Array{T,N},Nothing}) where {T,N} = ten
move_to_CPU!(ten::Diagonal) = Diagonal(move_to_CPU!(diag(ten)))

function move_to_CPU!(ψ::QMps{T}) where {T}
    for k ∈ keys(ψ.tensors)
        ψ[k] = move_to_CPU!(ψ[k])
    end
    ψ.onGPU = false
    ψ
end



which_device(::Nothing) = Set()
which_device(ψ::Union{QMpo{T},QMps{T}}) where {T} =
    union(which_device.(values(ψ.tensors))...)
which_device(ten::MpoTensor) =
    union(which_device(ten.ctr), which_device.(ten.top)..., which_device.(ten.bot)...)
which_device(ten::DiagonalTensor) = union(which_device.((ten.e1, ten.e2))...)
which_device(ten::VirtualTensor) = union(which_device.((ten.con,))...) # TODO cf. 1)  ten.projs
which_device(ten::CentralTensor) =
    union(which_device.((ten.e11, ten.e12, ten.e21, ten.e22))...)
which_device(ten::SiteTensor) = union(which_device.((ten.loc_exp,))...) # TODO cf. 2)  ten.projs
which_device(ten::Array{T,N}) where {T,N} = Set((:CPU,))
which_device(ten::CuArray{T,N}) where {T,N} = Set((:GPU,))
which_device(ten::Diagonal) = which_device(diag(ten))
