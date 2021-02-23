export local_basis

function local_basis(ψ::CuMPS, i::Int)
    d = physical_dim(ψ, i)
    ret = CUDA.zeros(Int, d)
    @inline function kernel(ret)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
            if state <= length(ret)
                if state == 1
                    ret[state] = -1
                else
                    ret[state] = state - 1
                end
            end
        return
    end
    threads, blocks = cudiv(length(ret))
    CUDA.@cuda threads=threads blocks=blocks kernel(ret)
    ret
end

LinearAlgebra.I(ψ::CuMPS, i::Int) = CuMatrix(I(physical_dim(ψ, i)))
