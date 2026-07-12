export measure_memory, format_bytes

measure_memory(ten::AbstractArray) = [Base.summarysize(ten), 0]  # [CPU_memory, GPU_memory]
measure_memory(ten::CuArray) = [0, prod(size(ten)) * sizeof(eltype(ten))]
measure_memory(ten::SparseMatrixCSC) =
    sum(measure_memory.([ten.colptr, ten.rowval, ten.nzval]))
measure_memory(ten::CuSparseMatrixCSC) =
    sum(measure_memory.([ten.colPtr, ten.rowVal, ten.nzVal]))
measure_memory(ten::CuSparseMatrixCSR) =
    sum(measure_memory.([ten.rowPtr, ten.colVal, ten.nzVal]))
measure_memory(ten::Diagonal) = measure_memory(diag(ten))
measure_memory(ten::SiteTensor) = sum(measure_memory.([ten.loc_exp]))  # ten.projs...]))
measure_memory(ten::CentralTensor) =
    sum(measure_memory.([ten.e11, ten.e12, ten.e21, ten.e22]))
measure_memory(ten::DiagonalTensor) = sum(measure_memory.([ten.e1, ten.e2]))
measure_memory(ten::VirtualTensor) = sum(measure_memory.([ten.con]))  # ten.projs...]))
measure_memory(ten::MpoTensor) = sum(measure_memory.([ten.top..., ten.ctr, ten.bot...]))
measure_memory(ten::Union{QMps,QMpo}) = sum(measure_memory.(values(ten.tensors)))
measure_memory(env::Environment) = sum(measure_memory.(values(env.env)))
measure_memory(env::EnvironmentMixed) = sum(measure_memory.(values(env.env)))
measure_memory(lp::PoolOfProjectors) = sum([measure_memory(da) for da ∈ values(lp.data)])
measure_memory(dict::Dict) = isempty(dict) ? [0, 0] : sum(measure_memory.(values(dict)))
measure_memory(tuple::Tuple) = sum(measure_memory.(tuple))
measure_memory(ten::Int) = [sizeof(ten), 0]
measure_memory(::Nothing) = [0, 0]


function format_bytes(bytes, decimals::Int = 2, k::Int = 1024)
    bytes == 0 && return "0 Bytes"
    dm = decimals < 0 ? 0 : decimals
    sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    i = convert(Int, floor(log(bytes) / log(k)))
    string(round((bytes / ^(k, i)), digits = dm)) * " " * sizes[i+1]
end

function measure_memory(caches::IdDict{Any,Any}, bytes::Bool = true)
    memoization_memory = bytes ? Dict{Any,Vector{String}}() : Dict{Any,Vector{Int64}}()
    for key in keys(caches)
        push!(
            memoization_memory,
            key =>
                bytes ? format_bytes.(measure_memory(caches[key])) :
                measure_memory(caches[key]),
        )
    end
    memoization_memory
end
