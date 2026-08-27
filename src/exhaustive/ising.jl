export energy_qubo, energy, kernel, kernel_qubo, exhaustive_search, exhaustive_search_bucket

struct Spectrum
    energies::AbstractVector
    states::Union{AbstractVecOrMat{Int}, AbstractVecOrMat{Int32}}
end

"""
$(SIGNATURES)
- `state_code`: state code for which the energy expressed in qubo is to be calculated.
- `graph`: graph of the ising model.
Returns the state energy expressed as QUBO.
"""
function energy_qubo(state_code, graph)
    F = 0
    N = size(graph,1)
    
    for i in 1:N
        
        tstbit(state_code, i) ? qi=1 : continue
        
        F += graph[i,i]*qi  
               
        for j in i+1:N
                       
            tstbit(state_code, j) ? qj=1 : continue
                                   
            F += graph[i,j]*qi*qj
        end
    end
    return F
end

"""
$(SIGNATURES)
- `state_code`: state code for which the energy.
- `graph`: graph of the ising model.
Returns the state energy.
"""
function energy(state_code, graph)
    F = 0
    N = size(graph,1)
    
    for i in 1:N
        
        tstbit(state_code, i) ? qi=1 : qi=-1
                             
        F += graph[i,i]*qi  
               
        for j in i+1:N
                       
            tstbit(state_code, j) ? qj=1 : qj=-1
                                   
            F += graph[i,j]*qi*qj
        end
    end
    return F
end

"""
$(SIGNATURES)
- `graph`: graph of the ising model.
- `energies`: array filled with zeros. Each array index represents the state of the system.
Returns energies for every state.
"""
function kernel(graph, energies)
    array_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    array_index > length(energies) && return nothing
    state_code = array_index - 1

    F = energy(state_code, graph) |> Float32

    @inbounds energies[array_index] = F
          
    return
end

"""
$(SIGNATURES)
- `graph`: graph of the ising model.
- `energies`: array filled with zeros. Each array index represents the state of the system.
Returns energies energy expressed as QUBO for every state.
"""
function kernel_qubo(graph, energies)
    array_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    array_index > length(energies) && return nothing
    state_code = array_index - 1

    F = energy_qubo(state_code, graph) |> Float32

    @inbounds energies[array_index] = F
          
    return
end

"""
$(SIGNATURES)
- `graph`: graph of the ising model.
- `energies`: array filled with zeros. Each array index represents the state of the system.
- `part_lst`: list for collecting partial energy results.
- `part_st`: list for collecting partial state results.
Returns energies for every state.
"""
# function kernel_part(graph, energies, part_lst, part_st)
#     i = blockIdx().x
#     j = threadIdx().x

#     state_code = (i - 1) * blockDim().x + j 

#     F = energy(state_code, graph) |> Float32
    
#     @inbounds energies[state_code] = F

#     sync_threads()
    
#     if j == 1
#         k = (i - 1) * blockDim().x + 1
#         n = blockDim().x
        
#         value=energies[k]
#         st=k
#         for ii in k:k+n-1
#             if value > energies[ii]
#                 value = energies[ii]
#                 st=k
#             end
#         end 
               
#         sync_threads()
        
#         part_lst[i] = value # low_en[k]
#         part_st[i] = st
#     end
    
#     return
# end

"""
$(SIGNATURES)
- `ig::IsingGraph`: graph of ising model represented by IsingGraph structure.
Returns energies and states for provided graph by brute-forece alorithm based on GPU.
"""
function exhaustive_search(ig::IsingGraph)
    L = SpinGlassNetworks.nv(ig)

    J = couplings(ig) + Diagonal(biases(ig))
    J_dev = CUDA.CuArray(J)
    
    N = size(J_dev,1)

    energies = CUDA.zeros(2^N)

    threads = 512
    blocks = cld(length(energies), threads)
    
    @cuda blocks=blocks threads=threads kernel(J_dev, energies)
    
    permutation = sortperm(energies)
       
    # Array indices are one-based, while state codes follow the CPU convention
    # and range from zero to 2^N - 1.
    Spectrum(energies[permutation], permutation .- 1)
end

"""
$(SIGNATURES)
- `ig::IsingGraph`: graph of ising model represented by IsingGraph structure.
Returns energies and states for provided graph by brute-forece alorithm supported by partial selection based on GPU.
"""
# function partial_exhaustive_search(ig::IsingGraph)
#     L = SpinGlassNetworks.nv(ig)
#     N = 2^L
    
   
#     σ = CUDA.fill(Int32(-1), L, N)
#     J = couplings(ig) + Diagonal(biases(ig))
#     J_dev = CUDA.CuArray(J)
  
#     k = 2

#     energies = CUDA.zeros(N)
#     part_st = CUDA.zeros(2^(L-k))
#     part_lst = CUDA.zeros(2^(L-k))
    
#     threads = 512
#     blocks = cld(L, threads)
    
#     @cuda blocks=blocks threads=threads kernel_part(J_dev, energies, part_lst, part_st)
   
#     idx = sortperm(part_lst)
#     part_st[idx]
#     part_lst[idx]
       
#     Spectrum(part_lst[idx], part_st[idx])
# end

"""
$(SIGNATURES)
Returns the maximum chunk size for the algorithm supported by bucket selection.
"""
function max_chunk_size()
    mem_bytes = CUDA.free_memory()
    elements_max = mem_bytes ÷ 16 ÷ 2

    chunk_size = 0

    while elements_max > 1
        elements_max >>= 1
        chunk_size += 1
    end

    if 2 * 16 * 2 ^ chunk_size + 1024 * 1024 * 1024 > mem_bytes
        chunk_size -= 1
    end

    chunk_size
end

"""
$(SIGNATURES)
- `graph`: graph of the ising model.
- `energies`: array filled with zeros. Each array index represents the state of the system.
- `idx`: list for collecting partial energy results.
Returns energies for given indexes.
"""
function kernel_bucket(graph, energies, state_offset)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    i > length(energies) && return nothing
    state_code = state_offset + i - 1

    
    F = energy(state_code, graph) |> Float32
    
    @inbounds energies[i] = F
          
    return
end

"""
$(SIGNATURES)
- `ig::IsingGraph`: graph of ising model represented by IsingGraph structure.
Returns energies and states for provided graph by brute-forece alorithm supported by bucket selection based on GPU.
"""
function exhaustive_search_bucket(ig::IsingGraph, how_many::Int = 8)
    L = SpinGlassNetworks.nv(ig)
    how_many > 0 || throw(ArgumentError("how_many must be positive"))
    num_results = min(how_many, 2^L)

    J = couplings(ig) + Diagonal(biases(ig))
    J_dev = CUDA.CuArray(J)
    
    N = size(J_dev,1)
    
    chunk_size = max_chunk_size()

    if chunk_size > N
        chunk_size = N
    end


    energies_d = CUDA.zeros(Float64, 2^chunk_size) 
    # `Inf` marks unused candidate slots when a requested result set is larger
    # than an individual chunk.
    lowest_d = CUDA.fill(Inf, num_results * 2)
    lowest_states_d = CUDA.zeros(Int64, num_results * 2)
    
    threads = 512
    blocks = cld(length(energies_d), threads)

    for i in 1:2^(N-chunk_size)

        state_offset = (i - 1) * (2^chunk_size)
        
        @cuda blocks=blocks threads=threads kernel_bucket(J_dev, energies_d, state_offset)

        local_count = min(num_results, length(energies_d))
        states_d = sortperm(energies_d)[1:local_count]

        if i == 1
            lowest_d[1:local_count] = energies_d[states_d]
            lowest_states_d[1:local_count] = states_d .- 1 .+ state_offset
        else
            lowest_d[num_results+1:num_results+local_count] = energies_d[states_d]
            lowest_states_d[num_results+1:num_results+local_count] =
                states_d .- 1 .+ state_offset

            states = sortperm(lowest_d)
            lowest_d = lowest_d[states]
            lowest_states_d = lowest_states_d[states]
        end 
    end


    Spectrum(lowest_d[1:num_results], lowest_states_d[1:num_results])
end
