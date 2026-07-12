export bitonicsort

# sort kernel
bisort!(shared, j, k) = begin
  tid = UInt(((blockIdx().x - 1) * blockDim().x + threadIdx().x)-1)
  ixj = tid⊻UInt(j)
  if ixj > tid
    if (tid & k) == 0
        if shared[tid+1] > shared[ixj+1]
          tmp = shared[ixj+1]
          shared[ixj+1] = shared[tid+1]
          shared[tid+1] = tmp
        end
    else
        if shared[tid+1] < shared[ixj+1]
          tmp = shared[ixj+1]
          shared[ixj+1] = shared[tid+1]
          shared[tid+1] = tmp
        end
    end
  end
  return
end

n_threat = 256

bitonicsort!(cushared) = begin
  CUDA.@sync begin
    k = UInt(2)
    NUM = length(cushared)
    nblocks = ceil(NUM/n_threat) |> Int
    while (k <= NUM)
      j = div(k, 2)
      while j >= 1
        @cuda threads = n_threat blocks = nblocks bisort!(cushared, j, k)
        j =  div(j,2)
      end
      k = UInt(k*2)
    end
  end
end