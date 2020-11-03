export idx, to_ising

to_ising(state) = 2 .* state .- 1

idx(s::Int) = idx(Val(s))
idx(::Val{-1}) = 1
idx(::Val{1}) = 2
