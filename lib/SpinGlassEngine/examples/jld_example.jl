using Pkg
Pkg.add("JLD")

using JLD
r = rand(3, 3, 3)
save("$(@__DIR__)/data.jld", "data", r)
load("$(@__DIR__)/data.jld")["data"]
