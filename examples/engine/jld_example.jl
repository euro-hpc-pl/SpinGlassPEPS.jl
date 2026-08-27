using JLD2

r = rand(3, 3, 3)
path = joinpath(@__DIR__, "data.jld")
jldsave(path; data = r)
load(path, "data")
