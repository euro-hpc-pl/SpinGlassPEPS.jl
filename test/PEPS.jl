
@testset "PepsTensor correctly builds PEPS network" begin

m = 3
n = 4
t = 3

β = 1

L = m * n * t

instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt" 

ig = ising_graph(instance, L)
update_cells!(
   ig, 
   rule = square_lattice((m, n, t)),
) 

fg = factor_graph(
    ig, 
    energy=energy, 
    spectrum=full_spectrum,
)

x = m
y = n
peps = PepsNetwork(x, y, fg, β, :NW)
println(typeof(peps))


for i ∈ 2:2, j ∈ 1:y
    println(i, j)
    @time A = generate_tensor(peps, (i, j))
    println(size(A))
end

i=2
for j ∈ 1:y-1
    A = generate_tensor(peps, (i, j), (i, j+1))
    println(size(A))
end

for i ∈ 1:x
    println(i)
    @time mpo = MPO(peps, i)
    println(size(mpo))
end

end