using SpinGlassPEPS
using MetaGraphs
using LightGraphs

#      grid
#     A1    |    A2
#           |
#   1 -- 3 -|- 5 -- 7
#   |    |  |  |    |
#   |    |  |  |    |
#   2 -- 4 -|- 6 -- 8
#           |

D = Dict{Tuple{Int64,Int64},Float64}()
push!(D, (1,1) => 2.5)
push!(D, (2,2) => 1.4)
push!(D, (3,3) => 2.3)
push!(D, (4,4) => 1.2)
push!(D, (5,5) => -2.5)
push!(D, (6,6) => -.5)
push!(D, (7,7) => -.3)
push!(D, (8,8) => -.2)

push!(D, (1,2) => 1.3)
push!(D, (3,4) => -1.)
push!(D, (5,6) => 1.1)
push!(D, (7,8) => .1)

push!(D, (1,3) => .8)
push!(D, (3,5) => .5)
push!(D, (5,7) => -1.)

push!(D, (2,4) => 1.7)
push!(D, (4,6) => -1.5)
push!(D, (6,8) => 1.2)

m = 1
n = 2
t = 4

L = m * n * t 

g_ising = ising_graph(D, L)

update_cells!(
  g_ising,
  rule = square_lattice((m, 1, n, 1, t)),
)

fg = factor_graph(
    g_ising,
    energy=energy,
    spectrum=full_spectrum,
)


#println([get_prop(fg, e, :edge) for e in edges(fg)])

origin = :NW
β = 2.

x, y = m, n
peps = PepsNetwork(x, y, fg, β, origin)
pp = PEPSRow(peps, 1)
println(pp)

bf = brute_force(g_ising; num_states = 1)
#println(bf.energies)

states = bf.states[1]

cell_A1 = states[1:4]
cell_A1_left = states[3:4]
cell_A2 = states[5:8]
cell_A2_right = states[5:6]

A2 = @state pp[2][cell_A1_left, 1, 1, 1, :]

_, spins = findmax(A2)

st = get_prop(fg, 2, :spectrum).states

println("ground state of A2 from PEPS i.e. at index #  $(spins)  = ", st[spins] )
println("this should correspond to the ground state of A2 from brute force = ", cell_A2)
