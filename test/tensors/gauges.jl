using Random

function random_mps(::Type{T}, sites, physical_dimensions, bond_dimensions) where {T}
    tensors = TensorMap{T}()
    for (index, site) ∈ enumerate(sites)
        tensors[site] = randn(
            T,
            bond_dimensions[index],
            bond_dimensions[index+1],
            physical_dimensions[index],
        )
    end
    QMps(tensors)
end

function mps_amplitude(ψ::QMps{T}, state) where {T}
    amplitude = ones(T, 1)
    for (site, local_state) ∈ zip(ψ.sites, state)
        amplitude = vec(transpose(amplitude) * view(ψ[site], :, :, local_state))
    end
    only(amplitude)
end

function reference_overlap_density(ϕ::QMps{T}, ψ::QMps{T}, position::Int) where {T}
    physical_dimensions = [size(ψ[site], 3) for site ∈ ψ.sites]
    ranges = [index == position ? (1:1) : (1:dimension) for
              (index, dimension) ∈ enumerate(physical_dimensions)]
    density = zeros(T, physical_dimensions[position], physical_dimensions[position])
    for rest ∈ Iterators.product(ranges...)
        ϕ_state = collect(rest)
        ψ_state = collect(rest)
        for x ∈ axes(density, 1), y ∈ axes(density, 2)
            ϕ_state[position] = x
            ψ_state[position] = y
            density[x, y] += conj(mps_amplitude(ϕ, ϕ_state)) * mps_amplitude(ψ, ψ_state)
        end
    end
    density
end

@testset "Overlap density matrices" begin
    Random.seed!(0x5eed)
    sites = Site[1 // 2, 2, 7 // 2]
    physical_dimensions = [2, 3, 2]
    ψ = random_mps(Float64, sites, physical_dimensions, [1, 2, 4, 1])
    ϕ = random_mps(Float64, sites, physical_dimensions, [1, 3, 5, 1])

    for (position, site) ∈ enumerate(sites)
        density = overlap_density_matrix(ϕ, ψ, site)
        @test size(density) == (physical_dimensions[position], physical_dimensions[position])
        @test density ≈ reference_overlap_density(ϕ, ψ, position)
        @test tr(density) ≈ dot(ϕ, ψ)
    end

    @test_throws ArgumentError overlap_density_matrix(ϕ, ψ, 99)
end

@testset "CPU gauge optimization preserves tensor layout and device" begin
    Random.seed!(0xc0ffee)
    sites = Site[1 // 2, 2, 7 // 2]
    physical_dimensions = [2, 3, 2]
    ψ_top = random_mps(Float64, sites, physical_dimensions, [1, 3, 4, 1])
    ψ_bot = random_mps(Float64, sites, physical_dimensions, [1, 2, 5, 1])

    gauges = optimize_gauges_for_overlaps!!(ψ_top, ψ_bot, 1e-10, 2)

    @test is_consistent(ψ_top)
    @test is_consistent(ψ_bot)
    @test which_device(ψ_top) == Set((:CPU,))
    @test which_device(ψ_bot) == Set((:CPU,))
    @test all(
        length(gauges[site]) == physical_dimensions[index] for
        (index, site) ∈ enumerate(sites)
    )
    @test all(gauge -> all(isfinite, gauge), values(gauges))
    @test isfinite(dot(ψ_top, ψ_bot))
end
