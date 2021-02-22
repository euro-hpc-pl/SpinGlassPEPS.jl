_make_left_env(ψ::CuMPS, k::Int) = CUDA.ones(eltype(ψ), 1, 1, k)
_make_left_env_new(ψ::CuMPS, k::Int) = CUDA.ones(eltype(ψ), 1, k)
_make_LL(ψ::CuMPS, b::Int, k::Int, d::Int) = CUDA.zeros(eltype(ψ), b, b, k, d)
_make_LL_new(ψ::CuMPS, b::Int, k::Int, d::Int) = CUDA.zeros(eltype(ψ), b, k, d)

function CuMPS(ig::MetaGraph, control::MPSControl)

    Dcut = control.max_bond
    tol = control.var_ϵ
    max_sweeps = control.max_sweeps
    schedule = control.β
    @info "Set control parameters for MPS" Dcut tol max_sweeps

    rank = get_prop(ig, :rank)

    @info "Preparing Hadamard state as MPS"
    ρ = CuMPS(HadamardMPS(Float32, rank))
    is_right = true
    @info "Sweeping through β and σ" schedule
    for dβ ∈ schedule
        ρ = _apply_layer_of_gates(ig, ρ, control, dβ)
    end
    ρ
end

function CuMPS(ig::MetaGraph, control::MPSControl, type::Symbol)
    L = nv(ig)
    Dcut = control.max_bond
    tol = control.var_ϵ
    max_sweeps = control.max_sweeps
    dβ = control.dβ
    β = control.β
    @info "Set control parameters for MPS" Dcut tol max_sweeps
    rank = get_prop(ig, :rank)

    @info "Preparing Hadamard state as MPS"
    ρ = CuMPS(HadamardMPS(Float32, rank))
    is_right = true
    @info "Sweeping through β and σ" dβ

    if type == :log
        k = ceil(log2(β/dβ))
        dβmax = β/(2^k)
        ρ = _apply_layer_of_gates(ig, ρ, control, dβmax)
        for j ∈ 1:k
            ρ = multiply_purifications(ρ, ρ, L)
            if bond_dimension(ρ) > Dcut
                @info "Compresing MPS" bond_dimension(ρ), Dcut
                ρ = compress(ρ, Dcut, tol, max_sweeps)
                is_right = true
            end
        end
        ρ
    elseif type == :lin
        k = β/dβ
        dβmax = β/k
        ρ = _apply_layer_of_gates(ig, ρ, control, dβmax)
        ρ0 = copy(ρ)
        for j ∈ 1:k
            ρ = multiply_purifications(ρ, ρ0, L)
            if bond_dimension(ρ) > Dcut
                @info "Compresing MPS" bond_dimension(ρ), Dcut
                ρ = compress(ρ, Dcut, tol, max_sweeps)
                is_right = true
            end
        end
    end
    ρ

end

function _apply_projector!(ψ::CuMPS, i::Int)
     # Hack!!!
     M = ψ[i]
     D = cu(I(physical_dim(ψ, i)))

     @cast M̃[a, σ, (y, b)] := D[σ, y] * M[a, σ, b]
     ψ[i] = M̃
end
