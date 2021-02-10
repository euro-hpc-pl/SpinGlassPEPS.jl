# needs testing!!
function CuMPS(ig::MetaGraph, control::MPSControl)
    
    Dcut = control.max_bond
    tol = control.var_ϵ
    max_sweeps = control.max_sweeps
    schedule = control.β
    @info "Set control parameters for MPS" Dcut tol max_sweeps

    β = get_prop(ig, :β)
    rank = get_prop(ig, :rank)

    @assert β ≈ sum(schedule) "Incorrect β schedule."

    @info "Preparing Hadamard state as MPS"
    ρ = CuMPS(HadamardMPS(rank))
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
    @info "Set control parameters for MPS" Dcut tol max_sweeps
    dβ = get_prop(ig, :dβ)
    β = get_prop(ig, :β)
    rank = get_prop(ig, :rank)

    @info "Preparing Hadamard state as MPS"
    ρ = HadamardMPS(rank)
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