export PepsTensor, MPO
#=
mutable struct PepsTensor
    tag::Int
    nbrs::Dict{String, Int}
    left::AbstractArray
    right::AbstractArray
    up::AbstractArray
    down::AbstractArray
    loc::AbstractArray
    tensor::AbstractArray

    function PepsTensor(fg::MetaDiGraph, v::Int)
        pc = new(v)
        pc.nbrs = Dict()
        pc.loc = get_prop(fg, v, :local_exp)
        
        outgoing = outneighbors(fg, v)
        incoming = inneighbors(fg, v)
                    
        for u ∈ outgoing 
            e = SimpleEdge(v, u)
            if get_prop(fg, e, :orientation) == "horizontal"
                pc.right = first(get_prop(fg, e, :decomposition))
                push!(pc.nbrs, "h_out" => u)
            else
                pc.down = first(get_prop(fg, e, :decomposition))
                push!(pc.nbrs, "v_out" => u)
            end 
        end

        for u ∈ incoming
            e = SimpleEdge(u, v)
            if get_prop(fg, e, :orientation) == "horizontal"
                pc.left = last(get_prop(fg, e, :decomposition))
                push!(pc.nbrs, "h_in" => u)
            else
                pc.up = last(get_prop(fg, e, :decomposition))
                push!(pc.nbrs, "v_in" => u)
            end 
        end
       
        # open boundary conditions
        if !isdefined(pc, :left)
            pc.left = ones(1, size(pc.right, 1))
        end

        if !isdefined(pc, :right)
            pc.right = ones(size(pc.left, 2), 1)
        end

        if !isdefined(pc, :up)
            pc.up = ones(1, size(pc.down, 1))
        end

        if !isdefined(pc, :down)
            pc.down = ones(size(pc.up, 2), 1)
        end

        @cast pc.tensor[l, r, u, d, σ] |= pc.loc[σ] * pc.left[l, σ] * pc.right[σ, r] * pc.up[u, σ] * pc.down[σ, d]

        pc
    end
end
=#
Base.size(A::PepsTensor) = size(A.tensor)

#=
function MPO(fg::MetaDiGraph, dim::Symbol=:r, i::Int; T::DataType=Float64)
    @assert dir ∈ (:r, :c)

    m, n = size(fg)
    idx = LinearIndices((1:m, 1:n))
    chain = dim == :r ? fg[idx[:, i]] : fg[idx[i, :]] 

    ψ = MPO(T, length(chain))

    for (j, v) ∈ enumerate(chain)
        ψ[j] = PepsTensor(fg, v).tensor
    end
    ψ
end

function MPS(fg::MetaDiGraph, which::Symbol=:d; T::DataType=Float64)
    @assert which ∈ (:l, :r, :u, :d)

    #ϕ = MPO()

    for (j, v) ∈ enumerate(_row(fg, 1))
        ψ[j] = dropdims(PepsTensor(fg, v).tensor, dims=4)
    end

    # TBW 

    ψ
end
=#

mutable struct PepsTensor
    tag::Int
    nbrs::Dict{String, Int}
    left::AbstractArray
    right::AbstractArray
    up::AbstractArray
    down::AbstractArray
    loc::AbstractArray
    tensor::AbstractArray

    function PepsTensor(fg::MetaDiGraph, v::Int)
        pc = new(v)
        pc.nbrs = Dict()
        pc.loc = get_prop(fg, v, :local_exp)
        
        outgoing = outneighbors(fg, v)
        incoming = inneighbors(fg, v)
                    
        for u ∈ outgoing 
            e = SimpleEdge(v, u)
            if get_prop(fg, e, :orientation) == "horizontal"
                pc.right = first(get_prop(fg, e, :decomposition))
                push!(pc.nbrs, "h_out" => u)
            else
                pc.down = first(get_prop(fg, e, :decomposition))
                push!(pc.nbrs, "v_out" => u)
            end 
        end

        for u ∈ incoming
            e = SimpleEdge(u, v)
            if get_prop(fg, e, :orientation) == "horizontal"
                pc.left = last(get_prop(fg, e, :decomposition))
                push!(pc.nbrs, "h_in" => u)
            else
                pc.up = last(get_prop(fg, e, :decomposition))
                push!(pc.nbrs, "v_in" => u)
            end 
        end
       
        # open boundary conditions
        if !isdefined(pc, :left)
            pc.left = ones(1, size(pc.right, 1))
        end

        if !isdefined(pc, :right)
            pc.right = ones(size(pc.left, 2), 1)
        end

        if !isdefined(pc, :up)
            pc.up = ones(1, size(pc.down, 1))
        end

        if !isdefined(pc, :down)
            pc.down = ones(size(pc.up, 2), 1)
        end

        @cast pc.tensor[l, r, u, d, σ] |= pc.loc[σ] * pc.left[l, σ] * pc.right[σ, r] * pc.up[u, σ] * pc.down[σ, d]

        pc
    end
end

function generate_tensor(ng::NetworkGraph, v::Int)
    loc_en = get_prop(ng.graph, v, :loc_en)
    tensor = exp.(-ng.β .* loc_en)
    for w ∈ ng.nbrs[v]
        if has_edge(ng.graph, w, v)
            pw, e, pv = get_prop(ng.graph, w, v, :decomposition)
            @cast tensor[σ, ..., γ] |= tensor[σ, ...] * pv[γ, σ]
        elseif has_edge(ng.graph, v, w)
            pv, e, pw = get_prop(ng.graph, v, w, :decomposition)
            @cast tensor[σ, ..., γ] |= tensor[σ, ...] * pv[σ, γ]
        else 
            pv = ones(size(loc_en), 1)
            @cast tensor[σ, ..., γ] |= tensor[σ, ...] * pv[σ, γ]
        end
    end
    tensor
end

function generate_tensor(ng::NetworkGraph, v::Int, w::Int)
    if has_edge(ng.graph, w, v)
        _, e, _ = get_prop(ng.graph, w, v, :decomposition)
        tensor = exp.(-ng.β .* e') #?transpose e
    elseif has_edge(ng.graph, v, w)
        _, e, _ = get_prop(ng.graph, v, w, :decomposition)
        tensor = exp.(-ng.β .* e) 
    else 
        tensor = ones(1, 1)
    end
    tensor
end