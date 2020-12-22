export PepsTensor

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
            #pc.left = ones(1, size(pc.right, 1))
            @cast pc.tensor[r, u, d, σ] |= pc.loc[σ] * pc.right[σ, r] * pc.up[u, σ] * pc.down[σ, d]
        end

        if !isdefined(pc, :right)
            #pc.right = ones(size(pc.left, 2), 1)
            @cast pc.tensor[l, u, d, σ] |= pc.loc[σ] * pc.left[l, σ] * pc.up[u, σ] * pc.down[σ, d]
        end

        if !isdefined(pc, :up)
            #pc.up = ones(1, size(pc.down, 1))
            @cast pc.tensor[l, r, d, σ] |= pc.loc[σ] * pc.left[l, σ] * pc.right[σ, r] * pc.down[σ, d]
        end

        if !isdefined(pc, :down)
            #pc.down = ones(size(pc.up, 2), 1)
            @cast pc.tensor[l, r, u, σ] |= pc.loc[σ] * pc.left[l, σ] * pc.right[σ, r] * pc.up[u, σ]
        end

        #@cast pc.tensor[l, r, u, d, σ] |= pc.loc[σ] * pc.left[l, σ] * pc.right[σ, r] * pc.up[u, σ] * pc.down[σ, d]

        pc
    end
end

Base.size(A::PepsTensor) = size(A.tensor)