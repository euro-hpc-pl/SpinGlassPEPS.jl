export PepsTensor

mutable struct PepsTensor
    left::AbstractArray
    right::AbstractArray
    up::AbstractArray
    down::AbstractArray
    tensor::AbstractArray

    function PepsTensor(fg::MetaGraph, v::Int)
        n = nothing
        pc = new(n, n, n, n, n)
        outgoing = outneighbors(fg, v)
        incoming = inneighbors(fg, v)
                   
        
        for u ∈ outgoing 
            e = SimpleEdge(v, u)
            if get_prop(fg, e, :orientation) == "horizontal"
                pc.right = last(get_prop(fg, e, :decomposition))
            else
                pc.down = last(get_prop(fg, e, :decomposition))
            end 
        end

        for u ∈ incoming
            e = SimpleEdge(u, v)
            if get_prop(fg, e, :orientation) == "horizontal"
                pc.left = first(get_prop(fg, e, :decomposition))
            else
                pc.up = first(get_prop(fg, e, :decomposition))
            end 
        end
       
        # open boundary conditions
        if pc.left === nothing
            pc.left = ones(1, size(pc.right, 1))
        end

        if pc.right === nothing
            pc.right = ones(size(pc.left, 2), 1)
        end

        if pc.up === nothing
            pc.up = ones(1, size(pc.down, 1))
        end

        if pc.down === nothing
            pc.down = ones(size(pc.up, 2), 1)
        end

        pc.tensor[l, r, u, d, σ] |= pc.left[l, σ] * pc.right[σ, r] * pc.up[u, σ] * pc.down[σ, d]

        pc
    end
end