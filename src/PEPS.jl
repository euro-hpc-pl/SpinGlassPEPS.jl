export PepsTensor

mutable struct PepsTensor
    left::AbstractArray
    right::AbstractArray
    up::AbstractArray
    down::AbstractArray
    tensor::AbstractArray

    function PepsTensor(fg::MetaGraph, v::Int)
        pc = new()
        outgoing = outneighbors(fg, v)
        incoming = inneighbours(fg, v)
        
        for u ∈ outgoing
            if get_prop(fg, (v, u), :orientation) == "horizontal"
                pc.right = last(get_prop(fg, (v, u), :decomposition))
            else
                pc.down = last(get_prop(fg, (v, u), :decomposition))
            end 
        end

        for u ∈ incoming
            if get_prop(fg, (u, v), :orientation) == "horizontal"
                pc.left = first(get_prop(fg, (u, v), :decomposition))
            else
                pc.up = first(get_prop(fg, (u, v), :decomposition))
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