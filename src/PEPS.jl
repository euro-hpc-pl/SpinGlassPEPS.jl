export PepsTensor

mutable struct PepsTensor
    left::AbstractArray
    right::AbstractArray
    up::AbstractArray
    down::AbstractArray
    loc::AbstractArray
    tensor::AbstractArray

    function PepsTensor(fg::MetaDiGraph, v::Int)
        pc = new()
        pc.loc = get_prop(fg, v, :local_exp)
        
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

        #@infiltrate
        println(v)
        println(size(pc.left))
        println(size(pc.right))
        println(size(pc.up))
        println(size(pc.down))

        @cast pc.tensor[l, r, u, d, σ] |= pc.loc[σ] * pc.left[l, σ] * pc.right[σ, r] * pc.up[u, σ] * pc.down[σ, d]

        pc
    end
end