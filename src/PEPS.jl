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
        incoming = inneighbors(fg, v)
                   
        
        for u ∈ outgoing 
            e = SimpleEdge(v, u)
            println(e)
            println(get_prop(fg, e, :orientation))
            if get_prop(fg, e, :orientation) == "horizontal"
                println("Setting right")
                pc.right = last(get_prop(fg, e, :decomposition))
            else
                println("Setting down")
                pc.down = last(get_prop(fg, e, :decomposition))
            end 
        end

        for u ∈ incoming
            e = SimpleEdge(u, v)
            println(e)
            println(get_prop(fg, e, :orientation))
            if get_prop(fg, e, :orientation) == "horizontal"
                println("Setting left")
                pc.left = first(get_prop(fg, e, :decomposition))
            else
                println("Setting up")
                pc.up = first(get_prop(fg, e, :decomposition))
            end 
        end
       
        # open boundary conditions
        if !isdefined(pc, :left)
            println("left was undefined")
            pc.left = ones(1, size(pc.right, 1))
        end

        if !isdefined(pc, :right)
            println("right was undefined")
            pc.right = ones(size(pc.left, 2), 1)
        end

        if !isdefined(pc, :up)
            println("up was undefined")
            pc.up = ones(1, size(pc.down, 1))
        end

        if !isdefined(pc, :down)
            println("down was undefined")
            pc.down = ones(size(pc.up, 2), 1)
        end
        println(v)
        println(size(pc.left))
        println(size(pc.right))
        println(size(pc.up))
        println(size(pc.down))
        
        @cast pc.tensor[l, r, u, d, σ] |= pc.left[l, σ] * pc.right[σ, r] * pc.up[u, σ] * pc.down[σ, d]

        pc
    end
end