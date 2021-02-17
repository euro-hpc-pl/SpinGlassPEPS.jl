function process_ref(ex)
    n = length(ex.args)
    args = Vector(undef, n)
    args[1] = ex.args[1]
    for i=2:length(ex.args)
        args[i] = :(state_to_ind($(ex.args[1]), $(i-1), $(ex.args[i])))
    end
    rex = Expr(:ref)
    rex.args = args
    rex
end

macro state(ex)
    if ex.head == :ref
        rex = process_ref(ex)
    elseif ex.head == :(=) || ex.head == Symbol("'")
        rex = copy(ex)
        rex.args[1] = process_ref(ex.args[1])
    else
        error("Not supported operation: $(ex.head)")
    end
    esc(rex)
end