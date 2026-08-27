# Compare two benchmark result files; flag regressions > 5%.
#
#   julia benchmark/compare.jl results/OLD.json results/NEW.json

function parse_json(s::String)
    # minimal parser for the flat structure run.jl emits
    pos = Ref(1)
    skipws() = while pos[] <= lastindex(s) && isspace(s[pos[]])
        pos[] += 1
    end
    function value()
        skipws()
        c = s[pos[]]
        if c == '{'
            pos[] += 1
            d = Dict{String,Any}()
            skipws()
            s[pos[]] == '}' && (pos[] += 1; return d)
            while true
                skipws()
                k = value()
                skipws()
                pos[] += 1 # ':'
                d[k] = value()
                skipws()
                s[pos[]] == ',' ? pos[] += 1 : (pos[] += 1; return d)
            end
        elseif c == '['
            pos[] += 1
            a = Any[]
            skipws()
            s[pos[]] == ']' && (pos[] += 1; return a)
            while true
                push!(a, value())
                skipws()
                s[pos[]] == ',' ? pos[] += 1 : (pos[] += 1; return a)
            end
        elseif c == '"'
            i = pos[] + 1
            j = findnext('"', s, i)
            pos[] = j + 1
            return s[i:j-1]
        elseif c == 't'
            pos[] += 4
            return true
        elseif c == 'f'
            pos[] += 5
            return false
        else
            i = pos[]
            while pos[] <= lastindex(s) && (s[pos[]] in "+-.eE0123456789Inf")
                pos[] += 1
            end
            return parse(Float64, s[i:pos[]-1])
        end
    end
    value()
end

old = parse_json(read(ARGS[1], String))
new = parse_json(read(ARGS[2], String))

exit_code = 0
for (name, nc) in new["cases"]
    haskey(old["cases"], name) || continue
    oc = old["cases"][name]
    for metric in ("t_solve", "alloc_gib")
        o, n = oc["warm"][metric], nc["warm"][metric]
        Δ = (n - o) / o * 100
        flag = Δ > 5 ? " ⚠ REGRESSION" : ""
        Δ > 5 && global exit_code = 1
        println(rpad(name, 32), rpad(metric, 12), "old=", round(o, digits = 3),
            "  new=", round(n, digits = 3), "  Δ=", round(Δ, digits = 1), "%", flag)
    end
    eo, en = oc["warm"]["best_energy"], nc["warm"]["best_energy"]
    if !isapprox(eo, en; rtol = 1e-9)
        println(rpad(name, 32), "ENERGY MISMATCH old=", eo, " new=", en, " ⚠")
        global exit_code = 1
    end
end
exit(exit_code)
