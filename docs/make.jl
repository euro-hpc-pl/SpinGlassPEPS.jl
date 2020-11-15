using Documenter, SpinGlassPEPS

format = Documenter.HTML(
    edit_link = "master",
    prettyurls = !("local" in ARGS),
    mathengine = mathengine = MathJax3(Dict(
        :tex=>Dict(
            "inlineMath"=>[ ["\$","\$"], ["\\(","\\)"] ],
            "processEscapes"=>true,
            "macros"=>Dict(
                "bra"=> ["{\\langle #1|}",1],
                "ket"=> ["{| #1\\rangle}",1],
                "ketbra"=> ["{\\left| #1 \\rangle \\langle #2 \\right|}",2],
                "braket"=> ["{\\langle #1|#2\\rangle}",2],
                "Tr"=> ["\\mathrm{Tr}",0],
                "tr"=> ["\\Tr",0],
                "ee"=> ["\\mathrm{e}"],
                "ii"=> ["\\mathrm{i}"],
                "dd"=> ["\\mathrm{d}"],
                "1"=> ["{\\mathbb{1}}"]
            )
        )
    ))
)

makedocs(
    clean = true,
    format = format,
    sitename = "SpinGlassPEPS.jl",
    authors = "Krzysztof Domino, Bartłomiej Gardas, Konrad Jałowiecki, Łukasz Pawela, Marek Rams",
    pages = [
        "Home" => "index.md",
        "Library" => "lib/SpinGlassPEPS.md"
    ]
)

deploydocs(
    repo = "github.com/iitis/SpinGlassPEPS.jl.git"
)