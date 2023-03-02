using Documenter, SpinGlassPEPS
using DocumenterTools: Themes
using SpinGlassTensors, SpinGlassNetworks, SpinGlassEngine
using MetaGraphs

cd(@__DIR__)
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using Pkg
Pkg.activate(@__DIR__)
CI && Pkg.instantiate()

# %%
#download the themes
for file in ("juliadynamics-lightdefs.scss", "juliadynamics-darkdefs.scss", "juliadynamics-style.scss")
   download("https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/$file", joinpath(@__DIR__, file))
end
# create the themes
for w in ("light", "dark")
    header = read(joinpath(@__DIR__, "juliadynamics-style.scss"), String)
    theme = read(joinpath(@__DIR__, "juliadynamics-$(w)defs.scss"), String)
    write(joinpath(@__DIR__, "juliadynamics-$(w).scss"), header*"\n"*theme)
end
# compile the themes
Themes.compile(joinpath(@__DIR__, "juliadynamics-light.scss"), joinpath(@__DIR__, "src/assets/themes/documenter-light.css"))
Themes.compile(joinpath(@__DIR__, "juliadynamics-dark.scss"), joinpath(@__DIR__, "src/assets/themes/documenter-dark.css"))

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

const Page = Union{Pair{String, String}, Pair{String, Vector{Pair{String, String}}}}

_pages::Vector{Page} = [
    "Home" => "index.md",
    "Example" => "examples.md"
]


function add_sgn_pages()
    sgn_dir = joinpath(@__DIR__, "src", "sgn")
    try
        rm(sgn_dir; recursive = true)
    catch
    end
    sgn_docs = joinpath(dirname(dirname(pathof(SpinGlassNetworks))), "docs")
    cp(joinpath(sgn_docs, "src"), sgn_dir; force = true)
    # Files in `sgn_docs` are probably in read-only mode (`0o444`). Let's give
    # ourselves write permission.
    chmod(sgn_dir, 0o777; recursive = true)
    make = read(joinpath(sgn_docs, "make.jl"), String)
     # Match from `_PAGES = [` until the start of in `# =====`
     s = strip(match(r"_pages = (\[.+?)\#"s, make)[1])
     # Rename every file to the `sgn/` directory.
     for m in eachmatch(r"\"([a-zA-Z\_\/]+?\.md)\"", s)
         s = replace(s, m[1] => "sgn/" * m[1])
     end
     push!(_pages, "SpinGlassNetworks" => eval(Meta.parse(s)))
end

function add_sgt_pages()

end

function add_sge_pages()

end

add_sgn_pages()
# =================================
makedocs(
    clean = true,
    format = format,
    modules=[SpinGlassPEPS, SpinGlassTensors, SpinGlassNetworks, SpinGlassEngine],
    sitename = "SpinGlassPEPS.jl",
    authors = "Krzysztof Domino, Bartłomiej Gardas, Konrad Jałowiecki, Łukasz Pawela, Marek Rams, Anna Dziubyna",
    pages = _pages,
    sidebar_sitename = false
)

# if "deploy" in ARGS
#     include("../../faketravis.jl")
# end

# deploydocs(
#     #root = "github.com/iitis/SpinGlassPEPS.jl/tree/ad/docs",
#     #repo = "github.com/iitis/SpinGlassPEPS.jl.git",
#     repo = "github.com/iitis/SpinGlassPEPS.jl.git",
#     branch = "ad/docs",
#     devbranch = "ad/docs"
#     #devbranch = "lp/docs-example",
#     #branch = "ad/docs"
# )