using Documenter, SpinGlassPEPS
using DocumenterTools: Themes
using SpinGlassTensors, SpinGlassNetworks, SpinGlassEngine

cd(@__DIR__)
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using Pkg
Pkg.activate(@__DIR__)
CI && Pkg.instantiate()

# %%
#download the themes
# for file in ("juliadynamics-lightdefs.scss", "juliadynamics-darkdefs.scss", "juliadynamics-style.scss")
#    download("https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/$file", joinpath(@__DIR__, file))
# end
# create the themes
# for w in ("light", "dark")
#     header = read(joinpath(@__DIR__, "juliadynamics-style.scss"), String)
#     theme = read(joinpath(@__DIR__, "juliadynamics-$(w)defs.scss"), String)
#     write(joinpath(@__DIR__, "juliadynamics-$(w).scss"), header*"\n"*theme)
# end
# compile the themes
# Themes.compile(joinpath(@__DIR__, "juliadynamics-light.scss"), joinpath(@__DIR__, "src/assets/themes/documenter-light.css"))
# Themes.compile(joinpath(@__DIR__, "juliadynamics-dark.scss"), joinpath(@__DIR__, "src/assets/themes/documenter-dark.css"))

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

# format = Documenter.LaTeX(platform="none")

const Page = Union{Pair{String, String}, Pair{String, Vector{Pair{String, String}}}}

_pages::Vector{Page} = [
    "Home" => "index.md",
    "Getting started" => "intro.md",
    "Brief description of the algorithm" => "algorithm.md",
    "More examples" => "examples.md"
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
    sgt_dir = joinpath(@__DIR__, "src", "sgt")
    try
        rm(sgt_dir; recursive = true)
    catch
    end
    sgt_docs = joinpath(dirname(dirname(pathof(SpinGlassTensors))), "docs")
    cp(joinpath(sgt_docs, "src"), sgt_dir; force = true)
    # Files in `sgn_docs` are probably in read-only mode (`0o444`). Let's give
    # ourselves write permission.
    chmod(sgt_dir, 0o777; recursive = true)
    make = read(joinpath(sgt_docs, "make.jl"), String)
     # Match from `_PAGES = [` until the start of in `# =====`
     s = strip(match(r"_pages = (\[.+?)\#"s, make)[1])
     # Rename every file to the `sgn/` directory.
     for m in eachmatch(r"\"([a-zA-Z\_\/]+?\.md)\"", s)
         s = replace(s, m[1] => "sgt/" * m[1])
     end
     push!(_pages, "SpinGlassTensors" => eval(Meta.parse(s)))
end

function add_sge_pages()
    sge_dir = joinpath(@__DIR__, "src", "sge")
    try
        rm(sge_dir; recursive = true)
    catch
    end
    sge_docs = joinpath(dirname(dirname(pathof(SpinGlassEngine))), "docs")
    cp(joinpath(sge_docs, "src"), sge_dir; force = true)
    # Files in `sgn_docs` are probably in read-only mode (`0o444`). Let's give
    # ourselves write permission.
    chmod(sge_dir, 0o777; recursive = true)
    make = read(joinpath(sge_docs, "make.jl"), String)
     # Match from `_PAGES = [` until the start of in `# =====`
     s = strip(match(r"_pages = (\[.+?)\#"s, make)[1])
     # Rename every file to the `sgn/` directory.
     for m in eachmatch(r"\"([a-zA-Z\_\/]+?\.md)\"", s)
         s = replace(s, m[1] => "sge/" * m[1])
     end
     push!(_pages, "SpinGlassEngine" => eval(Meta.parse(s)))
end

add_sgn_pages()
add_sge_pages()
add_sgt_pages()
push!(_pages, "More examples" => "examples.md")
# =================================
makedocs(
    # clean = true,
    format = format,
    modules=[SpinGlassPEPS, SpinGlassTensors, SpinGlassNetworks, SpinGlassEngine],
    sitename = "SpinGlassPEPS.jl",
    pages = _pages
)
# if "deploy" in ARGS
#     include("../../faketravis.jl")
# end

deploydocs(
    repo = "github.com/euro-hpc-pl/SpinGlassPEPS.jl.git",
)