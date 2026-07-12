using Documenter, SpinGlassExhaustive

using Pkg
Pkg.activate("..")
push!(LOAD_PATH,"../src/")


_pages = [
    # "Introduction" => "index.md",
    # "User Guide" => "guide.md",   
    "Manual" => Any[
        "man/quickstart.md",
        "man/how_use.md",
        "man/details.md",
        "man/integration.md"
    ],
    "API Reference" => "api.md",
    # "Library" => "lib/SpinGlassExhaustive.md"
]
# ============================

makedocs(
    # root = "../",
    # source = "src",
    modules = Module[SpinGlassExhaustive],
    sitename="SpinGlassExhaustive.jl",
    authors = "Dariusz Kurzyk, Łukasz Pawela",
    pages = _pages,
#    format = Documenter.HTML(prettyurls = false),
    format = Documenter.LaTeX(platform="none"),
    expandfirst = []
    )

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    target = "build",
    repo = "github.com/euro-hpc-pl/SpinGlassExhaustive.jl.git"
)
