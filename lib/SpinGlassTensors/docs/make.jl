using Documenter, SpinGlassTensors

_pages = [
    "User guide" => "index.md",
    "Matrix Product States and Matrix Product Operators" => "mpo.md",
]
# ============================

format =
    Documenter.HTML(edit_link = "master", prettyurls = get(ENV, "CI", nothing) == "true")

# format = Documenter.LaTeX(platform="none")

makedocs(
    sitename = "SpinGlassTensors.jl",
    modules = [SpinGlassTensors],
    pages = _pages,
    format = format,
)
