using Documenter, SpinGlassEngine

_pages = [
    "User Guide" => "guide.md",
    "Tensor network" => "peps.md",
    "Search parameters" => "params.md",
    "Low energy spectrum" => "search.md",
]
# "API Reference for auxiliary functions" => "api.md",
# ]
# ============================

format =
    Documenter.HTML(edit_link = "master", prettyurls = get(ENV, "CI", nothing) == "true")

# format = Documenter.LaTeX(platform="none")

makedocs(
    sitename = "SpinGlassEngine.jl",
    modules = [SpinGlassEngine],
    pages = _pages,
    format = format,
)
