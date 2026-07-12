using Documenter, SpinGlassNetworks

_pages = [
    "User guide" => "userguide.md",
    "Ising graph" => "ising.md",
    "Lattice geometries" => "lattice.md",
    "Potts hamiltonian" => "clh.md",
    "Local dimensional reduction" => "bp.md",
]
#     "API Reference for auxiliary functions" => "api.md",
# ]

# ============================
format =
    Documenter.HTML(edit_link = "master", prettyurls = get(ENV, "CI", nothing) == "true")

# format = Documenter.LaTeX(platform="none")

makedocs(
    sitename = "SpinGlassNetworks.jl",
    modules = [SpinGlassNetworks],
    pages = _pages,
    format = format,
)
