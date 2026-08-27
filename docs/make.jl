using Documenter
using SpinGlassPEPS
using SpinGlassPEPS:
    SpinGlassEngine, SpinGlassExhaustive, SpinGlassNetworks, SpinGlassTensors

const CI =
    get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing

const format = Documenter.HTML(
    edit_link = "master",
    prettyurls = !("local" in ARGS),
    mathengine = MathJax3(
        Dict(
            :tex => Dict(
                "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                "processEscapes" => true,
                "macros" => Dict(
                    "bra" => ["{\\langle #1|}", 1],
                    "ket" => ["{| #1\\rangle}", 1],
                    "ketbra" => ["{\\left| #1 \\rangle \\langle #2 \\right|}", 2],
                    "braket" => ["{\\langle #1|#2\\rangle}", 2],
                    "Tr" => ["\\mathrm{Tr}", 0],
                    "tr" => ["\\Tr", 0],
                    "ee" => ["\\mathrm{e}"],
                    "ii" => ["\\mathrm{i}"],
                    "dd" => ["\\mathrm{d}"],
                    "1" => ["{\\mathbb{1}}"],
                ),
            ),
        ),
    ),
)

const pages = [
    "Home" => "index.md",
    "Getting started" => "intro.md",
    "Algorithm" => "algorithm.md",
    "Examples" => "examples.md",
    "Models and lattices" => [
        "Overview" => "sgn/userguide.md",
        "Ising models" => "sgn/ising.md",
        "Lattice geometries" => "sgn/lattice.md",
        "Potts Hamiltonians" => "sgn/clh.md",
        "Local dimensional reduction" => "sgn/bp.md",
        "API reference" => "sgn/api.md",
    ],
    "Solver" => [
        "Overview" => "sge/guide.md",
        "PEPS construction" => "sge/peps.md",
        "Search parameters" => "sge/params.md",
        "Low-energy search" => "sge/search.md",
        "Concurrent sweeps and error control" => "sge/sweep.md",
        "API reference" => "sge/api.md",
    ],
    "Tensor backend" => [
        "Overview" => "sgt/index.md",
        "MPS and MPO" => "sgt/mpo.md",
        "API reference" => "sgt/api.md",
    ],
    "Exhaustive search" => [
        "Overview" => "sgx/index.md",
        "Quickstart" => "sgx/man/quickstart.md",
        "Usage" => "sgx/man/how_use.md",
        "GPU kernels" => "sgx/man/details.md",
        "Solver integration" => "sgx/man/integration.md",
        "API reference" => "sgx/api.md",
    ],
]

makedocs(
    format = format,
    modules = [
        SpinGlassPEPS,
        SpinGlassTensors,
        SpinGlassNetworks,
        SpinGlassExhaustive,
        SpinGlassEngine,
    ],
    sitename = "SpinGlassPEPS.jl",
    pages = pages,
)

deploydocs(repo = "github.com/euro-hpc-pl/SpinGlassPEPS.jl.git")
