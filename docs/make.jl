using Documenter
using GOAT

makedocs(
    sitename = "GOAT.jl",
    format = Documenter.HTML(),
    modules = [GOAT],
    pages = [
        "Home" => "index.md",
        "Background" => "background/Overview.md",
        "Structure" => Any[
            "core/Ansatz.md",
            "core/DataStructures.md",
            "core/Methods.md",
            "core/ReferenceFrames.md",
        ],
        "Examples" => Any[
            "Single qubit control"=>"examples/example1.md",
            "Working with QuantumOptics.jl"=>"examples/example2.md",
        ],
        "Recommendations" => "recommendations.md",
        "API Reference" => "core/API_reference.md",
    ],
)

deploydocs(repo = "github.com/MCS-Quantum/GOAT.jl.git")
