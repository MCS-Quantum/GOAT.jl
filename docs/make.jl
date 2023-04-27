using Documenter
using GOAT

makedocs(
    sitename = "GOAT.jl",
    format = Documenter.HTML(),
    modules = [GOAT],
    pages = [
        "Home" => "index.md",
        "Background" => Any["Overview" => "background/Overview.md","background/Reference_frames.md"],
        "Package Core Structure" => Any["core/ControllableSystem.md","core/QOCProblem.md", "core/Ansatz.md","core/Methods.md"],
        "Examples" => Any["Single qubit control"=>"examples/example1.md", "Working with QuantumOptics.jl"=>"examples/example2.md"],
        "Recommendations" => "recommendations.md"
    ]
)

deploydocs(
    repo = "github.com/MCS-Quantum/GOAT.jl.git",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
