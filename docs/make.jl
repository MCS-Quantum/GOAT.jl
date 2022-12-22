using Documenter
using GOAT

makedocs(
    sitename = "GOAT",
    format = Documenter.HTML(),
    modules = [GOAT]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
