using Documenter
using FastSolversForWeightedTV

makedocs(
    sitename = "FastSolversForWeightedTV",
    format = Documenter.HTML(),
    modules = [FastSolversForWeightedTV]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
