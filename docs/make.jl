using Documenter, FastSolversForWeightedTV

format = Documenter.HTML()

Introduction = "Introduction" => "index.md"
MainFunctions = "Main functions" => "functions.md"

PAGES = [
    Introduction,
    MainFunctions
    ]

makedocs(
    modules = [FastSolversForWeightedTV],
    sitename = "FastSolversForWeightedTV.jl",
    authors = "Gabrio Rizzuti",
    format = format,
    checkdocs = :exports,
    pages = PAGES
)

# makedocs(
#     sitename = "FastSolversForWeightedTV",
#     format = Documenter.HTML(),
#     modules = [FastSolversForWeightedTV],
#     pages = [
#         "index.md",
#         "Installation instructions" => "installation.md",
#         "Main functions" => "functions.md"#,
#         # "Subsection" => [
#         #     ...
#         # ]
#     ]
# )

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
