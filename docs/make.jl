using Documenter, FastSolversForWeightedTV

format = Documenter.HTML()

Introduction = "Introduction" => "index.md"
Installation = "Installation" => "installation.md"
GettingStarted = "Examples" => "examples.md"
MainFunctions = "Main functions" => "functions.md"

PAGES = [
    Introduction,
    Installation,
    GettingStarted,
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