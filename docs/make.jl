using CFHydrostatics
using Documenter

DocMeta.setdocmeta!(CFHydrostatics, :DocTestSetup, :(using CFHydrostatics); recursive=true)

makedocs(;
    modules=[CFHydrostatics],
    authors="The ClimFlows contributors",
    sitename="CFHydrostatics.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/CFHydrostatics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/CFHydrostatics.jl",
    devbranch="main",
)
