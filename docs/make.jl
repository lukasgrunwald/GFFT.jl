using GFFT
using Documenter

DocMeta.setdocmeta!(GFFT, :DocTestSetup, :(using GFFT); recursive=true)

makedocs(;
    modules=[GFFT],
    authors="Lukas Grunwald",
    sitename="GFFT.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lukasgrunwald.github.io/GFFT.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lukasgrunwald/GFFT.jl",
    devbranch="master",
)
