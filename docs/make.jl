using Documenter
using Literate

Sys.rm("src/models";recursive=true)
Sys.rm("src/assets";recursive=true)

Sys.mkdir("src/models")
Sys.cp("../models/model-r1.png","src/models/model-r1.png")

Sys.cp("../assets","src/assets")

Literate.markdown("../src/t001_poisson.jl", "src/pages";codefence="```julia" => "```")

makedocs(
    sitename = "Gridap tutorials",
    format = Documenter.HTML(),
    pages =["Home"=> "index.md", "1 Poisson equation" => "pages/t001_poisson.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.

deploydocs(
    repo = "github.com/gridap/Tutorials.git"
)
