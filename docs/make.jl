using Documenter
using Literate
using Printf
using Tutorials

models_src = joinpath(@__DIR__,"..","models")
models_dst = joinpath(@__DIR__,"src","models")

assets_src = joinpath(@__DIR__,"..","assets")
assets_dst = joinpath(@__DIR__,"src","assets")

Sys.rm(models_dst;recursive=true,force=true)
Sys.rm(assets_dst;recursive=true,force=true)

Sys.cp(models_src,models_dst)
Sys.cp(assets_src,assets_dst)

pages_dir = joinpath(@__DIR__,"src","pages")
notebooks_dir = joinpath(@__DIR__,"src","notebooks")

Sys.rm(pages_dir;recursive=true,force=true)
Sys.rm(notebooks_dir;recursive=true,force=true)

repo_src = joinpath(@__DIR__,"..","src")

# Add index.md file as introduction to navigation menu
pages = ["Introduction"=> "index.md"]

binder_logo = "https://mybinder.org/badge_logo.svg"
nbviwer_logo = "https://img.shields.io/badge/show-nbviewer-579ACA.svg"

for (i,(title,filename)) in enumerate(Tutorials.files)
  # Generate strings
  tutorial_prefix = string("t",@sprintf "%03d_" i)
  tutorial_title = string("# # Tutorial ", i, ": ", title)
  tutorial_file = string(tutorial_prefix,splitext(filename)[1])
  notebook_filename = string(tutorial_file, ".ipynb")
  binder_url = joinpath("@__BINDER_ROOT_URL__","notebooks", notebook_filename)
  nbviwer_url = joinpath("@__NBVIEWER_ROOT_URL__","notebooks", notebook_filename)
  binder_badge = string("# [![](",binder_logo,")](",binder_url,")")
  nbviwer_badge = string("# [![](",nbviwer_logo,")](",nbviwer_url,")")

  # Generate notebooks
  function preprocess_notebook(content)
    return string(tutorial_title, "\n\n", content)
  end
  Literate.notebook(joinpath(repo_src,filename), notebooks_dir; name=tutorial_file, preprocess=preprocess_notebook, documenter=false, execute=false)

  # Generate markdown
  function preprocess_docs(content)
    return string(tutorial_title, "\n", binder_badge, "\n", nbviwer_badge, "\n\n", content)
  end
  Literate.markdown(joinpath(repo_src,filename), pages_dir; name=tutorial_file, preprocess=preprocess_docs, codefence="```julia" => "```")

  # Generate navigation menu entries
  ordered_title = string(i, " ", title)
  path_to_markdown_file = joinpath("pages",string(tutorial_file,".md"))
  push!(pages, (ordered_title=>path_to_markdown_file))
end

makedocs(
    sitename = "Gridap tutorials",
    format = Documenter.HTML(),
    pages = pages
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.

deploydocs(
    repo = "github.com/gridap/Tutorials.git"
)
