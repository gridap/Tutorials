# Introduction

Welcome to the tutorial pages of the [Gridap.jl](https://github.com/gridap/Gridap.jl) project.

## Contents

```@contents
Depth = 1
```

## How to start

There are different ways to use the tutorials:

- **[Recommended]** Reading the html version of the tutorials. This is the recommended way if you want rapid access to the material with no setup steps. Simply click in one of the links in the [Contents](@ref) section.
- **[Recommended]** Running the Jupyter notebooks locally. A working installation of Julia in the system is required. See instructions in the [How to run the notebooks locally](@ref) section. This is the recommended way to follow the tutorials if you want to run the code and inspect the generated results with Paraview.
- Running the notebook remotely via [binder](https://mybinder.org). In that case, go to the desired tutorial and click the icon ![](https://mybinder.org/badge_logo.svg). No local installation of Julia needed.
- Reading a non-interactive version of the notebook via [nbviewer](https://nbviewer.jupyter.org/). In that case, go to the desired tutorial and click the icon ![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)

## How to run the notebooks locally

Clone the repository
```
$ git clone https://github.com/gridap/Tutorials.git
```

Move into the folder and open a Julia REPL setting the current folder as the project environment. 
```
$ cd Tutorials
$ julia --project=.
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.1.0 (2019-01-21)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> 

```

Instantiate the environment. This will automatically download all required packages.
```
# Type ] to enter in pkg mode
(Tutorials) pkg> instantiate
```

Build the notebooks
```
# Type Ctrl+C to get back to command mode
julia> include("deps/build.jl")
```

Open the notebooks
```
julia> using IJulia
julia> notebook(dir=pwd())
```
This will open a browser window. Navigate to the `notebooks` folder and open the tutorial you want. Enjoy!

## How to pull the latest version of the tutorials

If you have cloned the repository a while ago, you can update to the newest version with these steps.

Go to the Tutorials repo folder and git pull
```
$ git pull
```
Open Julia REPL
```
$ julia --project=.

```
and instantiate the environment and build the notebooks again
```
# Type ] to enter in pkg mode
(Tutorials) pkg> instantiate

# Type Ctrl+C to get back to command mode
julia> include("deps/build.jl")
```

Done!
