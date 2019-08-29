# Introduction

Welcome to the tutorial pages of the [Gridap.jl](https://github.com/gridap/Gridap.jl) project.

## How to start

The easiest way to start using the tutorials is to click in one of the following links to start reading the html version of the tutorial you want.

```@contents
Depth = 1
```

## Jupyter notebooks

In addition, the tutorials are available as jupyter notebooks. You can access them in three different ways:

- By running the notebooks locally. A working installation of Julia in the system is required. See instructions below. **This is the recommended way to follow the tutorials**. In particular, it allows to inspect the generated results with Paraview.

- By running the notebook remotely via [binder](https://mybinder.org). In that case, go to the desired tutorial and click the icon ![](https://mybinder.org/badge_logo.svg). No local installation of Julia needed.

- By reading a non-interactive version of the notebook via [nbviewer](https://nbviewer.jupyter.org/). In that case, go to the desired tutorial and click the icon ![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)

## How to run the notebooks locally

Clone the repository
```
$ git clone https://github.com/gridap/Tutorials.git
```

Move into the folder and open a Julia REPL setting the current folder as the project environment. NOTE: use at least Julia 1.1
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
Tutorials> instantiate
```

Build the notebooks
```
Tutorials> build
```

Open the notebooks
```
# Type Ctrl+C to get back to command mode
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
and instantiate and build the environment again
```
# Type ] to enter in pkg mode
Tutorials> instantiate
Tutorials> build
```

Done!
