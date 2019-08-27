# Gridap tutorials

[![Build Status](https://travis-ci.com/gridap/Gridap.jl.svg?branch=master)](https://travis-ci.com/gridap/Tutorials)

## How to start

Clone the repository
```
$ git clone https://github.com/gridap/Tutorials.git
```

Move into the folder and open a julia REPL setting the current folder as the project environment. NOTE: use at least Julia 1.1
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

Open the notebooks
```
# Type Ctrl+C to get back to command mode
julia> using IJulia
julia> notebook(dir=pwd())
```
This will open a browser window. Navigate to the `notebooks` folder and open the tutorial you want. Enjoy!

## How to pull the latest version of the tutorials

Go to the Tutorials repo folder and git pull
```
$ git pull
```
Open Julia REPL
```
$ julia --project=.

```

and instantiate the environment again
```
# Type ] to enter in pkg mode
Tutorials> instantiate
```

Done!

