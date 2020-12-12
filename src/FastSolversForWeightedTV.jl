module FastSolversForWeightedTV

# Modules
using LinearAlgebra, VectorFields, DifferentialOperatorsForTV, CUDA, Flux

import Base: *
import Flux: update!

# Utils
include("./utils.jl")

# Types
include("./functional_abstract_types.jl")

# Collection of proximable functions
include("./proximable_functions.jl")

# Collection of differentiable functions
include("./differentiable_functions.jl")

# FISTA
include("./fista_solver.jl")

# L21 solvers
include("./proxyl21_solver.jl")

# TV solvers
include("./proxyTV_solver.jl")

end