module TotalVariationRegularization

# Standard precision
T  = Float32
# T = Float64 # uncomment for double precision
CT = Complex{T}

# Custom types
include("vectorfield.jl")

# Linear operator wrappers
include("conv_utils.jl")
include("diffops.jl")

# Proximal operator solvers
include("proxop_solver.jl")

# Gradient preconditioning utils
include("grad_prec.jl")

# CUDA utils
include("cuda.jl")

end