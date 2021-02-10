#: Optimization functions and solvers

export DiffPlusProxFunction, Options, OptFISTA, optFISTA, solverFISTA!, solverFISTA


# Optimization type

struct DiffPlusProxFunction{T,N}
    f::DifferentiableFunction{T,N}
    g::ProximableFunction{T,N}
end

Base.:+(f::DifferentiableFunction{T,N}, g::ProximableFunction{T,N}) where {T,N} = DiffPlusProxFunction{T,N}(f, g)


# Options type for FISTA

abstract type Options{T} end

struct OptFISTA{T}<:Options{T}
    steplength::T
    niter::Int64
    tol_x::Union{Nothing,T}
    nesterov::Bool
end

optFISTA(; steplength::T=T(1), niter::Int64=100, tol_x::Union{Nothing,T}=nothing, nesterov::Bool=true) where T = OptFISTA{T}(steplength, niter, tol_x, nesterov)


# Generic FISTA solver

"""
Solver for the regularized problem via FISTA-like gradient-projections:
```math
min_x f(x)+g(x)
```
Reference: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
function solverFISTA!(fun::DiffPlusProxFunction{T,N}, x0::CuArray{T,N}, opt::OptFISTA{T}) where {T,N}

    # Initialization
    x = similar(x0)    # momentum pre-allocation
    xtmp = similar(x0) # temporary pre-allocation
    ∇f = similar(x0)   # gradient pre-allocation
    t0 = T(1)
    flag_conv = false

    # Optimization loop
    for i = 1:opt.niter

        fval = grad!(fun.f, x0, ∇f)                # Compute gradient
        xtmp .= x0-opt.steplength*∇f
        proxy!(xtmp, opt.steplength, fun.g, x)     # Compute proxy
        if opt.nesterov                            # > Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # - compute dynamic step size
            x .= x+(t0-T(1))/t*(x-x0)              # - update momentum
            t0 = t                                 # - update step size
        end                                        # <

        # Update unknowns
        opt.tol_x !== nothing && (flag_conv = norm(x-x0)<=opt.tol_x*norm(x0))
        x0 .= x
        
        # Convergence check
        flag_conv && break

    end

    # Return output
    return x

end

function solverFISTA!(fun::DiffPlusProxFunction{T,N}, x0::Array{T,N}, opt::OptFISTA{T}) where {T,N}

    # Initialization
    x = similar(x0)  # momentum pre-allocation
    xtmp = similar(x0)  # momentum pre-allocation
    ∇f = similar(x0) # gradient pre-allocation
    t0 = T(1)
    flag_conv = false

    # Optimization loop
    for i = 1:opt.niter

        fval = grad!(fun.f, x0, ∇f)                # Compute gradient
        xtmp .= x0-opt.steplength*∇f
        proxy!(xtmp, opt.steplength, fun.g, x)     # Compute proxy
        if opt.nesterov                            # > Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # - compute dynamic step size
            w = (t0-T(1))/t
            # for i = 1:length(x)
            #     x[i] = x[i]+w*(x[i]-x0[i])                          # - update momentum
            # end
            x .= x+w*(x-x0)
            t0 = t                                 # - update step size
        end                                        # <

        # Update unknowns
        opt.tol_x !== nothing && (flag_conv = norm(x-x0)<=opt.tol_x*norm(x0))
        x0 .= x
        
        # Convergence check
        flag_conv && break

    end

    # Return output
    return x

end

function solverFISTA(fun::DiffPlusProxFunction{T,N}, x0::AbstractArray{T,N}, opt::OptFISTA{T}) where {T,N}
    x0 = copy(x0)
    return solverFISTA!(fun, x0, opt)
end