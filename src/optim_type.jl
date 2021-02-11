#: Optimizable types

export OptimizableFunction, DiffPlusProxFunction


# Abstract type

abstract type OptimizableFunction{T,N} end


# Concrete types

struct DiffPlusProxFunction{T,N}<:OptimizableFunction{T,N}
    f::DifferentiableFunction{T,N}
    g::ProximableFunction{T,N}
end

Base.:+(f::DifferentiableFunction{T,N}, g::ProximableFunction{T,N}) where {T,N} = DiffPlusProxFunction{T,N}(f, g)