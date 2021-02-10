#: Convex set types


export ConvexSet, project, project!, no_constraints, indicator


# "Projectionable" convex sets abstract type

"""
Expected behavior for convex sets: y = project!(x, C, y), y = Π_C(x)
"""
abstract type ConvexSet{T,N} end

function project(x::AbstractArray{T,N}, C::ConvexSet{T,N}) where {T,N}
    y = similar(x)
    return project!(x, C, y)
end


# Concrete types

## No constraint set

struct NoConstraints{T,N}<:ConvexSet{T,N} end

no_constraints(T::DataType, N::Int64) = NoConstraints{T,N}()

function project!(x::Array{T,N}, ::NoConstraints{T,N}, y::Array{T,N}) where {T,N}
    @inbounds for i = 1:length(x)
        y[i] = x[i]
    end
    return y
end

function project!(x::CuArray{T,N}, ::NoConstraints{T,N}, y::CuArray{T,N}) where {T,N}
    y .= x
    return y
end


# Indicator functions

"""
Indicator function δ_C(x) = {0, if x ∈ C; ∞, otherwise} for convex sets C
"""
struct Indicator{T,N}<:ProximableFunction{T,N}
    C::ConvexSet{T,N}
end

proxy!(y::DT, ::Any, δ::Indicator{T,N}, x::DT) where {T,N,DT<:AbstractArray{T,N}} = project!(y, δ.C, x)

indicator(C::ConvexSet{T,N}) where {T,N} = Indicator{T,N}(C)