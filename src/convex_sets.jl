#: Convex set types


export ConvexSet, project, project!, no_constraints, indicator


# "Projectionable" convex sets abstract type

"""
Expected behavior for convex sets: y = project!(x, C, y), y = Π_C(x)
"""
abstract type ConvexSet{DT} end

function project(x::DT, C::ConvexSet{DT}) where {T,N,DT<:AbstractArray{T,N}}
    y = similar(x)
    return project!(x, C, y)
end


# Concrete types

## No constraint set

struct NoConstraints{DT}<:ConvexSet{DT} end

no_constraints(DT::DataType) = NoConstraints{DT}()
no_constraints(T::DataType, N::Int64) = NoConstraints{AbstractArray{T,N}}()

function project!(x::DT, C::NoConstraints{DT}, y::DT) where {T,N,DT<:AbstractArray{T,N}}
    y .= x
    return y
end


# Indicator functions

"""
Indicator function δ_C(x) = {0, if x ∈ C; ∞, otherwise} for convex sets C
"""
struct Indicator{DT}<:ProximableFunction{DT}
    C::ConvexSet{DT}
end

proxy!(y::DT, ::Any, δ::Indicator{DT}, x::DT) where {T,N,DT<:AbstractArray{T,N}} = project!(y, δ.C, x)

indicator(C::ConvexSet{DT}) where DT = Indicator{DT}(C)