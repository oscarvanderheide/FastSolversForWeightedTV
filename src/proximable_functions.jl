#: Proximable functions


export VectorFieldNorm, IndicatorFunction, projection_L2Inf


# Indicator functions

"""
Indicator function δ_C(x) = {0, if x ∈ C; ∞, otherwise}
"""
struct IndicatorFunction{T} <: ProximableFunction{AbstractField2D{T}}
    projection::Function # Π: x -> y ∈ C
end
proxy(λ::T, g::IndicatorFunction{T}, x::AbstractField2D{T}) where T = g.projection(x)

function projection_L2Inf(v::VectorField2D{T}) where T
    ptnorm_v = ptnorm(v; p=2)
    norm(ptnorm_v; p=Inf) <= T(1) ? (return (T(0), v)) : (return (Inf, v*( (ptnorm_v >= T(1)) /ptnorm_v+(ptnorm_v < T(1)) )))
end


# Vector field norm

struct VectorFieldNorm{T,p1,p2} <: ProximableFunction{VectorField2D{T}} end
function proxy(λ::T, g::VectorFieldNorm{T,2,1}, v::VectorField2D{T}; outval::Bool=true) where T
    ptnorm_v = ptnorm(v; p=2)
    y = (T(1)-λ/ptnorm_v)*(ptnorm_v >= λ)*v
    outval ? (gval = norm(toArray(ptnorm_v); p=1)) : (gval = nothing)
    return gval, y
end

VectorFieldNorm(T::DataType, p1::Int64, p2::Int64) = VectorFieldNorm{T,p1,p2}(T(1))

conjugate(g::VectorFieldNorm{T,2,1}) where T = IndicatorFunction{T}(projection_L2Inf)