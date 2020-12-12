#: Proximable functions


export VectorFieldNorm, IndicatorFunction, projection_L2Inf, project, project!


# Indicator functions

"""
Indicator function δ_C(x) = {0, if x ∈ C; ∞, otherwise}
"""
struct IndicatorFunction{DT} <: ProximableFunction{DT}
    projection::Function # Π: x -> y ∈ C
end
proxy(λ::T, g::IndicatorFunction{ScalarField2D{T}}, x::ScalarField2D{T}) where T = g.projection(x)
proxy(λ::T, g::IndicatorFunction{VectorField2D{T}}, x::VectorField2D{T}) where T = g.projection(x)

function projection_L2Inf(v::VectorField2D{T}; eps::T=T(1e-20)) where T
    ptnorm_v = ptnorm(v; p=2)
    norm(ptnorm_v; p=Inf) <= T(1) ? (return (T(0), v)) : (return (Inf, v*( (ptnorm_v >= T(1)) /(ptnorm_v+eps)+(ptnorm_v < T(1)) )))
end

project(x::DT, C::IndicatorFunction{DT}) where DT = C.projection(x)
project!(y::DT, x::DT, C::IndicatorFunction{DT}) where DT = update!(y, project(x, C))


# Vector field norm

struct VectorFieldNorm{T,p1,p2} <: ProximableFunction{VectorField2D{T}} end

function proxy(λ::T, g::VectorFieldNorm{T,2,1}, v::VectorField2D{T}; outval::Bool=true) where T
    ptnorm_v = ptnorm(v; p=2)
    y = (T(1)-λ/ptnorm_v)*(ptnorm_v >= λ)*v
    outval ? (gval = norm(to_array(ptnorm_v); p=1)) : (gval = nothing)
    return gval, y
end

function proxy(ε::T, g::VectorFieldNorm{T,2,Inf}, v::VectorField2D{T}; outval::Bool=true) where T
    ptnorm_v = ptnorm(v; p=2)
    I_ = findall(to_array(ptnorm_v) .== norm(to_array(ptnorm_v); p=Inf))
    y = deepcopy(v)
    y[I_] = (T(1)-ε/ptnorm_v[I_])*(ptnorm_v[I_] >= ε)*v[I_]
    outval ? (gval = norm(to_array(ptnorm_v); p=Inf)) : (gval = nothing)
    return gval, y
end

VectorFieldNorm(T::DataType, p1::Int64, p2::Int64) = VectorFieldNorm{T,p1,p2}(T(1))

# conjugate(g::VectorFieldNorm{T,2,1}) where T = IndicatorFunction{T}(projection_L2Inf)