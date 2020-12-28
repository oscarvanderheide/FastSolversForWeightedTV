#: Convex set utilities


export ConvexSet, NoConstraints, no_constraints, PositiveValues, positive_values, L_ball, ell_ball, project, project!


# Convex sets abstract type

"""
Expected behavior for convex sets: project(x, C), project!(x_, x, C)
"""
abstract type ConvexSet{DT} end


## No constraint set

struct NoConstraints{DT} <: ConvexSet{DT} end

no_constraints(DT::DataType) = NoConstraints{DT}()

project(x::DT, C::NoConstraints{DT}) where DT = x
project!(x_::DT, x::DT, C::NoConstraints{DT}) where DT = update!(x_, x)


## L_{p1,p2} unitary ball: C={x:||x||_{p1,p2}<=1}

### 2,Inf

struct L_ball{T,p1,p2} <: ConvexSet{VectorField2D{T}} end

ell_ball(T::DataType, p1::Number, p2::Number) = L_ball{T,p1,p2}()

function project(v::VectorField2D{T}, C::L_ball{T,2,Inf}; eps::T=T(1e-20)) where T
    ptnorm_v = ptnorm(v; p=2)
    norm(ptnorm_v; p=Inf) <= T(1) ? (return v) : (return v*( (ptnorm_v >= T(1)) /(ptnorm_v+eps)+(ptnorm_v < T(1)) ))
end

project!(v_, v::VectorField2D{T}, C::L_ball{T,2,Inf}; eps::T=T(1e-20)) where T = update!(v_, projection(v, C; eps=eps))

### TO DO: (1,2), (2,2), (Inf,2), ...


## Positivity

struct PositiveValues{T} <: ConvexSet{ScalarField2D{T}} end

positive_values(T::DataType) = PositiveValues{T}()

project(x::ScalarField2D{T}, C::PositiveValues{T}) where T = x*(x >= T(0))
project!(x_::ScalarField2D{T}, x::ScalarField2D{T}, C::PositiveValues{T}) where T = update!(x_, projection(x, C))