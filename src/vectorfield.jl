# Vector/matrix field data type utils


using LinearAlgebra, Flux, CUDA
export VectorField2D, ptdot, ptnorm
import Base.size, Base.IndexStyle, Base.getindex, Base.setindex!
import Base.+, Base.-, Base.*, Base./
import LinearAlgebra: dot, norm, transpose
import Flux.gpu


## Abstract types

abstract type AbstractMatrixField2D <: AbstractArray{T,4} end

size(A::AbstractMatrixField2D) = (size(A.xx)..., 2, 2)
IndexStyle(::Type{<:AbstractMatrixField2D}) = IndexLinear()
function getindex(A::AbstractMatrixField2D, i::Int64)
    l = length(A.xx)
    (i <= l) && (return A.xx[i])
    (i >  l) && (return A.yx[i-l])
end
function setindex!(v::VectorField2D, val::T, i::Int64)
    (i <= length(v.x)) && (v.x[i]             = val)
    (i >  length(v.x)) && (v.y[i-length(v.x)] = val)
end

+(A1::AbstractMatrixField2D, A2::AbstractMatrixField2D) = VectorField2D(v1.x+v2.x, v1.y+v2.y)
-(v1::VectorField2D, v2::VectorField2D) = VectorField2D(v1.x-v2.x, v1.y-v2.y)
-(v::VectorField2D) = VectorField2D(-v.x, -v.y)
*(c::T, v::VectorField2D) = VectorField2D(c*v.x, c*v.y)
*(v::VectorField2D, c::T) = c*v
*(c::AbstractArray{T,2}, v::VectorField2D) = VectorField2D(c.*v.x, c.*v.y)
*(v::VectorField2D, c::AbstractArray{T,2}) = c*v

abstract type AbstractVectorField2D <: AbstractArray{T,3} end

abstract type AbstractDiagMatrixField2D <: AbstractMatrixField2D end

## Data type for vector/matrix fields

struct VectorField2D <: AbstractVectorField2D
    x::AbstractArray{T,2}
    y::AbstractArray{T,2}
end

struct TransposedVectorField2D
    v::VectorField2D
end

struct DiagMatrixField2D <: AbstractDiagMatrixField2D
    v::VectorField2D
end

struct IdMatrixField2D <: AbstractDiagMatrixField2D end

struct Rank1MatrixField2D <: AbstractMatrixField2D
    u::VectorField2D
    v_T::TransposedVectorField2D
end


## Base functions

size(v::VectorField2D) = (size(v.x)..., 2)
IndexStyle(::Type{<:VectorField2D}) = IndexLinear()
function getindex(v::VectorField2D, i::Int64)
    (i <= length(v.x)) && (return v.x[i])
    (i >  length(v.x)) && (return v.y[i-length(v.x)])
end
function setindex!(v::VectorField2D, val::T, i::Int64)
    (i <= length(v.x)) && (v.x[i]             = val)
    (i >  length(v.x)) && (v.y[i-length(v.x)] = val)
end


## Operations

+(v1::VectorField2D, v2::VectorField2D) = VectorField2D(v1.x+v2.x, v1.y+v2.y)
-(v1::VectorField2D, v2::VectorField2D) = VectorField2D(v1.x-v2.x, v1.y-v2.y)
-(v::VectorField2D) = VectorField2D(-v.x, -v.y)
*(c::T, v::VectorField2D) = VectorField2D(c*v.x, c*v.y)
*(v::VectorField2D, c::T) = c*v
*(c::AbstractArray{T,2}, v::VectorField2D) = VectorField2D(c.*v.x, c.*v.y)
*(v::VectorField2D, c::AbstractArray{T,2}) = c*v
/(v::VectorField2D, c::T) = VectorField2D(v.x/c, v.y/c)
/(v::VectorField2D, c::AbstractArray{T,2}) = VectorField2D(v.x./c, v.y./c)

+(v1_T::TransposedVectorField2D, v2_T::TransposedVectorField2D) = transpose(v1_T.v+v2_T.v)
-(v1_T::TransposedVectorField2D, v2_T::TransposedVectorField2D) = transpose(v1_T.v-v2_T.v)
-(v1_T::TransposedVectorField2D) = transpose(-v1_T.v)
*(c::T, v_T::TransposedVectorField2D) = transpose(c*v_T.v)
*(v_T::TransposedVectorField2D, c::T) = transpose(c*v_T.v)
/(v_T::TransposedVectorField2D, c::T) = transpose(v_T.v/c)
/(v_T::TransposedVectorField2D, c::AbstractArray{T,2}) = transpose((v_T.v/c)

*(D::AbstractMatrixField2D, v::VectorField2D) = VectorField2D(D.xx.*v.x+D.xy.*v.y, D.yx.*v.x+D.yy.*v.y)
*(D::IdMatrixField2D, v::VectorField2D) = v
*(u_T::TransposedVectorField2D, v::VectorField2D) = u_T.v.x.*v.x+u_T.v.y.*v.y
*(u::VectorField2D, v_T::TransposedVectorField2D) = Rank1MatrixField2D(u, v_T)
*(D::Rank1MatrixField2D, w::VectorField2D) = D.u*(D.v_T*w)


## Linear algebra utils

ptdot(v1::VectorField2D, v2::VectorField2D) = v1.x.*v2.x+v1.y.*v2.y
ptnorm1(v::VectorField2D) = abs.(v.x)+abs.(v.y)
ptnorm2(v::VectorField2D) = sqrt.(v.x.^2+v.y.^2)
ptnormInf(v::VectorField2D) = max.(v.x, v.y)
function ptnorm(v::VectorField2D; p=2)
    (p == 2)   && (return ptnorm2(v))
    (p == 1)   && (return ptnorm1(v))
    (p == Inf) && (return ptnormInf(v))
end
norm1(x::CuArray) = sum(abs.(x))
norm2(x::CuArray) = norm(x,2)
normInf(x::CuArray) = maximum(abs.(x))
function norm(x::CuArray; p=2)
    (p == 2)   && (return norm2(x))
    (p == 1)   && (return norm1(x))
    (p == Inf) && (return normInf(x))
end
norm(x::Array{Float32}; p=2) = norm(x, p)

norm(v::VectorField2D; p1=2, p2=2) = norm(ptnorm(v; p=p2); p=p1)
dot(v1::VectorField2D, v2::VectorField2D) = dot(v1.x, v2.x)+dot(v1.y, v2.y)

transpose(v::VectorField2D) = TransposedVectorField2D(v)


## Flux utils

gpu(v::VectorField2D) = VectorField2D(gpu(v.x), gpu(v.y))