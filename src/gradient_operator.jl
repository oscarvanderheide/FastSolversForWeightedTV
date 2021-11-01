#: Gradient operator via convolution

export gradient_2D, gradient_3D, gradient_batch_2D


# Abstract gradient types

abstract type AbstractGradient_2D{T}<:AbstractLinearOperator{AbstractArray{T,2},AbstractArray{T,3}} end
abstract type AbstractGradientBatch_2D{T}<:AbstractLinearOperator{AbstractArray{T,4},AbstractArray{T,4}} end
abstract type AbstractGradient_3D{T}<:AbstractLinearOperator{AbstractArray{T,3},AbstractArray{T,4}} end


# Gradient (2D)

struct Gradient_2D{T}<:AbstractGradient_2D{T}
    n::NTuple{2,Int64}
    stencil::AbstractArray{T,4}
end

AbstractLinearOperators.domain_size(D::Gradient_2D) = D.n
AbstractLinearOperators.range_size(D::Gradient_2D) = (D.n[1]-1, D.n[2]-1, 2)
AbstractLinearOperators.matvecprod(D::Gradient_2D{T}, u::AbstractArray{T,2}) where {T<:Real} = reshape(conv(reshape(u, D.n[1],D.n[2],1,1), D.stencil), D.n[1]-1,D.n[2]-1,2)
function AbstractLinearOperators.matvecprod_adj(D::Gradient_2D{T}, v::AbstractArray{T,3}) where {T<:Real}
    cdims = DenseConvDims((D.n[1],D.n[2],1,1), (2,2,1,2))
    return reshape(∇conv_data(reshape(v, D.n[1]-1,D.n[2]-1,2,1), D.stencil, cdims), D.n[1],D.n[2])
end

function gradient_2D(n::NTuple{2,Int64}; T::DataType=Float32, h::Tuple{S,S}=(1f0,1f0)) where {S<:Real}

    # Stencil
    reT = real(T)
    D = zeros(reT, 2,2,1,2)
    D[:,2,1,1] .= [reT(1)/reT(h[1]); reT(-1)/reT(h[1])]
    D[2,:,1,2] .= [reT(1)/reT(h[2]); reT(-1)/reT(h[2])]
    ∇ = Gradient_2D{reT}(n, D)

    T <: Real ? (return ∇) : (return complex(∇))

end

Flux.gpu(D::Gradient_2D{T}) where {T<:Real} = Gradient_2D{T}(D.n, gpu(D.stencil))
Flux.cpu(D::Gradient_2D{T}) where {T<:Real} = Gradient_2D{T}(D.n, cpu(D.stencil))


# Gradient (2D, complex)

Base.complex(D::Gradient_2D{T}) where {T<:Real} = ComplexGradient_2D{T}(D)

struct ComplexGradient_2D{T}<:AbstractGradient_2D{Complex{T}}
    ∇::Gradient_2D{T}
end

AbstractLinearOperators.domain_size(D::ComplexGradient_2D) = domain_size(D.∇)
AbstractLinearOperators.range_size(D::ComplexGradient_2D) = range_size(D.∇)
AbstractLinearOperators.matvecprod(D::ComplexGradient_2D{T}, u::AbstractArray{Complex{T},2}) where {T<:Real} = D.∇*real.(u)+1im*D.∇*imag.(u)
AbstractLinearOperators.matvecprod_adj(D::ComplexGradient_2D{T}, v::AbstractArray{Complex{T},3}) where {T<:Real} = adjoint(D.∇)*real.(v)+1im*adjoint(D.∇)*imag.(v)

Flux.gpu(D::ComplexGradient_2D{T}) where {T<:Real} = ComplexGradient_2D{T}(gpu(D.∇))
Flux.cpu(D::ComplexGradient_2D{T}) where {T<:Real} = ComplexGradient_2D{T}(cpu(D.∇))


# Gradient (3D)

struct Gradient_3D{T}<:AbstractGradient_3D{T}
    n::NTuple{3,Int64}
    stencil::AbstractArray{T,5}
end

AbstractLinearOperators.domain_size(D::Gradient_3D) = D.n
AbstractLinearOperators.range_size(D::Gradient_3D) = (D.n[1]-1, D.n[2]-1, D.n[3]-1, 3)
AbstractLinearOperators.matvecprod(D::Gradient_3D{T}, u::AbstractArray{T,3}) where {T<:Real} = reshape(conv(reshape(u, D.n[1],D.n[2],D.n[3],1,1), D.stencil), D.n[1]-1,D.n[2]-1,D.n[3]-1,3)
function AbstractLinearOperators.matvecprod_adj(D::Gradient_3D{T}, v::AbstractArray{T,4}) where {T<:Real}
    cdims = DenseConvDims((D.n[1],D.n[2],D.n[3],1,1), (2,2,2,1,3))
    return reshape(∇conv_data(reshape(v, D.n[1]-1,D.n[2]-1,D.n[3]-1,3,1), D.stencil, cdims), D.n[1],D.n[2],D.n[3])
end

function gradient_3D(n::NTuple{3,Int64}; T::DataType=Float32, h::NTuple{3,S}=(1f0,1f0,1f0)) where {S<:Real}

    # Stencil
    reT = real(T)
    D = zeros(reT, 2,2,2,1,3)
    D[:,2,2,1,1] .= [reT(1)/reT(h[1]); reT(-1)/reT(h[1])]
    D[2,:,2,1,2] .= [reT(1)/reT(h[2]); reT(-1)/reT(h[2])]
    D[2,2,:,1,3] .= [reT(1)/reT(h[3]); reT(-1)/reT(h[3])]
    ∇ = Gradient_3D{reT}(n, D)

    T <: Real ? (return ∇) : (return complex(∇))

end

Flux.gpu(D::Gradient_3D{T}) where {T<:Real} = Gradient_3D{T}(D.n, gpu(D.stencil))
Flux.cpu(D::Gradient_3D{T}) where {T<:Real} = Gradient_3D{T}(D.n, cpu(D.stencil))


# Gradient (3D, complex)

Base.complex(D::Gradient_3D{T}) where {T<:Real} = ComplexGradient_3D{T}(D)

struct ComplexGradient_3D{T}<:AbstractGradient_3D{Complex{T}}
    ∇::Gradient_3D{T}
end

AbstractLinearOperators.domain_size(D::ComplexGradient_3D) = domain_size(D.∇)
AbstractLinearOperators.range_size(D::ComplexGradient_3D) = range_size(D.∇)
AbstractLinearOperators.matvecprod(D::ComplexGradient_3D{T}, u::AbstractArray{Complex{T},3}) where {T<:Real} = D.∇*real.(u)+1im*D.∇*imag.(u)
AbstractLinearOperators.matvecprod_adj(D::ComplexGradient_3D{T}, v::AbstractArray{Complex{T},4}) where {T<:Real} = adjoint(D.∇)*real.(v)+1im*adjoint(D.∇)*imag.(v)

Flux.gpu(D::ComplexGradient_3D{T}) where {T<:Real} = ComplexGradient_3D{T}(gpu(D.∇))
Flux.cpu(D::ComplexGradient_3D{T}) where {T<:Real} = ComplexGradient_3D{T}(cpu(D.∇))


# Gradient (2D-batch)

struct GradientBatch_2D{T}<:AbstractGradientBatch_2D{T}
    n::NTuple{4, Int64}
    stencil::AbstractArray{T,4}
end

AbstractLinearOperators.domain_size(D::GradientBatch_2D) = D.n
AbstractLinearOperators.range_size(D::GradientBatch_2D) = (D.n[1]-1, D.n[2]-1, 2*D.n[3], D.n[4])
AbstractLinearOperators.matvecprod(D::GradientBatch_2D{T}, u::AbstractArray{T,4}) where {T<:Real} = reshape(conv(reshape(u, D.n[1],D.n[2],1,D.n[3]*D.n[4]), D.stencil), D.n[1]-1,D.n[2]-1,2*D.n[3],D.n[4])
function AbstractLinearOperators.matvecprod_adj(D::GradientBatch_2D{T}, v::AbstractArray{T,4}) where {T<:Real}
    cdims = DenseConvDims((D.n[1],D.n[2],1,div(D.n[3],2)*D.n[4]), (2,2,1,2))
    return reshape(∇conv_data(reshape(v, D.n[1]-1,D.n[2]-1,2,div(D.n[3],2)*D.n[4]), D.stencil, cdims), D.n[1],D.n[2],div(D.n[3],2),D.n[4])
end

function gradient_batch_2D(n::NTuple{4,Int64}; T::DataType=Float32, h::NTuple{2,S}=(1f0,1f0)) where {S<:Real}

    # Stencil
    reT = real(T)
    D = cat(reshape([reT(0) reT(1)/reT(h[1]); reT(0) reT(-1)/reT(h[1])], 2, 2, 1, 1), reshape([reT(0) reT(0); reT(1)/reT(h[2]) reT(-1)/reT(h[2])], 2, 2, 1, 1); dims=4)
    ∇ = GradientBatch_2D{reT}(n, D)

    T <: Real ? (return ∇) : (return complex(∇)) 

end

Flux.gpu(D::GradientBatch_2D{T}) where {T<:Real} = GradientBatch_2D{T}(D.n, gpu(D.stencil))
Flux.cpu(D::GradientBatch_2D{T}) where {T<:Real} = GradientBatch_2D{T}(D.n, cpu(D.stencil))


# Gradient (2D-batch, complex)

Base.complex(D::GradientBatch_2D{T}) where {T<:Real} = ComplexGradientBatch_2D{T}(D)

struct ComplexGradientBatch_2D{T}<:AbstractGradientBatch_2D{Complex{T}}
    ∇::GradientBatch_2D{T}
end

AbstractLinearOperators.domain_size(D::ComplexGradientBatch_2D) = domain_size(D.∇)
AbstractLinearOperators.range_size(D::ComplexGradientBatch_2D) = range_size(D.∇)
AbstractLinearOperators.matvecprod(D::ComplexGradientBatch_2D{T}, u::AbstractArray{Complex{T},4}) where {T<:Real} = D.∇*real.(u)+1im*D.∇*imag.(u)
AbstractLinearOperators.matvecprod_adj(D::ComplexGradientBatch_2D{T}, v::AbstractArray{Complex{T},4}) where {T<:Real} = adjoint(D.∇)*real.(v)+1im*adjoint(D.∇)*imag.(v)

Flux.gpu(D::ComplexGradientBatch_2D{T}) where {T<:Real} = ComplexGradientBatch_2D{T}(gpu(D.∇))
Flux.cpu(D::ComplexGradientBatch_2D{T}) where {T<:Real} = ComplexGradientBatch_2D{T}(cpu(D.∇))