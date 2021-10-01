#: Gradient operator via convolution

export gradient_2D, gradient_batch_2D, structural_weight, gradient_mean


# Gradient (2D)

struct Gradient_2D{T}<:AbstractLinearOperator{AbstractArray{T,2},AbstractArray{T,3}}
    stencil::AbstractArray{T,4}
end

AbstractLinearOperators.domain_size(D::Gradient_2D) = "nx×ny"
AbstractLinearOperators.range_size(D::Gradient_2D) = "(nx-1)×(ny-1)×2"
function AbstractLinearOperators.matvecprod(D::Gradient_2D{T}, u::AbstractArray{T,2}) where T
    nx,ny = size(u)
    return reshape(conv(reshape(u, nx,ny,1,1), D.stencil), nx-1,ny-1,2)
end
function AbstractLinearOperators.matvecprod_adj(D::Gradient_2D{T}, v::AbstractArray{T,3}) where T
    nx,ny,_ = size(v)
    cdims = DenseConvDims((nx+1,ny+1,1,1), (2,2,1,2))
    return reshape(∇conv_data(reshape(v, nx,ny,2,1), D.stencil, cdims), nx+1,ny+1)
end

function gradient_2D(; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1))) where {S<:Real}

    # Stencil
    D = cat(reshape([T(0) T(1)/h[1]; T(0) T(-1)/h[1]], 2, 2, 1, 1), reshape([T(0) T(0); T(1)/h[2] T(-1)/h[2]], 2, 2, 1, 1); dims=4)

    return Gradient_2D{T}(D)

end

Flux.gpu(D::Gradient_2D{T}) where T = Gradient_2D{T}(gpu(D.stencil))
Flux.cpu(D::Gradient_2D{T}) where T = Gradient_2D{T}(cpu(D.stencil))


# Gradient (2D-batch)

struct GradientBatch_2D{T}<:AbstractLinearOperator{AbstractArray{T,4},AbstractArray{T,4}}
    stencil::AbstractArray{T,4}
end

AbstractLinearOperators.domain_size(D::GradientBatch_2D) = "nx×ny×nc×nb"
AbstractLinearOperators.range_size(D::GradientBatch_2D) = "(nx-1)×(ny-1)×2nc×nb"
function AbstractLinearOperators.matvecprod(D::GradientBatch_2D{T}, u::AbstractArray{T,4}) where T
    nx,ny,nc,nb = size(u)
    return reshape(conv(reshape(u, nx,ny,1,nc*nb), D.stencil), nx-1,ny-1,2*nc,nb)
end
function AbstractLinearOperators.matvecprod_adj(D::GradientBatch_2D{T}, v::AbstractArray{T,4}) where T
    nx,ny,nc,nb = size(v)
    cdims = DenseConvDims((nx+1,ny+1,1,div(nc,2)*nb), (2,2,1,2))
    return reshape(∇conv_data(reshape(v, nx,ny,2,div(nc,2)*nb), D.stencil, cdims), nx+1,ny+1,div(nc,2),nb)
end

function gradient_batch_2D(; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1))) where {S<:Real}

    # Stencil
    D = cat(reshape([T(0) T(1)/h[1]; T(0) T(-1)/h[1]], 2, 2, 1, 1), reshape([T(0) T(0); T(1)/h[2] T(-1)/h[2]], 2, 2, 1, 1); dims=4)

    return GradientBatch_2D{T}(D)

end

Flux.gpu(D::GradientBatch_2D{T}) where T = GradientBatch_2D{T}(gpu(D.stencil))
Flux.cpu(D::GradientBatch_2D{T}) where T = GradientBatch_2D{T}(cpu(D.stencil))


# Projection on vector field

struct ProjVectorField_2D{T}<:AbstractLinearOperator{AbstractArray{T,3},AbstractArray{T,3}}
    ∇ŵ::AbstractArray{T,3}
end

AbstractLinearOperators.domain_size(P::ProjVectorField_2D) = size(P.∇ŵ)
AbstractLinearOperators.range_size(P::ProjVectorField_2D) = size(P.∇ŵ)
AbstractLinearOperators.matvecprod(P::ProjVectorField_2D{T}, u::AbstractArray{T,3}) where T = u-ptdot_2D(u,P.∇ŵ).*P.∇ŵ
AbstractLinearOperators.matvecprod_adj(P::ProjVectorField_2D{T}, u::AbstractArray{T,3}) where T = P*u

structural_weight(P::ProjVectorField_2D) = P.∇ŵ
function structural_weight(w::AbstractArray{T,2}, η::T) where T
    ∇ = gradient_2D(; T=T); w isa CuArray && (∇ = ∇ |> gpu)
    ∇w = ∇*w
    return ProjVectorField_2D{T}(∇w./ptnorm2_2D(∇w; η=η))
end

function gradient_mean(w::AbstractArray{T,2}) where T
    ∇ = gradient_2D(; T=T); w isa CuArray && (∇ = ∇ |> gpu)
    return sum(ptnorm2_2D(∇*w))/prod(size(w)[1:2])
end

Flux.gpu(P::ProjVectorField_2D{T}) where T = ProjVectorField_2D{T}(Flux.gpu(structural_weight(P)))
Flux.cpu(P::ProjVectorField_2D{T}) where T = ProjVectorField_2D{T}(Flux.cpu(structural_weight(P)))