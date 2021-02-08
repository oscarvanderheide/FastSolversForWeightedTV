#: Differential operators

export HorzDerivative2D, VertDerivative2D, Gradient2D, horz_derivative_2D, vert_derivative_2D, gradient_2D


# Differential operators

## Horizontal derivative

mutable struct HorzDerivative2D{T}<:AbstractLinearOperator{AbstractArray{T,2},AbstractArray{T,2}}
    spacing::T
    padding::Padding2D{T}
    cdims::DenseConvDims
    stencil::AbstractArray{T,4}
end

AbstractLinearOperators.domain_size(D::HorzDerivative2D) = D.cdims.I
AbstractLinearOperators.range_size(D::HorzDerivative2D) = D.cdims.I
AbstractLinearOperators.matvecprod(D::HorzDerivative2D{T}, u::AbstractArray{T,2}) where T = conv(reshape(pad(u, D.padding), size(u,1)+1,size(u,2), 1, 1), D.stencil, D.cdims)[:,:,1,1]
AbstractLinearOperators.matvecprod_adj(D::HorzDerivative2D{T}, v::AbstractArray{T,2}) where T = restrict(∇conv_data(reshape(v, size(v)..., 1, 1), D.stencil, D.cdims)[:,:,1,1], D.padding)

function horz_derivative_2D(n::Tuple{Int64,Int64}, dx::T; pad_type::String="periodic", gpu::Bool=false) where T

    # Convolution
    Dx = reshape([T(1)/dx T(-1)/dx], 2, 1, 1, 1)
    cdims = DenseConvDims((n[1]+1,n[2],1,1), (2,1,1,1); stride=(1,1), padding=(0,0))
    gpu && (Dx = Dx |> Flux.gpu)

    # Padding
    P = padding(pad_type, T, 0, 1, 0, 0)

    return HorzDerivative2D{T}(dx, P, cdims, Dx)

end

Flux.gpu(Dx::HorzDerivative2D{T}) where T = HorzDerivative2D{T}(Dx.spacing, Dx.padding, Dx.cdims, gpu(Dx.stencil))
Flux.cpu(Dx::HorzDerivative2D{T}) where T = HorzDerivative2D{T}(Dx.spacing, Dx.padding, Dx.cdims, cpu(Dx.stencil))

## Vertical derivative

mutable struct VertDerivative2D{T}<:AbstractLinearOperator{AbstractArray{T,2},AbstractArray{T,2}}
    spacing::T
    padding::Padding2D{T}
    cdims::DenseConvDims
    stencil::AbstractArray{T,4}
end

AbstractLinearOperators.domain_size(D::VertDerivative2D) = D.cdims.I
AbstractLinearOperators.range_size(D::VertDerivative2D) = D.cdims.I
AbstractLinearOperators.matvecprod(D::VertDerivative2D{T}, u::AbstractArray{T,2}) where T = conv(reshape(pad(u, D.padding), size(u,1),size(u,2)+1, 1, 1), D.stencil, D.cdims)[:,:,1,1]
AbstractLinearOperators.matvecprod_adj(D::VertDerivative2D{T}, v::AbstractArray{T,2}) where T = restrict(∇conv_data(reshape(v, size(v)..., 1, 1), D.stencil, D.cdims)[:,:,1,1], D.padding)

function vert_derivative_2D(n::Tuple{Int64,Int64}, dy::T; pad_type::String="periodic", gpu::Bool=false) where T

    # Convolution
    Dy = reshape([T(1)/dy T(-1)/dy], 1, 2, 1, 1)
    cdims = DenseConvDims((n[1],n[2]+1,1,1), (1,2,1,1); stride=(1,1), padding=(0,0))
    gpu && (Dy = Dy |> Flux.gpu)

    # Padding
    P = padding(pad_type, T, 0, 0, 0, 1)

    return VertDerivative2D{T}(dy, P, cdims, Dy)

end

Flux.gpu(Dy::VertDerivative2D{T}) where T = VertDerivative2D{T}(Dy.spacing, Dy.padding, Dy.cdims, gpu(Dy.stencil))
Flux.cpu(Dy::VertDerivative2D{T}) where T = VertDerivative2D{T}(Dy.spacing, Dy.padding, Dy.cdims, cpu(Dy.stencil))

## Gradient

mutable struct Gradient2D{T}<:AbstractLinearOperator{AbstractArray{T,2},AbstractArray{T,3}}
    spacing::Tuple{T,T}
    padding::Padding2D{T}
    cdims::DenseConvDims
    stencil::AbstractArray{T,4}
end

AbstractLinearOperators.domain_size(D::Gradient2D) = D.cdims.I
AbstractLinearOperators.range_size(D::Gradient2D) = (D.cdims.I..., 2)
AbstractLinearOperators.matvecprod(D::Gradient2D{T}, u::AbstractArray{T,2}) where T = conv(reshape(pad(u, D.padding), size(u,1)+1,size(u,2)+1, 1, 1), D.stencil, D.cdims)[:,:,:,1]
AbstractLinearOperators.matvecprod_adj(D::Gradient2D{T}, v::AbstractArray{T,3}) where T = restrict(∇conv_data(reshape(v, size(v)..., 1), D.stencil, D.cdims)[:,:,1,1], D.padding)

function gradient_2D(n::Tuple{Int64,Int64}, h::Tuple{T,T}; pad_type::String="periodic", gpu::Bool=false) where T

    # Convolution
    D = cat(reshape([T(0) T(1)/h[1]; T(0) T(-1)/h[1]], 2, 2, 1, 1), reshape([T(0) T(0); T(1)/h[2] T(-1)/h[2]], 2, 2, 1, 1); dims=4)
    gpu && (D = D |> Flux.gpu)
    cdims = DenseConvDims((n[1]+1,n[2]+1,1,1), (2,2,1,2); stride=(1,1), padding=(0,0))

    # Padding
    P = padding(pad_type, T, 0, 1, 0, 1)

    return Gradient2D{T}(h, P, cdims, D)

end

Flux.gpu(D::Gradient2D{T}) where T = Gradient2D{T}(D.spacing, D.padding, D.cdims, gpu(D.stencil))
Flux.cpu(D::Gradient2D{T}) where T = Gradient2D{T}(D.spacing, D.padding, D.cdims, cpu(D.stencil))