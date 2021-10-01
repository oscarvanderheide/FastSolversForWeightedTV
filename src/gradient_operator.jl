#: Gradient operator via convolution

export gradient_2D


# Gradient (linear operator)

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