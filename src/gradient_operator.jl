#: Gradient operator via convolution

export gradient_operator, gradient_operator_batch


# Generic convolutional operator

struct RealConvolutionalOperator{T,N}<:AbstractLinearOperator{T,N,N}
    cdims::DenseConvDims
    stencil::AbstractArray{<:Real,N}
end

AbstractLinearOperators.domain_size(A::RealConvolutionalOperator) = NNlib.input_size(A.cdims)
AbstractLinearOperators.range_size(A::RealConvolutionalOperator) = NNlib.output_size(A.cdims)
AbstractLinearOperators.matvecprod(A::RealConvolutionalOperator{CT,N}, u::AbstractArray{T,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = conv(u, A.stencil)
AbstractLinearOperators.matvecprod(A::RealConvolutionalOperator{T,N}, u::AbstractArray{T,N}) where {T<:Complex,N} = matvecprod(A, real(u))+1im*matvecprod(A, imag(u))
AbstractLinearOperators.matvecprod_adj(A::RealConvolutionalOperator{CT,N}, v::AbstractArray{T,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ∇conv_data(v, A.stencil, A.cdims)
AbstractLinearOperators.matvecprod_adj(A::RealConvolutionalOperator{T,N}, v::AbstractArray{T,N}) where {T<:Complex,N} = matvecprod_adj(A, real(v))+1im*matvecprod_adj(A, imag(v))

Flux.gpu(A::RealConvolutionalOperator{T,N}) where {T,N} = RealConvolutionalOperator{T,N}(A.cdims, gpu(A.stencil))
Flux.cpu(A::RealConvolutionalOperator{T,N}) where {T,N} = RealConvolutionalOperator{T,N}(A.cdims, cpu(A.stencil))


# Gradient operator

function gradient_operator(n::NTuple{dim,Int64}, h::NTuple{dim,S}; T::DataType=Float32) where {dim,S<:Real}

    # Gradient operator (w/out shaping)
    stencil = gradient_stencil(real(T).(h))
    cdims = DenseConvDims((n...,1,1), size(stencil))
    ∇_ = RealConvolutionalOperator{T,dim+2}(cdims, stencil)

    # Reshaping operator
    Rin  = reshape_operator(T, n, (n...,1,1))
    Rout = reshape_operator(T, ((n.-1)...,dim,1), ((n.-1)...,dim))

    return Rout*∇_*Rin

end

function gradient_operator_batch(n::NTuple{dim,Int64}, nc::Int64, nb::Int64, h::NTuple{dim,S}; T::DataType=Float32) where {dim,S<:Real}

    # Gradient operator (w/out shaping)
    stencil = gradient_stencil(real(T).(h))
    cdims = DenseConvDims((n...,1,nc*nb), size(stencil))
    ∇_ = RealConvolutionalOperator{T,dim+2}(cdims, stencil)

    # Reshaping operator
    Rin  = reshape_operator(T, (n...,nc,nb), (n...,1,nc*nb))
    Rout = reshape_operator(T, ((n.-1)...,dim,nc*nb), ((n.-1)...,dim*nc,nb))

    return Rout*∇_*Rin

end


# Gradient stencil utils

function gradient_stencil(h::NTuple{dim,T}) where {dim,T}
    k = tuple(2*ones(Int64, dim)...)
    stencil = zeros(T,k...,1,dim)
    for i = 1:dim
        idx = Vector{UnitRange{Int64}}(undef,dim); fill!(idx, 2:2)
        idx[i] = 1:2
        view(stencil, tuple(idx...)...,1,i)[1:2] .= [T(1)/h[i]; T(-1)/h[i]]
    end
    return stencil
end