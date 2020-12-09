#: Least-square misfit loss for AbstractField2D types

export LeastSquaresMisfitField2D


function LeastSquaresMisfitField2D(A::AbstractFieldLinearOperator{DT,RT}, y::RT) where {DT<:AbstractField2D,RT<:AbstractField2D}
    return LeastSquaresMisfit{DT}(A, y)
end

only_eval(f::LeastSquaresMisfit{DT}, x::DT) where DT = eltype(x)(0.5)*norm(f.A*x-f.y)^2
function grad(f::LeastSquaresMisfit{DT}, x::DT) where DT
    T = eltype(x)
    r = f.A*x-f.y
    fx = T(0.5)*norm(r)^2
    gx = adjoint(f.A)*r
    return fx, gx
end
function grad!(g::DT, f::LeastSquaresMisfit{DT}, x::DT) where DT
    T = eltype(x)
    r = f.A*x-f.y
    fx = T(0.5)*norm(f.A*x-f.y)^2
    update!(g, adjoint(f.A)*r)
    return fx
end