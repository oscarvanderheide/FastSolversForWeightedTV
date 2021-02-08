using LinearAlgebra, FastSolversForWeightedTV, AbstractLinearOperators, Test

# Random input
n = (1024, 2048)

# AbstractLinearOperators
fw = x->x[1:2:end,1:2:end]
function bw(y)
    x = zeros(Float32,n)
    x[1:2:end,1:2:end] .= y
    return x
end
A = linear_operator(Array{Float32,2}, Array{Float32,2}, n, (512, 1024), fw, bw)

# Misfit
y = randn(Float32,512,1024)
J = leastsquares_misfit(A, y)

# Gradient test
x = randn(Float32,n)
g = similar(x)
fval = grad!(J,x,g)

@test fval ≈ 0.5f0*norm(A*x-y)^2 rtol=1f-3
@test g ≈ adjoint(A)*(A*x-y) rtol=1f-3