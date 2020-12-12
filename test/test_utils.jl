using VectorFields, DifferentialOperatorsForTV, FastSolversForWeightedTV, Flux, Test

flag_gpu = true
# flag_gpu = false

# Random input
n = (1001, 2001)
h = (abs(randn(Float32)), abs(randn(Float32)))
u = to_field(randn(Float32, n)); flag_gpu && (u = u |> gpu)

# In-place update
u_ = undef_as(u)
update!(u_, u)
@test u_ ≈ u rtol=1f-3