using DrWatson
@quickactivate "TotalVariationRegularization"

using Distributed
@everywhere using TotalVariationRegularization, LinearAlgebra, Test, PyPlot, Images, TestImages, FFTW
@everywhere using SetIntersectionProjection

# Test input
h = (T(1), T(1))
b = T.(red.(testimage("mandrill")))
n = size(b)

# Options
options = PARSDMM_options()
options.FL = T
options = default_PARSDMM_options(options, options.FL)
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 1000
options.feas_tol= T(0.001)
options.obj_tol = T(0.001)
options.evol_rel_tol = T(0.00001)

options.rho_ini = [T(1.0f0)]

set_zero_subnormals(true)
BLAS.set_num_threads(2)
FFTW.set_num_threads(2)
options.parallel = false
options.feasibility_only = false
options.zero_ini_guess = true

# TV constraint
struct compgrid
    d::Tuple
    n::Tuple
end
comp_grid = compgrid(h, n)
constraint = Vector{SetIntersectionProjection.set_definitions}()
(TV, dummy1, dummy2, dummy3) = get_TD_operator(comp_grid, "TV", options.FL)
m_min     = T(0.0)
m_max     = T(0.05)*norm(TV*vec(b),1)
set_type  = "l1"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

# Set up constraints, precompute some things and define projector
(P_sub, TD_OP, set_Prop) = setup_constraints(constraint, comp_grid, options.FL)
(TD_OP, AtA, l, y)        = PARSDMM_precompute_distribute(TD_OP, set_Prop, comp_grid, options)
options.rho_ini        = ones(T, length(TD_OP))*T(10.0)
proj_intersection = x-> PARSDMM(x, AtA, TD_OP, set_Prop, P_sub, comp_grid, options)
function proj!(input)
    (x, dummy1, dummy2, dymmy3) = proj_intersection(input)
    return x
end

# Projection
u = reshape(proj!(vec(b)), n)

# Plotting
fig = figure()
subplot(1, 2, 1)
imshow(b)
subplot(1, 2, 2)
imshow(u)
savefig("./plots/test_setinterproj.png")