using Oceananigans
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_tracer_tendency
using Oceananigans.Grids: get_active_cells_map
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using KernelAbstractions: @kernel, @index
using DifferentiationInterface
using DifferentiationInterface: Cache
using SparseConnectivityTracer
using ForwardDiff: ForwardDiff
using SparseMatrixColorings
using GLMakie

@info "Grid setup"

resolution = 4 // 1        # degrees
Nx = 360 ÷ resolution      # number of longitude points
Ny = 180 ÷ resolution + 1  # number of latitude points (avoiding poles)
Nz = 10
H = 5000                   # domain depth [m]
z = (-H, 0)                # vertical extent

underlying_tripolar_grid = TripolarGrid(
    size = (Nx, Ny, Nz),
    fold_topology = RightFaceFolded,
    z = z,
)

σφ, σλ = 5, 5     # mountain extent in latitude and longitude (degrees)
λ₀, φ₀ = 70, 55     # first pole location
h = H + 1000        # mountain height above the bottom (m)

gaussian(λ, φ) = exp(-((λ - λ₀)^2 / 2σλ^2 + (φ - φ₀)^2 / 2σφ^2))
gaussian_mountains(λ, φ) = (-H
    + h * (gaussian(λ, φ) + gaussian(λ - 180, φ) + gaussian(λ - 360, φ))
    + h/2 * (gaussian(λ - 90, 0) + gaussian(λ - 270, 0)) # extra seamounts
    + h/2 * (90 - φ) / 180 # slanted seafloor towards south pole
)

grid = ImmersedBoundaryGrid(underlying_tripolar_grid, GridFittedBottom(gaussian_mountains))

@info "Model setup"

# Instead of initializing with random velocities, infer them from a random initial streamfunction
# to ensure the velocity field is divergence-free at initialization.
ψ = Field{Face, Face, Center}(grid)
set!(ψ, rand(size(ψ)...))
velocities = PrescribedVelocityFields(; u = ∂y(ψ), v = -∂x(ψ))

@warn "Vertical closure must be explicit otherwise it won't be in the tendency!"
closure = (
    HorizontalScalarDiffusivity(κ = 300.0),
    VerticalScalarDiffusivity(ExplicitTimeDiscretization(); κ = 1.0e-5),
)

f0 = CenterField(grid)

@warn "Adding newton_div method to allow sparsity tracer to pass through WENO"

ADTypes = Union{SparseConnectivityTracer.AbstractTracer, SparseConnectivityTracer.Dual, ForwardDiff.Dual}
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b) where {FT <: ADTypes} = a / b

model = HydrostaticFreeSurfaceModel(
    grid;
    velocities = velocities,
    tracer_advection = Centered(order = 2),
    tracers = (c = f0,),
    closure = closure,
)

@info "Functions to get vector of tendencies"

Nx′, Ny′, Nz′ = size(f0)
N = Nx′ * Ny′ * Nz′
fNaN = CenterField(grid)
mask_immersed_field!(fNaN, NaN)
idx = findall(!isnan, interior(fNaN))
Nidx = length(idx)
@show N, Nidx
c0 = ones(Nidx)
kernel_parameters = KernelParameters(1:Nx′, 1:Ny′, 1:Nz′)
active_cells_map = get_active_cells_map(grid, Val(:interior))


@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end


function mytendency!(Gcvec, cvec, c_field, Gc_field)
    # Fill the field's interior directly from the vector
    interior(c_field) .= 0
    for (n, ijk) in enumerate(idx)
        interior(c_field)[ijk] = cvec[n]
    end
    fill_halo_regions!(c_field)

    # bits and pieces from model
    c_advection = model.advection[:c]
    c_forcing = model.forcing[:c]
    c_immersed_bc = immersed_boundary_condition(model.tracers[:c])

    args = tuple(
        Val(1),
        Val(:c),
        c_advection,
        model.closure,
        c_immersed_bc,
        model.buoyancy,
        model.biogeochemistry,
        model.transport_velocities,
        model.free_surface,
        (; c = c_field),
        model.closure_fields,
        model.auxiliary_fields,
        model.clock,
        c_forcing
    )

    launch!(
        CPU(), grid, kernel_parameters,
        compute_hydrostatic_free_surface_Gc!,
        Gc_field,
        grid,
        args;
        active_cells_map
    )
    # Fill output vector with interior wet values
    Gcvec .= view(interior(Gc_field), idx)
    return Gcvec
end

@info "Autodiff setup"

sparse_forward_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = TracerSparsityDetector(; gradient_pattern_type = Set{UInt}),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

@info "Compute the Jacobian"
using BenchmarkTools

# Preallocate Fields for Cache contexts
c_buf  = CenterField(grid)
Gc_buf = CenterField(grid)
dc0 = similar(c0)

# Warm up
@info "Warm-up..."
mytendency!(dc0, c0, c_buf, Gc_buf)

# Prepare Jacobian — single function, no strict=Val(false) needed
@info "Preparing Jacobian..."
@time "Prepare Jacobian" jac_prep = prepare_jacobian(
    mytendency!, dc0, sparse_forward_backend, c0,
    Cache(c_buf), Cache(Gc_buf),
)
jac_buffer = similar(sparsity_pattern(jac_prep), eltype(c0))

@info "Computing Jacobian..."
@time "Compute Jacobian" J = jacobian!(
    mytendency!, dc0, jac_buffer, jac_prep, sparse_forward_backend, c0,
    Cache(c_buf), Cache(Gc_buf),
)

@info "Benchmarking Jacobian computation..."
@benchmark jacobian!(
    mytendency!, $dc0, $jac_buffer, $jac_prep, $sparse_forward_backend, $c0,
    Cache($c_buf), Cache($Gc_buf),
)

# Linearity check: for a linear tendency, J * c ≈ G(c)
@info "Linearity check: J * c0 ≈ mytendency!(dc0, c0, ...)"
mytendency!(dc0, c0, c_buf, Gc_buf)
Jc = J * c0
max_err = maximum(abs, Jc .- dc0)
@info "  max|J*c - G(c)| = $max_err  (should be ≈ 0 for a linear tendency)"
@assert max_err < 1e-10 "Linearity check failed: max error = $max_err"

z1D = reshape(znodes(grid, Center(), Center(), Center()), 1, 1, Nz)
srf = z1D .≥ z1D[Nz] * ones(Nx, Ny)
using SparseArrays
using LinearAlgebra
L = sparse(Diagonal(srf[idx]))
M = J - L
age = M \ -ones(Nidx)
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
fig, ax, plt = hist(age / year)