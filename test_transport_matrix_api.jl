"""
Standalone smoke test for the newly exported API names:
  - hydrostatic_free_surface_tracer_tendency
  - compute_hydrostatic_tracer_tendencies!

Run with:
  julia --project=@diffocean test_transport_matrix_api.jl

The test uses no auto-diff; it just verifies that the names are accessible from
the public API and that the tendency kernel produces finite, non-NaN values on a
tiny grid.
"""

using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    hydrostatic_free_surface_tracer_tendency,
    compute_hydrostatic_tracer_tendencies!
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Grids: get_active_cells_map
using Oceananigans.Utils: KernelParameters, launch!
using KernelAbstractions: @kernel, @index

@info "Test: hydrostatic_free_surface_tracer_tendency and compute_hydrostatic_tracer_tendencies! are accessible"

# ── minimal grid ──────────────────────────────────────────────────────────────
grid = RectilinearGrid(
    CPU();
    size = (4, 4),
    x = (0, 1e6),
    z = (-100, 0),
    topology = (Periodic, Flat, Bounded),
)

model = HydrostaticFreeSurfaceModel(
    grid;
    tracer_advection = Centered(order = 2),
    tracers = :c,
    closure = HorizontalScalarDiffusivity(κ = 300.0),
)

set!(model, c = 1.0)

# ── Test 1: per-cell kernel function is accessible ────────────────────────────
@info "Test 1: hydrostatic_free_surface_tracer_tendency callable"

Gc_field = CenterField(grid)
c_field  = model.tracers.c

c_advection   = model.advection[:c]
c_forcing     = model.forcing[:c]
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
    model.tracers,
    model.closure_fields,
    model.auxiliary_fields,
    model.clock,
    c_forcing,
)

@kernel function _test_kernel!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

active_cells_map = get_active_cells_map(grid, Val(:interior))
kernel_parameters = KernelParameters(1:4, 1:1, 1:4)

launch!(CPU(), grid, kernel_parameters, _test_kernel!, Gc_field, grid, args; active_cells_map)

Gc_vals = interior(Gc_field)
@assert all(isfinite, Gc_vals) "hydrostatic_free_surface_tracer_tendency returned non-finite values"
@info "  PASSED: all tendency values are finite"

# ── Test 2: compute_hydrostatic_tracer_tendencies! is callable ─────────────────
@info "Test 2: compute_hydrostatic_tracer_tendencies! callable"

# Build a dummy timestepper-like object just to check if the function can be called
# via the model's own timestepper (which has the Gⁿ tendency storage)
simulation = Simulation(model; Δt = 1.0, stop_iteration = 1)
run!(simulation)  # one step to initialise internal state

# Now call compute_hydrostatic_tracer_tendencies! on the model
compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map)
Gn_c = interior(model.timestepper.Gⁿ.c)
@assert all(isfinite, Gn_c) "compute_hydrostatic_tracer_tendencies! returned non-finite values"
@info "  PASSED: all tendency values are finite"

@info "All tests passed."
