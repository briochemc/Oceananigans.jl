"""
Step 3 of debug plan: Systematically simplify to isolate which combination of
(topology, ImmersedBoundaryGrid, MutableVerticalDiscretization) causes the
structural (1,0,1) asymmetry in the Jacobian sparsity pattern.

Six configurations are tested:
  A: Periodic + IBG(PartialCellBottom) + MutableVerticalDiscretization  [baseline, η=0]
  B: Bounded  + IBG(PartialCellBottom) + MutableVerticalDiscretization  [removes periodic x]
  C: Periodic + plain RectilinearGrid  + MutableVerticalDiscretization  [removes IBG]
  D: Periodic + IBG(PartialCellBottom) + fixed z                        [removes zstar]
  E: Periodic + plain RectilinearGrid  + fixed z                        [simplest]
  F: Periodic + IBG(PartialCellBottom) + MutableVerticalDiscretization  [η=sin(2π x/Lx)]
     → same as A but with non-zero varying η; checks whether asymmetric entries
       are structurally phantom (value=0) or genuinely non-zero

Run with:  julia --project=@diffocean step3_simplify.jl
"""

using Oceananigans
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    hydrostatic_free_surface_tracer_tendency, _update_zstar_scaling!, surface_kernel_parameters
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: on_architecture, get_active_cells_map, MutableVerticalDiscretization
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.Fields: immersed_boundary_condition

using KernelAbstractions: @kernel, @index
using LinearAlgebra
using SparseArrays
using DifferentiationInterface
using DifferentiationInterface: Cache
using SparseConnectivityTracer
using ForwardDiff: ForwardDiff
using SparseMatrixColorings
using ADTypes: KnownJacobianSparsityDetector
using DifferentiationInterface: jacobian_sparsity_with_contexts

# ── shared grid parameters ───────────────────────────────────────────────────
const Nx = 4; const Ny = 1; const Nz = 4
const H  = 100.0; const Lx = 1.0e6; const Δt = 5400.0

const z_faces_ref   = collect(range(-H, 0; length = Nz + 1))
const bottom_height = [-H * (0.5 + 0.5 * (i - 1) / (Nx - 1)) for i in 1:Nx, j in 1:Ny]

# ── autodiff overrides for WENO newton_div ───────────────────────────────────
@warn "Adding newton_div overloads to allow sparsity tracer to pass through WENO"
autodifftypes = Union{SparseConnectivityTracer.AbstractTracer,
                      SparseConnectivityTracer.Dual,
                      ForwardDiff.Dual}
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a,     b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b    ) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT,     a::FT, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT,     a,     b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT,     a::FT, b    ) where {FT <: autodifftypes} = a / b

# ── kernel (global for compilation) ─────────────────────────────────────────
@kernel function _compute_GADc!(GADc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds GADc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

# ── linear tracer forcing (grid-agnostic) ────────────────────────────────────
const age_params = (; relaxation_timescale = 3Δt, source_rate = 1.0)

@inline _linear_src(i, j, k, grid, clock, fields, params) =
    ifelse(k ≥ grid.Nz, -fields.ADc[i, j, k] / params.relaxation_timescale, 0.0)

const linear_forcing = (; ADc = Forcing(_linear_src; parameters = age_params, discrete_form = true))

# ── helper: grid builders ────────────────────────────────────────────────────
function mutable_rect(topo)
    RectilinearGrid(CPU(); size = (Nx, Nz), x = (0, Lx),
                    z = MutableVerticalDiscretization(z_faces_ref),
                    topology = topo, halo = (4, 4))
end

function fixed_rect(topo)
    RectilinearGrid(CPU(); size = (Nx, Nz), x = (0, Lx),
                    z = z_faces_ref, topology = topo, halo = (4, 4))
end

function with_ibg(underlying)
    ImmersedBoundaryGrid(underlying, PartialCellBottom(bottom_height);
                         active_cells_map = true, active_z_columns = true)
end

# ── main diagnostic function ─────────────────────────────────────────────────
"""
    check_symmetry(label, grid; use_zstar=false, η_init=0.0, print_values=false) → n_asymmetric::Int

Build a HydrostaticFreeSurfaceModel on `grid`, compute the Jacobian of the ADc
tracer tendency via SparseConnectivityTracer + ForwardDiff, then report the
number of structurally asymmetric off-diagonal entries.

`η_init` is passed directly to `set!(η_fld, ...)` — can be a scalar or a function
of `(x, y)`.  When `print_values = true` the actual Float64 Jacobian values at
asymmetric positions are printed alongside the structural info.
"""
function check_symmetry(label::String, grid;
                        use_zstar::Bool   = false,
                        η_init            = 0.0,
                        print_values::Bool = false)
    sep = repeat("─", 60)
    @info sep
    @info "Config $label"
    @info sep

    # Prescribed velocity and free-surface fields
    u_fld = XFaceField(grid); set!(u_fld, 0.0); fill_halo_regions!(u_fld)
    v_fld = YFaceField(grid); set!(v_fld, 0.0); fill_halo_regions!(v_fld)
    w_fld = ZFaceField(grid); set!(w_fld, 0.0); fill_halo_regions!(w_fld)
    η_fld = Field{Center, Center, Nothing}(grid; indices = (:, :, 1))
    set!(η_fld, η_init); fill_halo_regions!(η_fld)

    vels = PrescribedVelocityFields(u = u_fld, v = v_fld, w = w_fld)
    fs   = PrescribedFreeSurface(displacement = η_fld)
    ADc0 = CenterField(grid)

    model = HydrostaticFreeSurfaceModel(
        grid;
        tracer_advection = Centered(order = 2),
        velocities       = vels,
        free_surface     = fs,
        tracers          = (; ADc = ADc0),
        closure          = HorizontalScalarDiffusivity(κ = 300.0),
        forcing          = linear_forcing,
    )

    if use_zstar
        launch!(CPU(), grid, surface_kernel_parameters(grid),
                _update_zstar_scaling!, η_fld, grid)
    end
    fill_halo_regions!(model.tracers.ADc)

    # Wet-cell index map
    Nx′, Ny′, Nz′ = size(ADc0)
    fNaN = CenterField(grid)
    mask_immersed_field!(fNaN, NaN)
    wet3D = .!isnan.(interior(on_architecture(CPU(), fNaN)))
    idx   = findall(wet3D)
    Nidx  = length(idx)
    @info "  wet cells: $Nidx / $(Nx′*Ny′*Nz′)"

    kp  = KernelParameters(1:Nx′, 1:Ny′, 1:Nz′)
    acm = get_active_cells_map(grid, Val(:interior))

    # In-place tendency using Cache contexts for preallocated Fields
    function _tend!(G, c, clock, ADc_cache, GADc_cache)
        interior(ADc_cache) .= 0
        for (n, ijk) in enumerate(idx)
            interior(ADc_cache)[ijk] = c[n]
        end
        fill_halo_regions!(ADc_cache)

        c_adv = model.advection[:ADc]
        c_frc = model.forcing[:ADc]
        c_ibc = immersed_boundary_condition(model.tracers[:ADc])

        args = (Val(1), Val(:ADc), c_adv, model.closure, c_ibc,
                model.buoyancy, model.biogeochemistry,
                model.transport_velocities, model.free_surface,
                (; ADc = ADc_cache), model.closure_fields, model.auxiliary_fields,
                clock, c_frc)

        launch!(CPU(), grid, kp, _compute_GADc!, GADc_cache, grid, args; active_cells_map = acm)
        G .= view(interior(GADc_cache), idx)
        return G
    end

    # Preallocate Fields for Cache contexts
    ADc_buf  = CenterField(grid)
    GADc_buf = CenterField(grid)

    # Warm-up (triggers compilation)
    cvec = ones(Nidx)
    Gvec = zeros(Nidx)
    @info "  Warm-up..."
    _tend!(Gvec, cvec, 0.0, ADc_buf, GADc_buf)

    # Detect sparsity pattern (may be asymmetric for IBG+zstar)
    @info "  Detecting sparsity pattern..."
    S = jacobian_sparsity_with_contexts(
        _tend!, Gvec, TracerSparsityDetector(; gradient_pattern_type = Set{UInt}), cvec,
        Constant(0.0), Cache(ADc_buf), Cache(GADc_buf),
    )

    # Symmetrize: ensure S[i,j] ↔ S[j,i]
    S_sym = S .| S'
    @info "  nnz(S) = $(nnz(S)), nnz(S_sym) = $(nnz(S_sym))"

    # Prepare and compute Jacobian with symmetric pattern
    sym_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector  = KnownJacobianSparsityDetector(S_sym),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    prep = prepare_jacobian(_tend!, Gvec, sym_backend, cvec,
                            Constant(0.0), Cache(ADc_buf), Cache(GADc_buf))
    buf = similar(sparsity_pattern(prep), Float64)
    @info "  Computing Jacobian..."
    jacobian!(_tend!, Gvec, buf, prep, sym_backend, cvec,
              Constant(0.0), Cache(ADc_buf), Cache(GADc_buf))
    M = copy(buf)

    # Structural symmetry check
    i_nz, j_nz, _ = findnz(M)
    M1   = sparse(i_nz, j_nz, true)
    asym = M1 - M1' .> 0
    n    = nnz(asym)

    @info "  nnz(M) = $(nnz(M)),  nnz(asym) = $n"
    if n > 0
        rows_a, cols_a, _ = findnz(asym)
        for (r, c) in zip(rows_a, cols_a)
            ri = Tuple(idx[r]); ci = Tuple(idx[c])
            if print_values
                @info "    ($r,$c): cell $ri ← $ci, offset = $(ci .- ri),  M[$r,$c] = $(M[r,c])"
            else
                @info "    ($r,$c): cell $ri ← $ci, offset = $(ci .- ri)"
            end
        end
    else
        @info "  SYMMETRIC ✓"
    end

    return n
end

# ── build all six grids ──────────────────────────────────────────────────────
@info "Building grids..."

grid_A = with_ibg(mutable_rect((Periodic, Flat, Bounded)))   # A: baseline
grid_B = with_ibg(mutable_rect((Bounded,  Flat, Bounded)))   # B: no periodic x
grid_C =          mutable_rect((Periodic, Flat, Bounded))    # C: no IBG
grid_D = with_ibg(fixed_rect( (Periodic, Flat, Bounded)))    # D: no zstar
grid_E =          fixed_rect( (Periodic, Flat, Bounded))     # E: simplest
grid_F = with_ibg(mutable_rect((Periodic, Flat, Bounded)))   # F: like A, η=sin(2π x/Lx)

@info "Grids built. Running configurations..."

# η = A·sin(2π x / Lx) with amplitude 1 m.
# The Flat y-topology means set! calls η_F(x, y) with y ≈ 0; y is ignored.
η_F(x) = sin(2π * x / Lx)

# ── run all configurations ───────────────────────────────────────────────────
results = Dict{String, Int}()
results["A: Periodic+IBG+zstar"]           = check_symmetry("A: Periodic+IBG+zstar",           grid_A; use_zstar = true)
results["B: Bounded+IBG+zstar"]            = check_symmetry("B: Bounded+IBG+zstar",            grid_B; use_zstar = true)
results["C: Periodic+noIBG+zstar"]         = check_symmetry("C: Periodic+noIBG+zstar",         grid_C; use_zstar = true)
results["D: Periodic+IBG+nozstar"]         = check_symmetry("D: Periodic+IBG+nozstar",         grid_D; use_zstar = false)
results["E: Periodic+noIBG+nozstar"]       = check_symmetry("E: Periodic+noIBG+nozstar",       grid_E; use_zstar = false)
results["F: Periodic+IBG+zstar+varyingη"]  = check_symmetry("F: Periodic+IBG+zstar+varyingη",  grid_F;
                                                             use_zstar = true, η_init = η_F,
                                                             print_values = true)

# ── summary ──────────────────────────────────────────────────────────────────
@info repeat("=", 60)
@info "SUMMARY: asymmetric entry counts"
@info repeat("=", 60)
for label in sort(collect(keys(results)))
    n = results[label]
    status = n == 0 ? "SYMMETRIC ✓" : "ASYMMETRIC ($n entries)"
    @info "  $label → $status"
end
@info repeat("=", 60)
@info "step3_simplify.jl complete."
