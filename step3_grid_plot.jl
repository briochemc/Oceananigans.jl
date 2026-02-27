"""
Visualization of the IBG + PartialCellBottom grid from step3_simplify.jl
in the x-z plane, showing the partial cell bottom staircase, wet/immersed
cells, and the asymmetric Jacobian dependencies.

Run with:  julia --project=@diffocean step3_grid_plot.jl
"""

using CairoMakie

# ── Grid parameters from step3_simplify.jl ──────────────────────────────────
Nx, Nz = 4, 4
H = 100.0
Lx = 1000.0  # km for display

x_faces = range(0, Lx, length = Nx + 1) |> collect
x_centers = [(x_faces[i] + x_faces[i+1]) / 2 for i in 1:Nx]
z_faces = collect(range(-H, 0, length = Nz + 1))
z_centers = [(z_faces[k] + z_faces[k+1]) / 2 for k in 1:Nz]

bottom = [-H * (0.5 + 0.5 * (i - 1) / (Nx - 1)) for i in 1:Nx]
# bottom ≈ [-50.0, -66.67, -83.33, -100.0]

# ── Staircase coordinates for PartialCellBottom ─────────────────────────────
function staircase_coords(x_faces, bottom)
    xs = Float64[]
    zs = Float64[]
    for i in eachindex(bottom)
        push!(xs, x_faces[i]);   push!(zs, bottom[i])
        push!(xs, x_faces[i+1]); push!(zs, bottom[i])
    end
    return xs, zs
end

sx, sz = staircase_coords(x_faces, bottom)

# ── Figure ──────────────────────────────────────────────────────────────────
fig = Figure(size = (900, 500))
ax = Axis(fig[1, 1];
    xlabel = "x (km)", ylabel = "z (m)",
    title = "IBG + PartialCellBottom grid (Nx=$Nx, Nz=$Nz)",
    xticks = (x_centers, ["i=$i" for i in 1:Nx]),
    xticksize = 10,
    yticks = (z_centers, ["k=$k" for k in 1:Nz]),
    yticksize = 10,
    xminorticks = x_faces,
    xminorticksvisible = true,
    xminorgridvisible = true,
    xminorgridstyle = :dash,
    xminorgridcolor = (:gray, 0.5),
    xgridvisible = true,
    xgridstyle = :solid,
    xgridcolor = (:gray, 0.3),
    yminorticks = z_faces,
    yminorticksvisible = true,
    yminorgridvisible = true,
    yminorgridstyle = :dash,
    yminorgridcolor = (:gray, 0.5),
    ygridvisible = true,
    ygridstyle = :solid,
    ygridcolor = (:gray, 0.3),
)

# ── Water band (above staircase, below z=0) ─────────────────────────────────
water_xs = vcat(sx, reverse(sx))
water_zs = vcat(zeros(length(sx)), reverse(sz))
poly!(ax, Point2f.(water_xs, water_zs); color = (:dodgerblue, 0.3))

# ── Land band (below staircase) ─────────────────────────────────────────────
land_xs = vcat(sx, reverse(sx))
land_zs = vcat(sz, fill(-H - 10, length(sx)))
poly!(ax, Point2f.(land_xs, land_zs); color = (:sienna, 0.3))

# ── Staircase bottom line ───────────────────────────────────────────────────
lines!(ax, sx, sz; linewidth = 3, color = :sienna)

# ── Mark immersed cells ─────────────────────────────────────────────────────
immersed = [(1, 1), (1, 2), (2, 1)]
for (i, k) in immersed
    x0, x1 = x_faces[i], x_faces[i+1]
    z0, z1 = z_faces[k], z_faces[k+1]
    poly!(ax, Point2f.([(x0, z0), (x1, z0), (x1, z1), (x0, z1)]);
          color = (:sienna, 0.5), strokewidth = 1, strokecolor = :sienna)
    text!(ax, (x0 + x1) / 2, (z0 + z1) / 2;
          text = "×", align = (:center, :center), fontsize = 24, color = :white)
end

# ── Label wet cells with (i, k) ────────────────────────────────────────────
wet_cells = [(i, k) for i in 1:Nx for k in 1:Nz if !((i, k) in immersed)]
for (i, k) in wet_cells
    text!(ax, x_centers[i], z_centers[k];
          text = "($i,$k)", align = (:center, :center), fontsize = 14,
          color = :black)
end

# ── Asymmetric dependencies (red arrows) ────────────────────────────────────
# From step3_simplify.jl output:
#   (2,1,2) ← (3,1,1): cell (2,1,2) depends on cell (3,1,1), offset (1,0,-1)
#   (1,1,3) ← (2,1,2): cell (1,1,3) depends on cell (2,1,2), offset (1,0,-1)
#   (1,1,3) ← (4,1,2): cell (1,1,3) depends on cell (4,1,2), offset (3,0,-1) via periodic
# Arrow goes FROM source cell TO destination cell (direction of dependency)
asymmetric = [
    ((3, 1), (2, 2), "G(2,2)←c(3,1)"),
    ((2, 2), (1, 3), "G(1,3)←c(2,2)"),
    ((4, 2), (1, 3), "G(1,3)←c(4,2)"),
]

for (n, (src, dst, label)) in enumerate(asymmetric)
    x1, z1 = x_centers[src[1]], z_centers[src[2]]
    x2, z2 = x_centers[dst[1]], z_centers[dst[2]]
    offset = n == 3 ? 5.0 : 0.0  # vertical offset for the periodic arrow label
    arrows2d!(ax, [x1], [z1], [x2 - x1], [z2 - z1];
              color = :red, linewidth = 2, tipwidth = 12)
    text!(ax, (x1 + x2) / 2, (z1 + z2) / 2 + offset;
          text = label, align = (:center, :bottom), fontsize = 10, color = :red)
end

# ── Legend annotation ───────────────────────────────────────────────────────
text!(ax, Lx * 0.02, -3;
      text = "Red arrows: one-directional dependencies (asymmetric entries in Jacobian)",
      align = (:left, :top), fontsize = 10, color = :red)

ylims!(ax, -H - 10, 5)
xlims!(ax, -Lx * 0.05, Lx * 1.05)

save("step3_grid_plot.png", fig; px_per_unit = 2)
@info "Saved step3_grid_plot.png"
fig
