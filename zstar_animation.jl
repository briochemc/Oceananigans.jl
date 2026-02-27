using CairoMakie

Lx, Lz = 1e3, 25 # m

x = range(-Lx/2, stop=Lx/2, length=200)

σ = Lx/15 # a horizontal length scale

# bottom, H(x)
x₀, h₀ = -Lx/3,  15 # m
slope = @. h₀ * (1 + tanh(-(x - x₀) / σ)) / 2
x₀, h₀ = 0, 12 # m
mountain = @. h₀ * sech((x - x₀) / 0.5σ)^2
H = @. Lz - slope - mountain

# free surface, η(x)
x₀ = -Lx/8
η₀ = 2.5 # m
t = Observable(0.0)
η = @lift @. -η₀ * ((x - x₀)^2 / σ^2 - 1) * exp(-(x - x₀)^2 / 2σ^2) * cos(2π * $t)

fig = Figure(size=(1000, 400))
axis_kwargs = (titlesize = 20, xlabel = "x", ygridvisible = false)
ax1 = Axis(fig[1, 1]; title="ZCoordinate", ylabel="z", axis_kwargs...)
ax2 = Axis(fig[1, 2]; title="ZStarCoordinate", axis_kwargs...)

for ax in (ax1, ax2)
    band!(ax, x, -H, η, color = (:dodgerblue, 0.5))
    band!(ax, x, -1.1 * Lz, -H, color = (:orange, 0.2))
    lines!(ax, x,  η, linewidth=5, color=:darkblue)
    lines!(ax, x, -H, linewidth=5, color=:darkgrey)
end

for r in range(-Lz, stop=0, length=6)
    # ZCoordinate
    z = r * ones(size(x))
    lines!(ax1, x, z, color=:crimson, linestyle=:dash)

    # ZStarCoordinate
    z = lift(η) do η_val
        @. r * (H + η_val) / H + η_val
    end
    lines!(ax2, x, z, color=:crimson, linestyle=:dash)
end

Nt = 50
times = 0:1/Nt:1-1/Nt # one period of cos(2πt)
CairoMakie.record(fig, "z-zstar.gif", times, framerate=12) do val
    t[] = val
end