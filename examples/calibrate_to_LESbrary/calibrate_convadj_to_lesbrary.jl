# Calibrate convective adjustment closure parameters to LESbrary 2-day "free_convection" simulation

using OceanLearning, LinearAlgebra, CairoMakie
using Oceananigans.Units

include("utils/lesbrary_paths.jl")
include("utils/one_dimensional_ensemble_model.jl")

# Build an observation from "free convection" LESbrary simulation

LESbrary_directory = "/Users/adelinehillier/Desktop/dev/2DaySuite/"

suite = OrderedDict("2d_free_convection" => (
    filename = joinpath(LESbrary_directory, "free_convection/instantaneous_statistics.jld2"),
    fields = (:b,)))

observations = SyntheticObservationsBatch(suite; first_iteration = 13, last_iteration = nothing, normalize = ZScore, Nz = 32)

closure = ConvectiveAdjustmentVerticalDiffusivity(;
    convective_κz = 1.0,
    background_κz = 1e-4
)

# Build an ensemble simulation based on observation

ensemble_size = 30
ensemble_model = OneDimensionalEnsembleModel(observations;
    architecture = CPU(),
    ensemble_size = ensemble_size,
    closure = closure
)

ensemble_simulation = Simulation(ensemble_model; Δt = 10seconds, stop_time = 4days)

# Specify priors

priors = (
    convective_κz = ConstrainedNormal(0.0, 1.0, 0.1, 1.0),
    background_κz = ConstrainedNormal(0.0, 1.0, 0.0, 10e-4)
)

free_parameters = FreeParameters(priors)

# Specify an output map that tracks 3 uniformly spaced time steps, ignoring the initial condition
track_times = floor.(range(1, stop = length(observations[1].times), length = 3))
popfirst!(track_times)
output_map = ConcatenatedOutputMap(track_times)

# Build `InverseProblem`
calibration = InverseProblem(observations, ensemble_simulation, free_parameters; output_map = output_map)

# Ensemble Kalman Inversion

eki = EnsembleKalmanInversion(calibration; noise_covariance = 0.01)

iterations = 10
iterate!(eki; iterations = iterations)

# Visualize the outputs of EKI calibration. Plots will be stored in `directory`.

directory = "calibrate_convadj_to_lesbrary/$(iterations)_iters_$(ensemble_size)_particles/"
isdir(directory) || mkpath(directory)

### Parameter convergence plot

# Vector of NamedTuples, ensemble mean at each iteration
ensemble_means = getproperty.(eki.iteration_summaries, :ensemble_mean)

# N_param x N_iter matrix, ensemble covariance at each iteration
θθ_std_arr = sqrt.(hcat(diag.(getproperty.(eki.iteration_summaries, :ensemble_cov))...))

N_param, N_iter = size(θθ_std_arr)
iter_range = 0:(N_iter-1)
pnames = calibration.free_parameters.names

n_cols = 3
n_rows = Int(ceil(N_param / n_cols))
ax_coords = [(i, j) for i = 1:n_rows, j = 1:n_cols]

f = Figure(resolution = (500n_cols, 200n_rows))
for (i, pname) in enumerate(pnames)
    coords = ax_coords[i]
    ax = Axis(f[coords...],
        xlabel = "Iteration",
        xticks = iter_range,
        ylabel = string(pname))

    ax.ylabelsize = 20

    mean_values = [getproperty.(ensemble_means, pname)...]
    lines!(ax, iter_range, mean_values)
    band!(ax, iter_range, mean_values .+ θθ_std_arr[i, :], mean_values .- θθ_std_arr[i, :])
end

save(joinpath(directory, "conv_adj_to_LESbrary_parameter_convergence.pdf"), f);

### Pairwise ensemble plots

N_param, N_iter = size(θθ_std_arr)
for pname1 in pnames, pname2 in pnames
    if pname1 != pname2

        f = Figure()
        axtop = Axis(f[1, 1])
        axmain = Axis(f[2, 1],
            xlabel = string(pname1),
            ylabel = string(pname2)
        )
        axright = Axis(f[2, 2])
        scatters = []
        for iteration in [0, 1, 2, N_iter - 1]
            ensemble = eki.iteration_summaries[iteration].parameters
            ensemble = [[particle[pname1], particle[pname2]] for particle in ensemble]
            ensemble = transpose(hcat(ensemble...)) # N_ensemble x 2
            push!(scatters, scatter!(axmain, ensemble))
            density!(axtop, ensemble[:, 1])
            density!(axright, ensemble[:, 2], direction = :y)
        end
        colsize!(f.layout, 1, Fixed(300))
        colsize!(f.layout, 2, Fixed(200))
        rowsize!(f.layout, 1, Fixed(200))
        rowsize!(f.layout, 2, Fixed(300))
        Legend(f[1, 2], scatters,
            ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration $N_iter"],
            position = :lb)
        hidedecorations!(axtop, grid = false)
        hidedecorations!(axright, grid = false)
        xlims!(axright, 0, 10)
        ylims!(axtop, 0, 10)
        save(joinpath(directory, "conv_adj_to_LESbrary_eki_$(pname1)_$(pname2).pdf"), f)
    end
end

# Compare EKI result to true values

y = observation_map(calibration)
output_distances = [mapslices(norm, (forward_map(calibration, [ensemble_means...])[:, 1:N_iter] .- y), dims = 1)...]

f = Figure()
lines(f[1, 1], iter_range, output_distances, color = :blue, linewidth = 2,
    axis = (title = "Output distance",
        xlabel = "Iteration",
        ylabel = "|G(θ̅ₙ) - y|",
        yscale = log10))
save(joinpath(directory, "conv_adj_to_LESbrary_error_convergence_summary.pdf"), f);

include("./utils/visualize_profile_predictions.jl")
visualize!(calibration, ensemble_means[end];
    field_names = (:b,),
    directory = directory,
    filename = "realizations.pdf"
)

θglobalmin = NamedTuple((:convective_κz => 0.275, :background_κz => 0.000275))
visualize!(calibration, θglobalmin;
    field_names = (:b,),
    directory = directory,
    filename = "realizations_θglobalmin.pdf"
)

## Visualize loss landscape

name = "Loss landscape"

pvalues = Dict(
    :convective_κz => collect(0.075:0.025:1.025),
    :background_κz => collect(0e-4:0.25e-4:10e-4),
)

ni = length(pvalues[:convective_κz])
nj = length(pvalues[:background_κz])

params = hcat([[pvalues[:convective_κz][i], pvalues[:background_κz][j]] for i = 1:ni, j = 1:nj]...)
xc = params[1, :]
yc = params[2, :]

# build an `InverseProblem` that can accommodate `ni*nj` ensemble members 
ensemble_model = OneDimensionalEnsembleModel(observations;
    architecture = CPU(),
    ensemble_size = ni * nj,
    closure = closure)
ensemble_simulation = Simulation(ensemble_model; Δt = 10seconds, stop_time = 6days)
calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

y = observation_map(calibration)

using FileIO
# a = forward_map(calibration, params) .- y
# save("calibrate_convadj_to_lesbrary/loss_landscape.jld2", "a", a)

a = load("calibrate_convadj_to_lesbrary/loss_landscape.jld2")["a"]
zc = [mapslices(norm, a, dims = 1)...]

# 2D contour plot with EKI particles superimposed
begin
    f = Figure()
    ax1 = Axis(f[1, 1],
        title = "EKI Particle Traversal Over Loss Landscape",
        xlabel = "convective_κz",
        ylabel = "background_κz")

    co = CairoMakie.contourf!(ax1, xc, yc, zc, levels = 50, colormap = :default)

    cvt(iter) = hcat(collect.(eki.iteration_summaries[iter].parameters)...)
    diffc = cvt(2) .- cvt(1)
    diff_mag = mapslices(norm, diffc, dims = 1)
    # diffc ./= 2
    us = diffc[1, :]
    vs = diffc[2, :]
    xs = cvt(1)[1, :]
    ys = cvt(1)[2, :]

    arrows!(xs, ys, us, vs, arrowsize = 10, lengthscale = 0.3,
        arrowcolor = :yellow, linecolor = :yellow)

    am = argmin(zc)
    minimizing_params = [xc[am] yc[am]]

    scatters = [scatter!(ax1, minimizing_params, marker = :x, markersize = 30)]
    for (i, iteration) in enumerate([1, 2, iterations])
        ensemble = eki.iteration_summaries[iteration].parameters
        ensemble = [[particle[:convective_κz], particle[:background_κz]] for particle in ensemble]
        ensemble = transpose(hcat(ensemble...)) # N_ensemble x 2
        push!(scatters, scatter!(ax1, ensemble))
    end
    Legend(f[1, 2], scatters,
        ["Global minimum", "Initial ensemble", "Iteration 1", "Iteration $(iterations)"],
        position = :lb)

    save(joinpath(directory, "loss_contour.pdf"), f)
end

# 3D loss landscape
begin
    f = Figure()
    ax1 = Axis3(f[1, 1],
        title = "Loss Landscape",
        xlabel = "convective_κz",
        ylabel = "background_κz",
        zlabel = "MSE loss"
    )

    CairoMakie.surface!(ax1, xc, yc, zc, colorscheme = :thermal)

    save(joinpath(directory, "loss_landscape.png"), f)
end
