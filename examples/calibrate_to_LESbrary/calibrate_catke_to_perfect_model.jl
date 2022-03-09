pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using OceanLearning, LinearAlgebra, CairoMakie
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using OceanLearning.EnsembleKalmanInversions: NaNResampler, FullEnsembleDistribution

include("utils/lesbrary_paths.jl")
include("utils/parameters.jl")
include("utils/visualize_profile_predictions.jl")

examples_path = joinpath(pathof(OceanLearning), "../../examples")
include(joinpath(examples_path, "intro_to_inverse_problems.jl"))

# Pick a parameter set defined in `./parameters.jl`
parameter_set = CATKEParametersRiDependent
closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

# Pick the secret `true_parameters`
true_parameters = (Cᵟu = 0.5, CᴷRiʷ = 1.0, Cᵂu★ = 2.0, CᵂwΔ = 1.0, Cᴷeʳ = 5.0, Cᵟc = 0.5, Cᴰ = 2.0, Cᴷc⁻ = 0.5, Cᴷe⁻ = 0.2, Cᴷcʳ = 3.0, Cᴸᵇ = 1.0, CᴷRiᶜ = 1.0, Cᴷuʳ = 4.0, Cᴷu⁻ = 0.8, Cᵟe = 0.5)
true_closure = closure_with_parameters(closure, true_parameters)

# Generate and load synthetic observations
kwargs = (tracers = (:b, :e), Δt = 10.0, stop_time = 4hours)

# Note: if an output file of the same name already exist, `generate_synthetic_observations` will return the existing path and skip re-generating the data.
observ_path1 = generate_synthetic_observations("perfect_model_observation1"; Qᵘ = 2e-4, Qᵇ = 4e-8, f₀ = 1e-4, closure = true_closure, kwargs...)
observ_path2 = generate_synthetic_observations("perfect_model_observation2"; Qᵘ = -1e-4, Qᵇ = 0, f₀ = 0, closure = true_closure, kwargs...)
observation1 = SyntheticObservations(observ_path1, field_names = (:b, :e, :u, :v), normalize = ZScore)
observation2 = SyntheticObservations(observ_path2, field_names = (:b, :e, :u, :v), normalize = ZScore)
observations = [observation1, observation2]

# Set up ensemble model
ensemble_simulation, closure = build_ensemble_simulation(observations; Nensemble = 50)

# Build free parameters
build_prior(name) = ConstrainedNormal(0.0, 1.0, bounds(name) .* 0.5...)
free_parameters = FreeParameters(named_tuple_map(names(parameter_set), build_prior))

# Pack everything into Inverse Problem `calibration`
calibration = InverseProblem(observations, ensemble_simulation, free_parameters, output_map = ConcatenatedOutputMap());

# Make sure the forward map evaluated on the true parameters matches the observation map
x = forward_map(calibration, true_parameters)[:, 1:1];
y = observation_map(calibration);
@show x == y

directory = "calibrate_catke_to_perfect_model/"
isdir(directory) || mkpath(directory)

visualize!(calibration, true_parameters;
    field_names = (:u, :v, :b, :e),
    directory = directory,
    filename = "perfect_model_visual_true_params.png"
)

# Calibrate

noise_covariance = Matrix(Diagonal(observation_map_variance_across_time(calibration)[1, :, 1] .+ 1e-5))
resampler = NaNResampler(; abort_fraction=0.5, distribution=FullEnsembleDistribution())

eki = EnsembleKalmanInversion(calibration; noise_covariance = noise_covariance, 
                                           resampler = resampler)
params = iterate!(eki; iterations = 5)

###
### Summary Plots
###

using CairoMakie
using LinearAlgebra

visualize!(calibration, params;
    field_names = [:u, :v, :b, :e],
    directory = directory,
    filename = "perfect_model_visual_calibrated.png"
)
@show params

# Last, we visualize the outputs of EKI calibration.

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
    hlines!(ax, [true_parameters[pname]], color = :red)
end
save(joinpath(directory, "perfect_model_calibration_parameter_convergence.png"), f);

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
        vlines!(axmain, [true_parameters[pname1]], color = :red)
        vlines!(axtop, [true_parameters[pname1]], color = :red)
        hlines!(axmain, [true_parameters[pname2]], color = :red, alpha = 0.6)
        hlines!(axright, [true_parameters[pname2]], color = :red, alpha = 0.6)
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
        save(joinpath(directory, "eki_$(pname1)_$(pname2).png"), f)
    end
end

# Compare EKI result to true values

weight_distances = [norm(collect(ensemble_means[iter]) .- collect(true_parameters)) for iter = 0:N_iter-1]
output_distances = [mapslices(norm, (forward_map(calibration, [ensemble_means...])[:, 1:N_iter] .- y), dims = 1)...]

f = Figure()
lines(f[1, 1], iter_range, weight_distances, color = :red, linewidth = 2,
    axis = (title = "Parameter distance",
        xlabel = "Iteration",
        ylabel = "|θ̅ₙ - θ⋆|",
        yscale = log10))
lines(f[1, 2], iter_range, output_distances, color = :blue, linewidth = 2,
    axis = (title = "Output distance",
        xlabel = "Iteration",
        ylabel = "|G(θ̅ₙ) - y|",
        yscale = log10))

save(joinpath(directory, "error_convergence_summary.png"), f);
