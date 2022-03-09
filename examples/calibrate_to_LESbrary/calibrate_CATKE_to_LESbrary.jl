pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using Plots, LinearAlgebra, Distributions, JLD2
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using OceanLearning

include("utils/lesbrary_paths.jl")
include("utils/one_dimensional_ensemble_model.jl")
include("utils/parameters.jl")
include("utils/visualize_profile_predictions.jl")

# two_day_suite_dir = "/Users/gregorywagner/Projects/OceanLearning/data/2DaySuite"
# four_day_suite_dir = "/Users/gregorywagner/Projects/OceanLearning/data/4DaySuite"
# six_day_suite_dir = "/Users/gregorywagner/Projects/OceanLearning/data/6DaySuite"

# two_day_suite = TwoDaySuite(two_day_suite_dir)
# four_day_suite = FourDaySuite(four_day_suite_dir)
# six_day_suite = SixDaySuite(six_day_suite_dir)

# calibration = InverseProblem(two_day_suite, parameters; relative_weights = relative_weight_options["all_but_e"],
#                              architecture = GPU(), ensemble_size = 10, Δt = 30.0)

# validation = InverseProblem(four_day_suite, calibration; Nz = 32);

#####
##### Set up ensemble model
#####

## NEED TO IMPLEMENT COARSE-GRAINING
##
##

lesbrary_directory = "/Users/adelinehillier/Desktop/dev/"

observations = TwoDaySuite(lesbrary_directory; first_iteration = 13, last_iteration = nothing, normalize = ZScore, Nz = 128)

parameter_set = CATKEParametersRiDependent
closure = closure_with_parameter_set(CATKEVerticalDiffusivity(Float64;), parameter_set)

ensemble_model = OneDimensionalEnsembleModel(observations;
    architecture = CPU(),
    ensemble_size = 20,
    closure = closure
)

ensemble_simulation = Simulation(ensemble_model; Δt = 10seconds, stop_time = 2days)

pop!(ensemble_simulation.diagnostics, :nan_checker)

#####
##### Build free parameters
#####

free_parameter_names = keys(parameter_set.defaults)
free_parameter_means = collect(values(parameter_set.defaults))
priors = NamedTuple(pname => ConstrainedNormal(0.0, 1.0, bounds(pname) .* 0.5...) for pname in free_parameter_names)

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters);

# #####
# ##### Calibrate
# #####

iterations = 5
# eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)
# params = iterate!(eki; iterations = iterations)

directory = 
# visualize!(calibration, params;
#     field_names = [:u, :v, :b, :e],
#     directory = @__DIR__,
#     filename = "perfect_model_visual_calibrated.png"
# )
# @show params
