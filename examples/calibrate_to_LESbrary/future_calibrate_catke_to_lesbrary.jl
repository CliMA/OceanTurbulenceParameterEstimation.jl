# In this example, we use EKI to tune the closure parameters of a HydrostaticFreeSurfaceModel 
# with a CATKEBasedVerticalDiffusivity closure in order to align the predictions of the model 
# to those of a high-resolution LES data generated in LESbrary.jl. Here `predictions` refers to the
# 1-D profiles of temperature, velocity, and turbulent kinetic energy horizontally averaged over a
# 3-D physical domain.

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "projects", "OceanBoundaryLayerParameterizations", "src"))

image_dir = joinpath(@__DIR__, "quick_calibrate")
mkpath(image_dir)

using Oceananigans
using OceanLearning
using OceanBoundaryLayerParameterizations

two_day_suite_dir = "/Users/gregorywagner/Projects/OceanLearning/data/2DaySuite"
four_day_suite_dir = "/Users/gregorywagner/Projects/OceanLearning/data/4DaySuite"
six_day_suite_dir = "/Users/gregorywagner/Projects/OceanLearning/data/6DaySuite"

two_day_suite = TwoDaySuite(two_day_suite_dir)
four_day_suite = FourDaySuite(four_day_suite_dir)
six_day_suite = SixDaySuite(six_day_suite_dir)

calibration = InverseProblem(two_day_suite, parameters; relative_weights = relative_weight_options["all_but_e"],
                             architecture = GPU(), ensemble_size = 10, Δt = 30.0)

validation = InverseProblem(four_day_suite, calibration; Nz = 64);

# Loss on default parameters
l0 = calibration()

# Example parameters
θ = calibration.default_parameters

# Loss on parameters θ.
# θ can be 
#   1. a vector
#   2. a FreeParameters object
#   3. a vector of parameter vectors (one for each ensemble member)
#   4. or a vector of FreeParameter objects (one for each ensemble member)
# If (1) or (2), the ensemble members are redundant and the loss is computed for just the one parameter set.
lθ = calibration(θ)

# Output files/figures
directory = joinpath(pwd(), "quick_calibrate")

# Run the model forward and store the solution
output = model_time_series(calibration, θ)

# Run the model forward with parameters θ and visualize the solution compared to the truth
visualize!(output; filename = "visualize_default_parameters.png")

#=

# Use EKI to calibrate the model parameters
calibration_algorithm = EKI(noise_level = 10^(-2.0),
                            N_iter = 15,
                            stds_within_bounds = 0.6,
                            informed_priors = false)

best_parameters = calibrate(calibration; algorithm = calibration_algorithm)

loss = calibration(best_parameters)

=#

# Runs `visualize!` and records a summary of the calibration results in a `result.txt` file.
visualize_and_save!(calibration, validation, θ, directory)
