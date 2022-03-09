module OceanBoundaryLayerParameterizations

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using OceanLearning
using OceanLearning.Models: set!
using CairoMakie, LaTeXStrings, OrderedCollections
using Oceananigans, FileIO

export
    # lesbrary_paths.jl
    TwoDaySuite, FourDaySuite, SixDaySuite, GeneralStrat,

    # one_dimensional_ensemble_model.jl
    OneDimensionalEnsembleModel,

    # EKI_hyperparameter_search.jl
    plot_stds_within_bounds, plot_prior_variance, plot_num_ensemble_members, 
    plot_observation_noise_level, plot_prior_variance_and_obs_noise_level,

    # visualize_profile_predictions.jl
    visualize!,
    visualize_and_save!

include("utils/lesbrary_paths.jl")
include("utils/one_dimensional_ensemble_model.jl")
include("EKI_hyperparameter_search.jl")
include("visualize_profile_predictions_utils.jl")
include("utils/visualize_profile_predictions.jl")


end # module
