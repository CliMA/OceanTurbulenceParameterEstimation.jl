module ParameterEstimation

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.Data
using ..OceanTurbulenceParameterEstimation.Models
using ..OceanTurbulenceParameterEstimation.Models.CATKEVerticalDiffusivityModel
using ..OceanTurbulenceParameterEstimation.LossFunctions

using Oceananigans
using CairoMakie: Figure

import ..OceanTurbulenceParameterEstimation.Models: set!
import ..OceanTurbulenceParameterEstimation.LossFunctions: model_time_series

export
       Parameters,
       InverseProblem,
       validation_loss_reduction,
       relative_weight_options,
       model_time_series,
       set!,

       # EKI
       eki

relative_weight_options = Dict(
                "all_e"     => Dict(:b => 0.0, :u => 0.0, :v => 0.0, :e => 1.0),
                "all_T"     => Dict(:b => 1.0, :u => 0.0, :v => 0.0, :e => 0.0),
                "uniform"   => Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0),
                "all_but_e" => Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 0.0),
                "all_uv"    => Dict(:b => 0.0, :u => 1.0, :v => 1.0, :e => 0.0),
                "mostly_T"  => Dict(:b => 1.0, :u => 0.5, :v => 0.5, :e => 0.0)
)

Base.@kwdef struct Parameters{T <: UnionAll}
    RelevantParameters::T
    ParametersToOptimize::T
end

struct InverseProblem{DB, PM, RW, LF, FP}
        data_batch::DB
        model::PM
        relative_weights::RW # field weights
        loss::LF
        default_parameters::FP
end

(ip::InverseProblem)(θ=ip.default_parameters) = ip.loss(ip.model, ip.data_batch, θ)

model_time_series(ip::InverseProblem, parameters) = model_time_series(ip.loss.ParametersToOptimize(parameters), ip.model, ip.data_batch, ip.loss.Δt)

function validation_loss_reduction(calibration::InverseProblem, validation::InverseProblem, parameters::FreeParameters)
    validation_loss = validation.loss(parameters)
    calibration_loss = calibration.loss(parameters)

    default_validation_loss = validation.loss(ce.default_parameters)
    default_calibration_loss = calibration.loss(ce.default_parameters)

    validation_loss_reduction = validation_loss/default_validation_loss
    println("Parameters: $([parameters...])")
    println("Validation loss reduction: $(validation_loss_reduction)")
    println("Training loss reduction: $(calibration_loss/default_calibration_loss)")

    return validation_loss_reduction
end

include("catke_vertical_diffusivity_model_setup.jl")
include("EKI/EKI.jl")

using .EKI

end # module
