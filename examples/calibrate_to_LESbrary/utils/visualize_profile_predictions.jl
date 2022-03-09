using OceanLearning.InverseProblems: vectorize, forward_run!, transpose_model_output
using CairoMakie

include("visualize_profile_predictions_utils.jl")

"""
    visualize!(ip::InverseProblem, parameters;
                    field_names = [:u, :v, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png"
                    )

    For visualizing 1-dimensional time series predictions.
"""
function visualize!(ip::InverseProblem, parameters;
                    field_names = [:u, :v, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png"
                    )

    isdir(directory) || makedir(directory)

    model = ip.simulation.model

    n_fields = length(field_names)

    observations = vectorize(ip.observations)

    forward_run!(ip, parameters)

    # Vector of SyntheticObservations objects, one for each observation
    predictions = transpose_model_output(ip.time_series_collector, observations)
        
    fig = Figure(resolution = (200*(length(field_names)+1), 200*(length(ip.observations)+1)), font = "CMU Serif")
    colors = [:black, :red, :blue]

    function empty_plot!(fig_position)
        ax = fig_position = Axis(fig_position)
        hidedecorations!(ax)
        hidespines!(ax, :t, :b, :l, :r)
    end

    calibration.observations[1].field_time_serieses.b[1].grid

    for (oi, observation) in enumerate(observations)

        i = oi + 1
        prediction = predictions[oi]

        targets = observation.times
        snapshots = round.(Int, range(1, length(targets), length=3))

        Qᵇ = model.tracers.b.boundary_conditions.top.condition[1,oi]
        Qᵘ = model.velocities.u.boundary_conditions.top.condition[1,oi]
        fv = model.coriolis[1,oi].f 

        empty_plot!(fig[i,1])
        text!(fig[i,1], "Qᵇ = $(tostring(Qᵇ)) m⁻¹s⁻³\nQᵘ = $(tostring(Qᵘ)) m⁻¹s⁻²\nf = $(tostring(fv)) s⁻¹", 
                    position = (0, 0), 
                    align = (:center, :center), 
                    textsize = 15,
                    justification = :left)

        for (j, field_name) in enumerate(field_names)

            middle = j > 1 && j < n_fields
            remove_spines = j == 1 ? (:t, :r) : j == n_fields ? (:t, :l) : (:t, :l, :r)
            axis_position = j == n_fields ? (ylabelposition=:right, yaxisposition=:right) : NamedTuple()

            j += 1 # reserve the first column for row labels

            info = field_guide[field_name]

            grid = observation.grid

            z = field_name ∈ [:u, :v] ? grid.zᵃᵃᶠ[1:grid.Nz] : grid.zᵃᵃᶜ[1:grid.Nz]

            to_plot = field_name ∈ keys(observation.field_time_serieses)

            if to_plot

                # field data for each time step
                obsn = getproperty(observation.field_time_serieses, field_name)
                pred = getproperty(prediction.field_time_serieses, field_name)

                ax = Axis(fig[i,j]; xlabelpadding=0, xtickalign=1, ytickalign=1, 
                                            merge(axis_position, info.axis_args)...)

                hidespines!(ax, remove_spines...)

                middle && hideydecorations!(ax, grid=false)

                lins = []
                for (color_index, t) in enumerate(snapshots)
                    l = lines!([obsn[1,1,1:grid.Nz,t] .* info.scaling ...], z; color = (colors[color_index], 0.4))
                    push!(lins, l)
                    l = lines!([pred[1,1,1:grid.Nz,t] .* info.scaling ...], z; color = (colors[color_index], 1.0), linestyle = :dash)
                    push!(lins, l)
                end

                times = @. round((targets[snapshots] - targets[snapshots[1]]) / 86400, sigdigits=2)

                legendlabel(time) = ["LES, t = $time days", "Model, t = $time days"]
                Legend(fig[1,2:3], lins, vcat([legendlabel(time) for time in times]...), nbanks=2)
                lins = []
            else
                
                empty_plot!(fig[i,j])
            end
        end
    end

    save(joinpath(directory, filename), fig, px_per_unit = 2.0)
    return nothing
end

# visualize!(ip::InverseProblem, parameters; kwargs...) = visualize!(model_time_series(ip, parameters); kwargs...)

# function visualize_and_save!(calibration, validation, parameters, directory; fields=[:u, :v, :b, :e])
#         isdir(directory) || makedir(directory)

#         path = joinpath(directory, "results.txt")
#         o = open_output_file(path)
#         write(o, "Training relative weights: $(calibration.relative_weights) \n")
#         write(o, "Validation relative weights: $(validation.relative_weights) \n")
#         write(o, "Training default parameters: $(validation.default_parameters) \n")
#         write(o, "Validation default parameters: $(validation.default_parameters) \n")

#         write(o, "------------ \n \n")
#         default_parameters = calibration.default_parameters
#         train_loss_default = calibration(default_parameters)
#         valid_loss_default = validation(default_parameters)
#         write(o, "Default parameters: $(default_parameters) \nLoss on training: $(train_loss_default) \nLoss on validation: $(valid_loss_default) \n------------ \n \n")

#         train_loss = calibration(parameters)
#         valid_loss = validation(parameters)
#         write(o, "Parameters: $(parameters) \nLoss on training: $(train_loss) \nLoss on validation: $(valid_loss) \n------------ \n \n")

#         write(o, "Training loss reduction: $(train_loss/train_loss_default) \n")
#         write(o, "Validation loss reduction: $(valid_loss/valid_loss_default) \n")
#         close(o)

#         parameters = calibration.parameters.ParametersToOptimize(parameters)

#         for inverse_problem in [calibration, validation]

#             all_data = inverse_problem.observations
#             simulation = inverse_problem.simulation
#             set!(simulation.model, parameters)

#             for data_length in Set(length.(getproperty.(all_data, :t)))

#                 observations = [d for d in all_data if length(d.t) == data_length]
#                 days = observations[1].t[end]/86400

#                 new_ip = InverseProblem()

#                 visualize!(simulation, observations, parameters;
#                             fields = fields,
#                             filename = joinpath(directory, "$(days)_day_simulations.png"))
#             end
#         end
    
#     return nothing
# end
