using LaTeXStrings

import OceanLearning.TurbulenceClosureParameters: closure_with_parameters

parameter_guide = Dict(:Cᴰ => (name = "Dissipation parameter (TKE equation)", latex = L"C^D",
          default = 2.9079, bounds = (0.0, 10.0)), :Cᴸᵇ => (name = "Mixing length parameter", latex = L"C^{\ell}_b",
          default = 1.1612, bounds = (0.0, 10.0)), :Cᵟu => (name = "Ratio of mixing length to grid spacing", latex = L"C^{\delta}u",
          default = 0.5, bounds = (0.0, 3.0)), :Cᵟc => (name = "Ratio of mixing length to grid spacing", latex = L"C^{\delta}c",
          default = 0.5, bounds = (0.0, 3.0)), :Cᵟe => (name = "Ratio of mixing length to grid spacing", latex = L"C^{\delta}e",
          default = 0.5, bounds = (0.0, 3.0)), :Cᵂu★ => (name = "Mixing length parameter", latex = L"C^W_{u\star}",
          default = 3.6188, bounds = (0.0, 20.0)), :CᵂwΔ => (name = "Mixing length parameter", latex = L"C^Ww\Delta",
          default = 1.3052, bounds = (0.0, 5.0)), :CᴷRiʷ => (name = "Stability function parameter", latex = L"C^KRi^w",
          default = 0.7213, bounds = (0.0, 5.0)), :CᴷRiᶜ => (name = "Stability function parameter", latex = L"C^KRi^c",
          default = 0.7588, bounds = (-1.5, 4.0)), :Cᴷu⁻ => (name = "Velocity diffusivity LB", latex = L"C^Ku^-",
          default = 0.1513, bounds = (0.0, 2.0)), :Cᴷuʳ => (name = "Velocity diffusivity (UB-LB)/LB", latex = L"C^Ku^r",
          default = 3.8493, bounds = (0.0, 50.0)), :Cᴷc⁻ => (name = "Velocity diffusivity LB", latex = L"C^Kc^-",
          default = 0.3977, bounds = (0.0, 5.0)), :Cᴷcʳ => (name = "Velocity diffusivity (UB-LB)/LB", latex = L"C^Kc^r",
          default = 3.4601, bounds = (0.0, 50.0)), :Cᴷe⁻ => (name = "Velocity diffusivity LB", latex = L"C^Ke^-",
          default = 0.1334, bounds = (0.0, 3.0)), :Cᴷeʳ => (name = "Velocity diffusivity (UB-LB)/LB", latex = L"C^Ke^r",
          default = 8.1806, bounds = (0.0, 50.0)), :Cᴬu => (name = "Convective adjustment velocity parameter", latex = L"C^A_U",
          default = 0.0057, bounds = (0.0, 0.2)), :Cᴬc => (name = "Convective adjustment tracer parameter", latex = L"C^A_C",
          default = 0.6706, bounds = (0.0, 2.0)), :Cᴬe => (name = "Convective adjustment TKE parameter", latex = L"C^A_E",
          default = 0.2717, bounds = (0.0, 2.0)),
)

bounds(name) = parameter_guide[name].bounds
default(name) = parameter_guide[name].default

function named_tuple_map(names, f)
     names = Tuple(names)
     return NamedTuple{names}(f.(names))
end

"""
    ParameterSet()

Parameter set containing the names `names` of parameters, and a 
NamedTuple `settings` mapping names of "background" parameters 
to their fixed values to be maintained throughout the calibration.
"""
struct ParameterSet{N,S}
        names :: N
     settings :: S
end

"""
    ParameterSet(names::Set; nullify = Set())

Construct a `ParameterSet` containing all of the information necessary 
to build a closure with the specified default parameters and settings,
given a Set `names` of the parameter names to be tuned, and a Set `nullify`
of parameters to be set to zero.
"""
function ParameterSet(names::Set; nullify = Set())
     zero_set = named_tuple_map(nullify, name -> 0.0)
     bkgd_set = named_tuple_map(keys(parameter_guide), name -> default(name))
     settings = merge(bkgd_set, zero_set) # order matters: `zero_set` overrides `bkgd_set`
     return ParameterSet(Tuple(names), settings)
end

names(ps::ParameterSet) = ps.names

###
### Define some convenient parameter sets based on the present CATKE formulation
###

required_params = Set([:Cᵟu, :Cᵟc, :Cᵟe, :Cᴰ, :Cᴸᵇ, :Cᵂu★, :CᵂwΔ, :Cᴷu⁻, :Cᴷc⁻, :Cᴷe⁻])
ri_depen_params = Set([:CᴷRiʷ, :CᴷRiᶜ, :Cᴷuʳ, :Cᴷcʳ, :Cᴷeʳ])
conv_adj_params = Set([:Cᴬu, :Cᴬc, :Cᴬe])

CATKEParametersRiDependent = ParameterSet(union(required_params, ri_depen_params); nullify = conv_adj_params)
CATKEParametersRiIndependent = ParameterSet(union(required_params); nullify = union(conv_adj_params, ri_depen_params))
CATKEParametersRiDependentConvectiveAdjustment = ParameterSet(union(required_params, conv_adj_params, ri_depen_params))
CATKEParametersRiIndependentConvectiveAdjustment = ParameterSet(union(required_params, conv_adj_params); nullify = ri_depen_params)
