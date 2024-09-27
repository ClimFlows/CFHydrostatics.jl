module CFHydrostaticsAdaptExt

using CFHydrostatics: HPE
using Adapt: Adapt, adapt, @adapt_structure 

@adapt_structure HPE

#= Adapt.adapt_structure(to, model::HPE{Mgr}) where Mgr = HPE(
    adapt_mgr(to),
    map(
        adapt(to),
        (model.vcoord, model.planet, model.domain, model.gas, model.fcov, model.Phis),
    )...,
)


Adapt.adapt_structure(mgr::LoopManager, model::HPE) = HPE(
    mgr,
    model.vcoord |> mgr,
    model.planet |> mgr,
    model.domain |> mgr,
    model.gas |> mgr,
    model.fcov |> mgr,
    model.Phis |> mgr,
)
=#

end
