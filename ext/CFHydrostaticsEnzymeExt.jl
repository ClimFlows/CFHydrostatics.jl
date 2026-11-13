module CFHydrostaticsEnzymeExt

using CFDomains.LazyExpressions: @lazy
using CFDomains.LazyOperators: setminus!
using Enzyme: Const, Duplicated
using Enzyme.EnzymeRules: AugmentedReturn, RevConfigWidth
import Enzyme.EnzymeRules: augmented_primal, reverse
using MutatingOrNot: similar!
import VoronoiSpheres.Operators as Ops
using CFHydrostatics.Voronoi.Dynamics: mass_budget!

function primal_mass_budget!(output, model, inputs)
    dmass_air_, dmass_consvar_, flux_air_, flux_consvar_ = output
    mass_air, consvar, ucov = inputs
    dmass_air = similar!(dmass_air_, mass_air)
    dmass_consvar = similar!(dmass_consvar_, consvar)
    flux_air = similar!(flux_air_, ucov)
    flux_consvar = similar!(flux_consvar_, ucov)

    vsphere = model.domain.layer
    metric = model.planet.radius^-2
    centered_flux! = Ops.CenteredFlux(vsphere)
    centered_tracer_flux! = Ops.CenteredTracerFlux(vsphere)
    divergence! = Ops.Divergence(vsphere, setminus!)
    div_centered_flux! = Ops.DivCenteredFlux(vsphere, setminus!)
    @lazy metric_mass(mass_air; metric) = metric * mass_air

    Ops.apply_internal!(flux_air, model.mgr, centered_flux!, metric_mass, ucov)
    Ops.apply_internal!(flux_consvar, model.mgr, centered_tracer_flux!, flux_air, consvar)
    Ops.apply_internal!(dmass_air, model.mgr, divergence!, flux_air)
    Ops.apply_internal!(dmass_consvar, model.mgr, div_centered_flux!, consvar, flux_air)

    return dmass_air, dmass_consvar, flux_air, flux_consvar
end

function augmented_primal(
    config::RevConfigWidth{1}, ::Const{typeof(mass_budget!)}, ::Type{Const{T}},
    output::Duplicated{<:NTuple{4,<:AbstractArray}}, model::Const,
    inputs::Duplicated{<:NTuple{3,<:AbstractArray}},
) where {T}
    result = primal_mass_budget!(output.val, model.val, inputs.val)
    return AugmentedReturn(result, nothing, nothing)
end

function augmented_primal(
    config::RevConfigWidth{1}, ::Const{typeof(mass_budget!)}, ::Type{Duplicated{T}},
    output::Duplicated{<:NTuple{4,<:AbstractArray}}, model::Const,
    inputs::Duplicated{<:NTuple{3,<:AbstractArray}},
) where {T}
    result = primal_mass_budget!(output.val, model.val, inputs.val)
    shadow = map(result) do array
        dx = similar(array)
        fill!(dx, 0)
        dx
    end
    return AugmentedReturn(result, shadow, shadow)
end

function reverse(
    config::RevConfigWidth{1}, ::Const{typeof(mass_budget!)}, ::Type{Const{T}}, tape,
    output::Duplicated{<:NTuple{4,<:AbstractArray}}, model::Const,
    inputs::Duplicated{<:NTuple{3,<:AbstractArray}},
) where {T}
    return reverse_mass_budget!(output, output.dval, model, inputs)
end

function reverse(
    config::RevConfigWidth{1}, ::Const{typeof(mass_budget!)}, ::Type{Duplicated{T}}, tape,
    output::Duplicated{<:NTuple{4,<:AbstractArray}}, model::Const,
    inputs::Duplicated{<:NTuple{3,<:AbstractArray}},
) where {T}
    output_bar = map(copy, output.dval)
    for (bar, return_bar) in zip(output_bar, tape)
        bar .+= return_bar
    end
    result = reverse_mass_budget!(output, output_bar, model, inputs)
    for i in 1:3
        fill!(output.dval[i], 0)
    end
    copyto!(output.dval[4], output_bar[4])
    return result
end

function reverse_mass_budget!(output, output_bar, model, inputs)
    mass_air, consvar, ucov = inputs.val
    ∂mass_air, ∂consvar, ∂ucov = inputs.dval
    ∂dmass_air, ∂dmass_consvar, ∂flux_air, ∂flux_consvar = output_bar

    mgr = model.val.mgr
    vsphere = model.val.domain.layer
    metric = model.val.planet.radius^-2

    ∂flux = similar(ucov)
    copyto!(∂flux, ∂flux_air)

    centered_tracer_flux! = Ops.CenteredTracerFlux(vsphere)
    Ops.apply_adj!(∂flux_consvar, mgr, centered_tracer_flux!, ∂flux, ∂consvar,
                   (output.val[3], consvar))

    divergence! = Ops.Divergence(vsphere, setminus!)
    div_centered_flux! = Ops.DivCenteredFlux(vsphere, setminus!)
    Ops.apply_adj!(∂dmass_air, mgr, divergence!, ∂flux, nothing)
    Ops.apply_adj!(∂dmass_consvar, mgr, div_centered_flux!, ∂consvar, ∂flux,
                   (consvar, output.val[3]))

    metric_mass = similar(mass_air)
    @. metric_mass = metric * mass_air
    ∂metric_mass = similar(mass_air)
    fill!(∂metric_mass, 0)
    centered_flux! = Ops.CenteredFlux(vsphere)
    Ops.apply_adj!(∂flux, mgr, centered_flux!, ∂metric_mass, ∂ucov,
                   (metric_mass, ucov))
    @. ∂mass_air += metric * ∂metric_mass

    return (nothing, nothing, nothing)
end

end
