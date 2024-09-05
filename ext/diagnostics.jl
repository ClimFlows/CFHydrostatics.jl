module Diagnostics

using MutatingOrNot: void, Void
using CookBooks
using SHTnsSpheres:
    analysis_scalar!,
    synthesis_scalar!,
    analysis_vector!,
    synthesis_vector!,
    synthesis_spheroidal!,
    divergence!,
    curl!
using ManagedLoops: @loops, @vec

using ..Dynamics

diagnostics() = CookBook(;
    # independent from vertical coordinate
    uv,
    surface_pressure,
    pressure,
    geopotential,
    conservative_variable,
    temperature,
    sound_speed,
    Omega,
    Phi_dot,
    # depend on vertical coordinate
    mass,
    masses,
    dmass,
    duv,
    # intermediate computations
    dstate,
    ps_spec,
    vertical_velocities,
    ugradp,
    ugradPhi,
    # mostly for debugging
    dstate_all,
    gradmass,
    ugradps,
    gradPhi,
)

dstate(dstate_all) = dstate_all[1]
dstate_all(model, state) = Dynamics.tendencies!(void, void, model, state, 0.0)

mass(model, state) =
    model.planet.radius^-2 *
    synthesis_scalar!(void, copy(state.mass_spec), model.domain.layer)

function masses(model, state)
    fac, sph = model.planet.radius^-2, model.domain.layer
    return (
        air = synthesis_scalar!(void, fac*state.masses_spec.air, sph),
        consvar = synthesis_scalar!(void, fac*state.masses_spec.consvar, sph),
    )
end

dmass(model, dstate) = mass(model, dstate)

function uv(model, state)
    (; ucolat, ulon) = synthesis_vector!(void, map(copy, state.uv_spec), model.domain.layer)
    invrad = model.planet.radius^-1
    return (ucolat = invrad * ucolat, ulon = invrad * ulon)
end

duv(model, dstate) = uv(model, dstate)

ps_spec(model, state) = (model.planet.radius^-2) * sum(state.masses_spec.air; dims = 2)
# ps_spec(model, state) =
#    @views (model.planet.radius^-2) * sum(state.mass_spec[:, :, 1]; dims = 2)
surface_pressure(model, ps_spec) =
    synthesis_scalar!(void, ps_spec[:, 1], model.domain.layer) .+ model.vcoord.ptop

function conservative_variable(mass)
    mass_air = @view mass[:, :, :, 1]
    mass_consvar = @view mass[:, :, :, 2]
    return @. mass_consvar / mass_air
end

# conservative_variable(masses) = @. masses.consvar / masses.air

temperature(model, pressure, conservative_variable) =
    model.gas(:p, :consvar).temperature.(pressure, conservative_variable)

sound_speed(model, pressure, temperature) =
    model.gas(:p, :T).sound_speed.(pressure, temperature)

function pressure(model, mass)
    p = similar(mass[:, :, :, 1])
    compute_pressure!(model.mgr, p, model, mass)
    return p
end

function geopotential(model, mass, pressure)
    Phi = similar(pressure, size(pressure) .+ (0, 0, 1))
    compute_geopot!(nothing, Phi, model, mass, pressure)
    return Phi
end

function ugradp(model, uv, gradmass)
    massx, massy = gradmass.ucolat, gradmass.ulon
    ux, uy = uv.ucolat, uv.ulon
    ugradp = similar(massx)
    # apply factor 1/radius to convert physical velocities (ux,uy) to contravariant (=angular)
    compute_ugradp(model.mgr, ugradp, model, ux, uy, massx, massy, inv(model.planet.radius))
    return ugradp
end

function ugradps(model, uv, ps_spec)
    gradx, grady = synthesis_spheroidal!(void, ps_spec[:, 1], model.domain.layer)
    ux, uy = uv.ucolat, uv.ulon
    ugradp = similar(ux)
    # apply factor 1/radius to convert physical velocities (ux,uy) to contravariant (=angular)
    return @. (1 / model.planet.radius) * (ux * gradx + uy * grady)
end

function ugradPhi(model, uv, geopotential)
    ux, uy = uv
    ugradPhi = similar(ux)
    sph = model.domain.layer
    Phi_spec = analysis_scalar!(void, copy(geopotential), sph)
    Phi_x, Phi_y = synthesis_spheroidal!(void, Phi_spec, sph)
    # apply factor 1/radius to physical velocities (ux,uy) to contravariant (=angular)
    compute_ugradPhi(
        model.mgr,
        ugradPhi,
        model,
        ux,
        uy,
        Phi_x,
        Phi_y,
        inv(model.planet.radius),
    )
    return ugradPhi
end

function gradPhi(model, uv, geopotential)
    ux, uy = uv
    sph = model.domain.layer
    Phi_spec = analysis_scalar!(void, copy(geopotential), sph)
    Phi_x, Phi_y = synthesis_spheroidal!(void, Phi_spec, sph)
    gradPhi = @. sqrt(Phi_x^2 + Phi_y^2) / model.planet.radius
    return gradPhi
end

gradmass(model, state) =
    synthesis_spheroidal!(void, state.mass_spec[:, :, 1], model.domain.layer)

Omega(vertical_velocities) = vertical_velocities.Omega
Phi_dot(vertical_velocities) = vertical_velocities.Phi_dot

function vertical_velocities(model, mass, dmass, ugradp, ugradPhi, pressure)
    Omega, Phi_dot, dp = (similar(pressure) for _ = 1:3)
    compute_vertical_velocities(
        model.mgr, model,
        (Omega, Phi_dot, dp),
        (mass, dmass, ugradp, ugradPhi, pressure),
    )
    return (; Omega, Phi_dot, dp)
end

include("compute_diagnostics.jl")

end # module Diagnostics
