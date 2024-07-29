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
    debug,
    dstate,
    dmass,
    duv,
    mass,
    uv,
    surface_pressure,
    pressure,
    geopotential,
    conservative_variable,
    temperature,
    sound_speed,
    gradmass,
    ugradp,
    ugradPhi,
    Omega,
    Omega2,
    Phi_dot,
    vertical_velocities,
)

dstate(model, state) = Dynamics.tendencies!(void, model, state, void, 0.0)
debug(model, state) = Dynamics.tendencies_all(void, model, state, void, 0.0)

mass(model, state) =
    model.planet.radius^-2 * synthesis_scalar!(void, state.mass_spec, model.domain.layer)

dmass(model, dstate) = mass(model, dstate)

function uv(model, state)
    (; ucolat, ulon) = synthesis_vector!(void, state.uv_spec, model.domain.layer)
    invrad = model.planet.radius^-1
    return (ucolat = invrad * ucolat, ulon = invrad * ulon)
end

duv(model, dstate) = uv(model, dstate)

function surface_pressure(model, state)
    radius = model.planet.radius
    ps_spec = @views (radius^-2) * sum(state.mass_spec[:, :, 1]; dims = 2)
    ps_spat = synthesis_scalar!(void, ps_spec[:, 1], model.domain.layer)
    return ps_spat .+ model.vcoord.ptop
end

function conservative_variable(mass)
    mass_air = @view mass[:, :, :, 1]
    mass_consvar = @view mass[:, :, :, 2]
    return @. mass_consvar / mass_air
end

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
    # apply factor 1/radius to physical velocities (ux,uy) to contravariant (=angular)
    compute_ugradp(model.mgr, ugradp, model, ux, uy, massx, massy, inv(model.planet.radius))
    return ugradp
end

function ugradPhi(model, uv, geopotential)
    ux, uy = uv
    ugradPhi = similar(ux)
    sph = model.domain.layer
    Phi_spec = analysis_scalar!(void, geopotential, sph)
    Phi_x, Phi_y = synthesis_spheroidal!(void, Phi_spec, sph)
    # apply factor 1/radius to physical velocities (ux,uy) to contravariant (=angular)
    compute_ugradPhi(model.mgr, ugradPhi, model, ux, uy, Phi_x, Phi_y, inv(model.planet.radius))
    return ugradPhi
end

gradmass(model, state) =
    synthesis_spheroidal!(void, state.mass_spec[:, :, 1], model.domain.layer)

function Omega(model, dmass, ugradp)
    dmass = dmass[:, :, :, 1] # scalar, spatial
    Omega = similar(dmass)
    compute_Omega(model.mgr, Omega, model, dmass, ugradp)
    return Omega
end

Omega2(vertical_velocities) = vertical_velocities[1]
Phi_dot(vertical_velocities) = vertical_velocities[2]
pressure_tendency(vertical_velocities) = vertical_velocities[3]

function vertical_velocities(model, mass, dmass, ugradp, ugradPhi, pressure)
    Omega, dp, Phi_dot = (similar(pressure) for _ = 1:3)
    compute_vertical_velocities(
        nothing,
        Omega,
        Phi_dot,
        dp,
        model,
        mass,
        dmass,
        ugradp,
        ugradPhi,
        pressure,
    )
    return Omega, Phi_dot, dp
end

include("compute_diagnostics.jl")

end # module Diagnostics
