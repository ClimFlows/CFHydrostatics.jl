module Diagnostics

using MutatingOrNot: void, Void
using CookBooks
using SHTnsSpheres: analysis_scalar!, synthesis_scalar!, analysis_vector!, synthesis_vector!, divergence!, curl!

using ..Dynamics

diagnostics() = CookBook(;
    debug, dstate, dmass, duv,
    mass, uv, surface_pressure, pressure,
    conservative_variable, temperature, sound_speed)

dstate(model, state) = Dynamics.tendencies!(void, model, state, void, 0.0)
debug(model, state) = Dynamics.tendencies_all(void, model, state, void, 0.0)

mass(model, state) = model.planet.radius^-2 *
    synthesis_scalar!(void, state.mass_spec, model.domain.layer)

dmass(model, dstate) = mass(model, dstate)

function uv(model, state)
    (; ucolat, ulon) = synthesis_vector!(void, state.uv_spec, model.domain.layer)
    invrad = model.planet.radius^-1
    return (ucolat = invrad*ucolat, ulon=invrad*ulon)
end

duv(model, dstate) = uv(model, dstate)

pressure(model, mass) = Dynamics.hydrostatic_pressure!(void, model, mass)

function surface_pressure(model, state)
    radius = model.planet.radius
    ps_spec = @views (radius^-2)*sum(state.mass_spec[:,:,1]; dims=2)
    ps_spat = synthesis_scalar!(void, ps_spec[:,1], model.domain.layer)
    return ps_spat .+ model.vcoord.ptop
end

function conservative_variable(mass)
    mass_air = @view mass[:,:,:,1]
    mass_consvar = @view mass[:,:,:,2]
    return @. mass_consvar / mass_air
end

temperature(model, pressure, conservative_variable) =
    model.gas(:p, :consvar).temperature.(pressure, conservative_variable)

sound_speed(model, pressure, temperature) =
    model.gas(:p, :T).sound_speed.(pressure, temperature)

end # module Diagnostics
