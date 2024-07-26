module Diagnostics

using MutatingOrNot: void, Void
using CookBooks
using SHTnsSpheres: analysis_scalar!, synthesis_scalar!, analysis_vector!, synthesis_vector!, divergence!, curl!

using ..Dynamics

diagnostics() = CookBook(;
    debug, dstate, dmass, duv,
    mass, uv, surface_pressure, pressure,
    gradmass, ugradp,
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

gradmass(model, state) = synthesis_spheroidal!(void, state.mass_spec, model.domain.layer)

function ugradp(model, uv, gradmass)
    massx, massy = gradmass.ucolat, gradmass.ulon
    ux, uy = uv.ucolat, uv.ulon
    ugradp = similar(massx)
    compute_gradp(model.mgr, ugradp, model, ux, uy, massx, massy)
    return ugradp
end

@loops function compute_ugradp(_, ugradp, model, ux, uy, massx, massy)
    let (irange, jrange) = (axes(p,1), axes(p,2))
        radius, nz = model.planet.radius, model.vcoord.ptop, size(p,3)
        half_invrad2 = radius^-2 /2
        for j in jrange
            @vec for i in irange
                px = half_invrad2*massx[i,j,nz,1]
                py = half_invrad2*massx[i,j,nz,1]
                ugradp[i,j,nz] = ux[i,j,nz]*px + uy[i,j,nz]*py
                for k in nz:-1:2
                    px += half_invrad2*(massx[i,j,k,1]+massx[i,j,k-1,1])
                    py += half_invrad2*(massy[i,j,k,1]+massy[i,j,k-1,1])
                    ugradp[i,j,k] = ux[i,j,k]*px + uy[i,j,k]*py
                end
            end
        end
    end
end

end # module Diagnostics
