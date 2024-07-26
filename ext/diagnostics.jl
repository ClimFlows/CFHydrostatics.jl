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
    gradmass,
    ugradp,
    Omega,
    conservative_variable,
    temperature,
    sound_speed,
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

pressure(model, mass) = Dynamics.hydrostatic_pressure!(void, model, mass)

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

function ugradp(model, uv, gradmass)
    massx, massy = gradmass.ucolat, gradmass.ulon
    ux, uy = uv.ucolat, uv.ulon
    ugradp = similar(massx)
    # ux, uy are physical velocities
    # we must apply a factor 1/radius to convert them to contravariant (=angular) velocities
    compute_ugradp(model.mgr, ugradp, model, ux, uy, massx, massy, inv(model.planet.radius))
    return ugradp
end

gradmass(model, state) = synthesis_spheroidal!(void, state.mass_spec[:,:,1], model.domain.layer)

@loops function compute_ugradp(_, ugradp, model, ux, uy, massx, massy, factor)
    # the computation expects ux, uy as contravariant components and massx, massy as covariant gradient
    # the scaling by radius^-2 turns the mass 2-form into a scalar (O-form)
    let (irange, jrange) = (axes(ugradp, 1), axes(ugradp, 2))
        radius, nz = model.planet.radius, size(ugradp, 3)
        half_invrad2 = radius^-2 / 2
        for j in jrange
            @vec for i in irange
                px = half_invrad2 * massx[i, j, nz]
                py = half_invrad2 * massy[i, j, nz]
                ugradp[i, j, nz] = factor*(ux[i, j, nz] * px + uy[i, j, nz] * py)
                for k = nz:-1:2
                    px += half_invrad2 * (massx[i, j, k] + massx[i, j, k-1])
                    py += half_invrad2 * (massy[i, j, k] + massy[i, j, k-1])
                    ugradp[i, j, k] = factor*(ux[i, j, k] * px + uy[i, j, k] * py)
                end
            end
        end
    end
end

function Omega(model, dmass, ugradp)
    dmass = dmass[:,:,:,1] # scalar, spatial
    Omega = similar(dmass)
    compute_Omega(model.mgr, Omega, model, dmass, ugradp)
    return Omega
end

function compute_Omega(_, Omega, model, dmass, ugradp)
    # dmass is a scalar (O-form)
    let (irange, jrange) = (axes(ugradp, 1), axes(ugradp, 2))
        radius, nz = model.planet.radius, size(ugradp, 3)
        for j in jrange
            @vec for i in irange
                dp = dmass[i, j, nz]/2
                Omega[i, j, nz] = dp + ugradp[i, j, nz]
                for k = nz:-1:2
                    dp += (dmass[i, j, k] + dmass[i, j, k-1])/2
                    Omega[i, j, k] = dp + ugradp[i, j, k]
#                    Omega[i, j, k] = dp
                end
            end
        end
    end
end

end # module Diagnostics
