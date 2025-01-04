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
using ManagedLoops: @with, @vec

using ..Dynamics

function diagnostics()
    return CookBook(;
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
                    masses,
                    # dX is the local time derivative of X, assuming a Lagrangian vertical coordinate
                    dmasses,
                    duv,
                    dulon,
                    dulat,
                    dpressure,
                    dgeopotential,
                    # intermediate computations
                    dstate,
                    ps_spec,
                    gradPhi_cov,
                    # mostly for debugging
                    dstate_all,
                    gradmass,
                    ugradps,
                    gradPhi,)
end

#=================== independent from vertical coordinate ==============#

function uv(model, state)
    (; ucolat, ulon) = synthesis_vector!(void, map(copy, state.uv_spec), model.domain.layer)
    invrad = model.planet.radius^-1
    return (ucolat=invrad * ucolat, ulon=invrad * ulon)
end

function surface_pressure(model, ps_spec)
    return synthesis_scalar!(void, ps_spec[:, 1], model.domain.layer) .+ model.vcoord.ptop
end

function pressure(model, masses)
    mass = masses.air
    p = similar(mass)
    ptop = model.vcoord.ptop # avoids capturing `model`
    @with model.mgr let (irange, jrange) = (axes(p, 1), axes(p, 2))
        nz = size(p, 3)
        for j in jrange
            @vec for i in irange
                p[i, j, nz] = ptop + mass[i, j, nz] / 2
                for k in nz:-1:2
                    p[i, j, k - 1] = p[i, j, k] + (mass[i, j, k] + mass[i, j, k - 1]) / 2
                end
            end
        end
    end
    return p
end

function geopotential(model, masses, pressure)
    Phi = similar(pressure, size(pressure) .+ (0, 0, 1))
    vol = model.gas(:p, :consvar).specific_volume
    p = pressure
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        for j in jrange
            @vec for i in irange
                Phi[i, j, 1] = model.Phis[i, j]
            end
            for k in axes(p, 3)
                @vec for i in irange
                    consvar_ijk = masses.consvar[i, j, k] / masses.air[i, j, k]
                    v = vol(p[i, j, k], consvar_ijk)
                    dPhi = masses.air[i, j, k] * v
                    Phi[i, j, k + 1] = Phi[i, j, k] + dPhi
                end
            end
        end
    end
    return Phi
end

conservative_variable(masses) = @. masses.consvar / masses.air

function temperature(model, pressure, conservative_variable)
    return model.gas(:p, :consvar).temperature.(pressure, conservative_variable)
end

function sound_speed(model, pressure, temperature)
    return model.gas(:p, :T).sound_speed.(pressure, temperature)
end

function Omega(model, uv, gradmass, dpressure)
    (ux, uy), (massx, massy) = uv, gradmass
    dp, p_dot = dpressure, similar(massx)
    radius = model.planet.radius # avoids capturing `model`
    @with model.mgr let (irange, jrange) = (axes(p_dot, 1), axes(p_dot, 2))
        nz = size(p_dot, 3)
        # apply factor 1/radius to convert physical velocities (ux,uy) to contravariant (=angular)
        factor = inv(radius)
        # the scaling by radius^-2 turns the mass 2-form into a scalar (O-form)
        half_invrad2 = radius^-2 / 2
        for j in jrange
            @vec for i in irange
                px = half_invrad2 * massx[i, j, nz]
                py = half_invrad2 * massy[i, j, nz]
                p_dot[i, j, nz] = dp[i, j, nz] + factor * (ux[i, j, nz] * px + uy[i, j, nz] * py)
                for k in (nz - 1):-1:1
                    px += half_invrad2 * (massx[i, j, k + 1] + massx[i, j, k])
                    py += half_invrad2 * (massy[i, j, k + 1] + massy[i, j, k])
                    p_dot[i, j, k] = dp[i, j, k] + factor * (ux[i, j, k] * px + uy[i, j, k] * py)
                end
            end
        end
    end
    return p_dot
end

function Phi_dot(model, uv, gradPhi_cov, dgeopotential)
    (ux, uy), (Phi_x, Phi_y) = uv, gradPhi_cov
    Phi_dot_k = similar(ux)
    dPhi = dgeopotential
    # apply factor 1/radius to convert physical velocities (ux,uy) to contravariant (=angular)
    factor = inv(model.planet.radius)
    @with model.mgr let (irange, jrange, krange) = axes(Phi_dot_k)
        for j in jrange, k in krange
            @vec for i in irange
                gradx = Phi_x[i, j, k] + Phi_x[i, j, k + 1]
                grady = Phi_y[i, j, k] + Phi_y[i, j, k + 1]
                Phi_dot_k[i, j, k] = (dPhi[i, j, k] + dPhi[i, j, k + 1]) / 2 +
                                     (factor / 2) *
                                     (ux[i, j, k] * gradx + uy[i, j, k] * grady)
            end
        end
    end
    return Phi_dot_k
end

#======================= depend on vertical coordinate ======================#

function masses(model, state)
    fac, sph = model.planet.radius^-2, model.domain.layer
    return (air=synthesis_scalar!(void, fac * state.mass_air_spec, sph),
            consvar=synthesis_scalar!(void, fac * state.mass_consvar_spec, sph))
end

dmasses(model, dstate) = masses(model, dstate)
duv(model, dstate) = uv(model, dstate)
dulon(duv) = duv.ulon
dulat(duv) = -duv.ucolat

function dpressure(model, dmasses)
    dmass = dmasses.air
    dp_mid = similar(dmasses.air)
    @with model.mgr let (irange, jrange) = (axes(dmass, 1), axes(dmass, 2))
        nz = size(dmass, 3)
        for j in jrange
            @vec for i in irange
                # top_down: dp_mid
                dp_top = zero(dp_mid[i, j, 1]) # could be SIMD
                for k in nz:-1:1
                    dp_bot = dp_top + dmass[i, j, k]
                    dp_mid[i, j, k] = (dp_top + dp_bot) / 2
                    dp_top = dp_bot
                end
            end
        end
    end
    return dp_mid
end

function dgeopotential(model, masses, dmasses, pressure, dpressure)
    # masses.X is a scalar (O-form in kg/m²)
    # dmasses.X is a scalar (O-form in kg/m²/s)
    dPhi_l = similar(pressure, size(pressure) .+ (0, 0, 1))
    dp_mid = dpressure
    volume = model.gas(:p, :consvar).volume_functions
    @with model.mgr let (irange, jrange) = (axes(masses.air, 1), axes(masses.air, 2))
        mass = masses.air
        nz = size(mass, 3)
        for j in jrange
            @vec for i in irange
                dPhi = zero(dPhi_l[i, j, 1]) # could be SIMD
                dPhi_l[i, j, 1] = dPhi
                for k in 1:nz
                    consvar = masses.consvar[i, j, k] / mass[i, j, k]
                    mass_dconsvar = dmasses.consvar[i, j, k] -
                                    consvar * dmasses.air[i, j, k]
                    v, dv_dp, dv_dconsvar = volume(pressure[i, j, k], consvar)
                    ddPhi = v * dmasses.air[i, j, k] +
                            dv_dconsvar * mass_dconsvar +
                            dv_dp * mass[i, j, k] * dp_mid[i, j, k]
                    dPhi += ddPhi
                    dPhi_l[i, j, k + 1] = dPhi
                end
            end
        end
    end
    return dPhi_l
end

#======================== intermediate computations =========================#

dstate(dstate_all) = dstate_all[1]
ps_spec(model, state) = (model.planet.radius^-2) * sum(state.mass_air_spec; dims=2)

function gradPhi_cov(model, geopotential)
    sph = model.domain.layer
    Phi_spec = analysis_scalar!(void, geopotential, sph)
    return synthesis_spheroidal!(void, Phi_spec, sph)
end

function gradmass(model, state)
    return synthesis_spheroidal!(void, state.mass_air_spec, model.domain.layer)
end

#==========================  mostly for debugging ===========================#

dstate_all(model, state) = Dynamics.tendencies!(void, void, model, state, 0.0)

function ugradps(model, uv, ps_spec)
    gradx, grady = synthesis_spheroidal!(void, ps_spec[:, 1], model.domain.layer)
    ux, uy = uv.ucolat, uv.ulon
    # apply factor 1/radius to convert physical velocities (ux,uy) to contravariant (=angular)
    return @. (1 / model.planet.radius) * (ux * gradx + uy * grady)
end

function gradPhi(model, gradPhi_cov)
    Phi_x, Phi_y = gradPhi_cov
    return @. sqrt(Phi_x^2 + Phi_y^2) / model.planet.radius
end

end # module Diagnostics
