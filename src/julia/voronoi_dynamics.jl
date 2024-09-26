module Dynamics

using CFPlanets: lonlat_from_cov
using ManagedLoops: @with, @vec, @unroll
using MutatingOrNot: Void, void
using CFDomains: Stencils
using CFHydrostatics: debug_flags

similar!(x, _...) = x
similar!(::Void, y...) = similar(y...)

# transpose! can be specialized for specific managers
# for instance:
#   import CFHydrostatics.Voronoi.Dynamics: transpose!, Void
#   function transpose!(x, ::MultiThread, y)
#       @strided permutedims!(x, y, (2,1))
#       return x # otherwise returns a StridedView
#   end
#   transpose!(::Void, ::MultiThread, y) = permutedims(y, (2,1)) # for non-ambiguity

# ManagedLoops might be a better place to host this function
transpose!(::Void, mgr, y) = permutedims(y, (2, 1))
transpose!(x, mgr, y) = permutedims!(x, y, (2, 1))

function tendencies!(dstate, scratch, model, state, t)
    (; mass_air, mass_consvar, ucov) = state
    dmass_air, dmass_consvar, ducov = (dstate.mass_air, dstate.mass_consvar, dstate.ucov)

    # flux-form mass budgets
    (; flux_air, flux_consvar, consvar) = scratch.mass_budget
    dmass_air, dmass_consvar, flux_air, flux_consvar, consvar = mass_budget!(
        (dmass_air, dmass_consvar, flux_air, flux_consvar, consvar),
        model,
        (mass_air, mass_consvar, ucov),
    )

    # hydrostatic balance and Bernoulli function
    (; pressure, Phi, B, exner) = scratch.Bernoulli
    Phi, pressure = hydrostatic_balance!(Phi, pressure, model, mass_air, consvar)
    B, exner = Bernoulli!(B, exner, model, ucov, consvar, pressure, Phi)

    # Potential vorticity
    (; PV_e, PV_v) = scratch.PV
    PV_e, PV_v = potential_vorticity!(PV_e, PV_v, model, ucov, mass_air)

    # curl-form momentum budget
    ducov = curl_form!(ducov, model, PV_e, flux_air, B, consvar, exner)

    scratch = (
        mass_budget = (; flux_air, flux_consvar, consvar),
        Bernoulli = (; pressure, Phi, B, exner),
        PV = (; PV_e, PV_v),
    )

    return (mass_air = dmass_air, mass_consvar = dmass_consvar, ucov = ducov), scratch
end

function tendencies_HV!(dstate, scratch, model, state, t)
    (; mass_air, mass_consvar, ucov) = state
    dmass_air, dmass_consvar, ducov = (dstate.mass_air, dstate.mass_consvar, dstate.ucov)

    # flux-form mass budgets
    (; flux_air, flux_consvar, consvar) = scratch.mass_budget
    dmass_air, dmass_consvar, flux_air, flux_consvar, consvar = mass_budget!(
        (dmass_air, dmass_consvar, flux_air, flux_consvar, consvar),
        model,
        (mass_air, mass_consvar, ucov),
    )

    # hydrostatic balance and Bernoulli function
    # we switch to/from [ij,k] to make recurrences faster on GPU
    (; consvar_HV, mass_air_HV, pressure_HV, Phi_HV) = scratch.HV
    consvar_HV = transpose!(consvar_HV, model.mgr, consvar)
    mass_air_HV = transpose!(mass_air_HV, model.mgr, mass_air)
    Phi_HV, pressure_HV =
        hydrostatic_balance_HV!(Phi_HV, pressure_HV, model, mass_air_HV, consvar_HV)

    (; pressure, Phi, B, exner) = scratch.Bernoulli
    pressure = transpose!(pressure, model.mgr, pressure_HV)
    Phi = transpose!(Phi, model.mgr, Phi_HV)
    B, exner = Bernoulli!(B, exner, model, ucov, consvar, pressure, Phi)

    # Potential vorticity
    (; PV_e, PV_v) = scratch.PV
    PV_e, PV_v = potential_vorticity!(PV_e, PV_v, model, ucov, mass_air)

    # curl-form momentum budget
    ducov = curl_form!(ducov, model, PV_e, flux_air, B, consvar, exner)

    scratch = (
        mass_budget = (; flux_air, flux_consvar, consvar),
        HV = (; consvar_HV, mass_air_HV, pressure_HV, Phi_HV),
        Bernoulli = (; pressure, Phi, B, exner),
        PV = (; PV_e, PV_v),
    )

    return (mass_air = dmass_air, mass_consvar = dmass_consvar, ucov = ducov), scratch
end

# mass budget
#   U_air = mass_air * u  (contravariant)
#   U_consvar = (mass_consvar/mass_air)*U
#   d(mass_air)/dt = -div(U_air)
#   d(mass_consvar)/dt = -div(U_consvar)
function mass_budget!(
    (dmass_air_, dmass_consvar_, flux_air_, flux_consvar_, consvar_),
    model,
    (mass_air, mass_consvar, ucov),
)
    consvar = similar!(consvar_, mass_air)
    dmass_air = similar!(dmass_air_, mass_air)
    dmass_consvar = similar!(dmass_consvar_, mass_consvar)
    flux_air = similar!(flux_air_, ucov)
    flux_consvar = similar!(flux_consvar_, ucov)

    vsphere, metric = model.domain.layer, model.planet.radius^-2

    @with model.mgr, let (krange, ijrange) = axes(consvar)
        @inbounds for ij in ijrange
            @vec for k in krange
                consvar[k, ij] = mass_consvar[k, ij] * inv(mass_air[k, ij])
            end
        end
    end

    @with model.mgr, let (krange, ijrange) = axes(flux_air)
        @inbounds for ij in ijrange
            flux = Stencils.centered_flux(vsphere, ij)
            avg = Stencils.average_ie(vsphere, ij)
            @vec for k in krange
                flux_air[k, ij] = metric * flux(mass_air, ucov, k)
                flux_consvar[k, ij] = flux_air[k, ij] * avg(consvar, k)
            end
        end
    end

    @with model.mgr, let (krange, ijrange) = axes(dmass_air)
        @inbounds for ij in ijrange
            deg = vsphere.primal_deg[ij]
            # @assert deg in 5:7 "deg=$deg not in 5:7"
            @unroll deg in 5:7 begin
                dvg = Stencils.divergence(vsphere, ij, Val(deg))
                @vec for k in krange
                    dmass_air[k, ij] = -dvg(flux_air, k)
                    dmass_consvar[k, ij] = -dvg(flux_consvar, k)
                end
            end
        end
    end

    return dmass_air, dmass_consvar, flux_air, flux_consvar, consvar
end

function hydrostatic_balance!(Phi_, p_, model, mass_air, consvar)
    p = similar!(p_, mass_air)
    Phi = similar!(Phi_, p)

    ptop, nz = model.vcoord.ptop, size(p, 1)
    Phis = model.Phis # surface geopotential
    metric = model.planet.radius^-2
    vol = model.gas(:p, :consvar).specific_volume

    @with model.mgr, let ijrange = 1:size(p, 2)
        @inbounds for ij in ijrange
            p_top = ptop
            for k = nz:-1:1
                p_bot = p_top + metric * mass_air[k, ij]
                p[k, ij] = (p_bot + p_top) / 2
                p_top = p_bot
            end
            Phi_bot = Phis[ij]
            for k in axes(Phi, 1)
                v = vol(p[k, ij], consvar[k, ij])
                Phi_top = Phi_bot + metric * mass_air[k, ij] * v
                Phi[k, ij] = (Phi_top + Phi_bot) / 2
                Phi_bot = Phi_top
            end
        end
    end
    return Phi, p
end

function hydrostatic_balance_HV!(Phi_, p_, model, mass_air, consvar)
    p = similar!(p_, mass_air)
    Phi = similar!(Phi_, p)

    ptop, nz = model.vcoord.ptop, size(p, 2)
    Phis = model.Phis # surface geopotential
    metric = model.planet.radius^-2
    vol = model.gas(:p, :consvar).specific_volume

    @with model.mgr, let ijrange = axes(p, 1)
        @inbounds begin
            @vec for ij in ijrange
                p_top = ptop
                for k = nz:-1:1
                    p_bot = p_top + metric * mass_air[ij, k]
                    p[ij, k] = (p_bot + p_top) / 2
                    p_top = p_bot
                end
                Phi_bot = Phis[ij]
                for k in axes(Phi, 2)
                    v = vol(p[ij, k], consvar[ij, k])
                    Phi_top = Phi_bot + metric * mass_air[ij, k] * v
                    Phi[ij, k] = (Phi_top + Phi_bot) / 2
                    Phi_bot = Phi_top
                end
            end
        end
    end
    return Phi, p
end

# Bernoulli function
#    B = geopotentail + kinetic energy + (h-consvar*exner)
function Bernoulli!(B_, exner_, model, ucov, consvar, p, Phi)
    B = similar!(B_, consvar)
    exner = similar!(exner_, consvar)

    half_metric = (model.planet.radius^-2) / 2
    Exner = model.gas(:p, :consvar).exner_functions
    vsphere = model.domain.layer
    degree = vsphere.primal_deg

    @with model.mgr,
    let (krange, ijrange) = axes(B)
        fl = debug_flags()
        @inbounds for ij in ijrange
            deg = degree[ij]
            # @assert deg in 5:7 "deg=$deg not in 5:7"
            @unroll deg in 5:7 begin
                dot_product = Stencils.dot_product(vsphere, ij, Val(deg))
                @vec for k in krange
                    consvar_ijk = consvar[k, ij]
                    h, v, exner_ijk = Exner(p[k, ij], consvar_ijk)
                    ke = half_metric * dot_product(ucov, ucov, k)
                    exner[k, ij] = exner_ijk
                    B[k, ij] =
                        (fl.Phi) * Phi[k, ij] +
                        (fl.ke) * ke +
                        (h - consvar_ijk * exner_ijk)
                end
            end
        end
    end

    return B, exner
end

function potential_vorticity!(PV_e_, PV_v_, model, ucov, mass_air)
    fcov, vsphere = model.fcov, model.domain.layer
    nz = size(ucov, 1)
    PV_v = similar!(PV_v_, fcov, nz, length(fcov))
    PV_e = similar!(PV_e_, ucov)

    @with model.mgr, let (krange, ijrange) = axes(PV_v)
        @inbounds for ij in ijrange
            curl = Stencils.curl(vsphere, ij)
            avg = Stencils.average_iv(vsphere, ij) # area-weighted average from cells to vertices
            Av = vsphere.Av[ij]    # unit sphere cell area
            fcov_ij = fcov[ij]     # Coriolis * cell area Av
            @vec for k in krange
                zeta = curl(ucov, k)   # vorticity * Av
                mv = Av * avg(mass_air, k)  # mass * Av
                PV_v[k, ij] = (zeta + fcov_ij) * inv(mv)
            end
        end
    end
    @with model.mgr, let (krange, ijrange) = axes(PV_e)
        @inbounds for ij in ijrange
            avg = Stencils.average_ve(vsphere, ij) # centered averaging from vertices to edges
            @vec for k in krange
                PV_e[k, ij] = avg(PV_v, k)
            end
        end
    end
    return PV_e, PV_v
end

#== velocity tendency du = -q x U - grad B - consvar * grad exner ==#

function curl_form!(ducov_, model, PV_e, flux_air, B, consvar, exner)
    ducov = similar!(ducov_, flux_air)
    vsphere = model.domain.layer

    @with model.mgr,
    let (krange, ijrange) = axes(ducov)
        fl = debug_flags()
        @inbounds for ij in ijrange
            grad = Stencils.gradient(vsphere, ij) # covariant gradient
            avg = Stencils.average_ie(vsphere, ij) # centered average from cells to edges

            deg = vsphere.trisk_deg[ij]
            # @assert deg in 9:11 "deg=$deg not in 9:11"
            @unroll deg in 9:11 begin
                trisk = Stencils.TRiSK(vsphere, ij, Val(deg))
                @vec for k in krange
                    gradB =
                        (fl.gradB) * grad(B, k) +
                        (fl.CgradExner) * avg(consvar, k) * grad(exner, k)
                    ducov[k, ij] = (fl.qU) * trisk(flux_air, PV_e, k) - gradB
                end
            end
        end
    end
    return ducov
end

end # module
