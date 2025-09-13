module Dynamics

using CFPlanets: lonlat_from_cov
using ManagedLoops: @with, @vec, @unroll
using MutatingOrNot: Void, void, similar!
using CFDomains: Stencils, transpose!
using CFHydrostatics: debug_flags

include("ext/zero_array.jl") # belongs to CFDomains ?

#=========================== API for fully explicit time scheme =======================#

function tendencies_HV!(dstate, scratch, model, state, t)
    (; mass_air, mass_consvar, ucov) = state
    dmass_air, dmass_consvar, ducov = (dstate.mass_air, dstate.mass_consvar, dstate.ucov)

    # flux-form mass budgets
    (; flux_air, flux_consvar, consvar) = scratch.mass_budget
    consvar = consvar!(consvar, model.mgr, mass_air, mass_consvar)
    dmass_air, dmass_consvar, flux_air, flux_consvar = mass_budget!(
        (dmass_air, dmass_consvar, flux_air, flux_consvar),
        model,
        (mass_air, consvar, ucov),
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

#=========================== API for IMEX time scheme =======================#

function tendencies_HV!(slow, fast, tmp, model, state, t, tau)
    (; mass_air, mass_consvar, ucov) = state

    consvar = consvar!(tmp.mass_budget.consvar, model.mgr, mass_air, mass_consvar)

    # fast tendencies: 0, 0, -( ∇(Φ+h-θπ)+θ∇π )

    #   switch to [ij,k] to make recurrences faster on GPU
    (; consvar_HV, mass_air_HV, pressure_HV, Phi_HV) = tmp.HV
    consvar_HV = transpose!(consvar_HV, model.mgr, consvar)
    mass_air_HV = transpose!(mass_air_HV, model.mgr, mass_air)
    Phi_HV, pressure_HV = hydrostatic_balance_HV!(Phi_HV, pressure_HV, model, mass_air_HV, consvar_HV)

    #   switch to [k,ij] to apply horizontal stencils
    (; pressure, Phi, fast_B, exner) = tmp.fast
    pressure = transpose!(pressure, model.mgr, pressure_HV)
    Phi = transpose!(Phi, model.mgr, Phi_HV)

    #   update ucov with fast tendency
    fast_ucov, fast_B, exner = fast_ucov!(fast.ucov, fast_B, exner, model, ucov, Phi, pressure, consvar)
    new_ucov = @. tmp.fast.new_ucov = ucov + tau*fast_ucov

    # slow tendencies: -∇⋅U, -∇⋅θU, -(ζ×u + ∇u²/2)
    #   flux-form mass budgets
    (; flux_air, flux_consvar) = tmp.mass_budget
    dmass_air, dmass_consvar, flux_air, flux_consvar = mass_budget!(
        (slow.mass_air, slow.mass_consvar, flux_air, flux_consvar, consvar),
        model,
        (mass_air, consvar, new_ucov),
    )
    #   Potential vorticity
    (; PV_e, PV_v, KE) = tmp.slow
    PV_e, PV_v = potential_vorticity!(PV_e, PV_v, model, new_ucov, mass_air)
    #   curl-form momentum budget
    slow_ucov, KE = slow_curl_form!(slow.ucov, KE, model, consvar, PV_e, flux_air, new_ucov)

    # wrap up
    zero_mass = zero_array(dmass_air)
    fast = (mass_air=zero_mass, mass_consvar=zero_mass, ucov=fast_ucov)
    slow = (mass_air=dmass_air, mass_consvar=dmass_consvar, ucov=slow_ucov)

    tmp = (
        HV = (; consvar_HV, mass_air_HV, pressure_HV, Phi_HV),
        fast = (; pressure, Phi, new_ucov, fast_B, exner),
        mass_budget = (; flux_air, flux_consvar, consvar),
        slow = (; PV_e, PV_v, KE),
    )

    return slow, fast, tmp
end

#=========================== shared kernels =======================#

# mass budget
#   U_air = mass_air * u  (contravariant)
#   U_consvar = (mass_consvar/mass_air)*U
#   d(mass_air)/dt = -div(U_air)
#   d(mass_consvar)/dt = -div(U_consvar)

function consvar!(consvar_, mgr, mass_air, mass_consvar)
    consvar = similar!(consvar_, mass_air)
    @with mgr, let (krange, ijrange) = axes(consvar)
        #=@inbounds=# for ij in ijrange
            @vec for k in krange
                consvar[k, ij] = mass_consvar[k, ij] * inv(mass_air[k, ij])
            end
        end
    end
    return consvar
end

function mass_budget!(
    (dmass_air_, dmass_consvar_, flux_air_, flux_consvar_),
    model,
    (mass_air, consvar, ucov),
)
    dmass_air = similar!(dmass_air_, mass_air)
    dmass_consvar = similar!(dmass_consvar_, consvar)
    flux_air = similar!(flux_air_, ucov)
    flux_consvar = similar!(flux_consvar_, ucov)

    vsphere, metric = model.domain.layer, model.planet.radius^-2

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

    return dmass_air, dmass_consvar, flux_air, flux_consvar
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

#=========================== explicit-scheme kernels =======================#

# Bernoulli function
#    B = geopotentail + kinetic energy + (h-consvar*exner)
function Bernoulli!(B_, exner_, model, ucov, consvar, p, Phi)
    B = similar!(B_, consvar)
    exner = similar!(exner_, consvar)

    half_metric = (model.planet.radius^-2) / 2
    Exner = model.gas(:p, :consvar).exner_functions
    degree = model.domain.layer.primal_deg
    vsphere = Stencils.dot_product(model.domain.layer) # extract only relevant fields

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

#=========================== IMEX-scheme kernels =======================#

function fast_ucov!(ducov_, B_, exner_, model, ucov, Phi, p, consvar)
    B = similar!(B_, consvar)
    exner = similar!(exner_, consvar)
    Exner = model.gas(:p, :consvar).exner_functions
    @with model.mgr,
    let (krange, cells) = axes(B)
        #=@inbounds=# for cell in cells
            @vec for k in krange
                consvar_cell = consvar[k, cell]
                h, v, exner_cell = Exner(p[k, cell], consvar_cell)
                exner[k, cell] = exner_cell
                B[k, cell] = Phi[k, cell] + (h - consvar_cell * exner_cell)
            end
        end
    end

    ducov = similar!(ducov_, ucov)
    vsphere = model.domain.layer
    @with model.mgr,
    let (krange, edges) = axes(ducov)
        #=@inbounds=# for edge in edges
            grad = Stencils.gradient(vsphere, edge) # covariant gradient
            avg = Stencils.average_ie(vsphere, edge) # centered average from cells to edges
            @vec for k in krange
                ducov[k, edge] = - (grad(B, k) + avg(consvar, k) * grad(exner, k))
            end
        end
    end

    return ducov, B, exner
end

function slow_curl_form!(ducov_, B_, model, consvar, PV_e, flux_air, ucov)
    vsphere = model.domain.layer
    # vsphere = Stencils.dot_product(model.domain.layer) # extract only relevant fields

    half_metric = (model.planet.radius^-2) / 2
    degree = model.domain.layer.primal_deg

    B = similar!(B_, consvar) # kinetic energy
    @with model.mgr,
    let (krange, cells) = axes(B)
        #=@inbounds=# for cell in cells
            deg = degree[cell]
            # @assert deg in 5:7 "deg=$deg not in 5:7"
            @unroll deg in 5:7 begin
                dot_product = Stencils.dot_product(vsphere, cell, Val(deg))
                @vec for k in krange
                    B[k, cell] = half_metric * dot_product(ucov, ucov, k)
                end
            end
        end
    end

    ducov = similar!(ducov_, flux_air)
    @with model.mgr,
    let (krange, edges) = axes(ducov)
        #=@inbounds=# for edge in edges
            grad = Stencils.gradient(vsphere, edge) # covariant gradient
            avg = Stencils.average_ie(vsphere, edge) # centered average from cells to edges
            deg = vsphere.trisk_deg[edge]
            # @assert deg in 9:11 "deg=$deg not in 9:11"
            @unroll deg in 9:11 begin
                trisk = Stencils.TRiSK(vsphere, edge, Val(deg))
                @vec for k in krange
                    ducov[k, edge] = trisk(flux_air, PV_e, k) - grad(B, k)
                end
            end
        end
    end

    return ducov, B
end

#= 

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


=#


end # module
