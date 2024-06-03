#=============================== Vertical remap ===========================#

function vertical_remap_LHPE!(backend, (mass, ucov), model, domain::Shell, (phi, B, U, qv, qe), (fluxq, flux_e, fluxu))
    # check vertical sizes
    @assert size(mass,1) == size(ucov,1)  "Size mismatch"
    @assert size(phi,1) == size(mass,1)+1 "Size mismatch"
    @assert size(mass,1) == size(B,1)     "Size mismatch"
    @assert size(mass,1) == size(U,1)     "Size mismatch"
    @assert size(mass,1) == size(qv,1)    "Size mismatch"
    @assert size(mass,1) == size(qe,1)    "Size mismatch"
    # check horizontal sizes
    @assert size(mass,2) == size(phi,2)  "Size mismatch"
    @assert size(mass,2) == size(B,2)    "Size mismatch"
    @assert size(ucov,2) == size(U,2)    "Size mismatch"
    @assert size(ucov,2) == size(qe,2)   "Size mismatch"

    (; vcoord, domain) = model
    limiter = Transport.minmod_simd

    barrier(backend) #, @__HERE__)

    # mass fluxes and new mass
    newmg = @view B[:,:,1]
    flux = phi
    mg = @view mass[:,:,1]
    remap_fluxes_ps!(backend, flux, newmg, mg, vcoord)

    # vertical transport of densities
    vanleer = Transport.VanLeerScheme(:density, limiter, 1, 2)
    q, dq = ( (@view B[:,:,i]) for i in 2:3 )
    for iq in 2:size(mass,3)
        mgq = @view mass[:,:,iq]
        Transport.concentrations!(backend, q, mgq, mg)
        Transport.slopes!(backend, vanleer, dq, q)
        zero_bottom_top!(backend, dq)
        Transport.fluxes!(backend, vanleer, fluxq, mg, flux, q, dq)
        zero_bottom_top!(backend, fluxq)
        Transport.FV_update!(backend, vanleer, mgq, mgq, fluxq)
    end

    barrier(backend) #, @__HERE__)

    # interpolate mass and vertical mass flux to edges
    mg_e = qe
    to_edges!(backend, mg_e, mg, domain.layer.edge_left_right)
    to_edges(backend, flux_e, flux, domain.layer.edge_left_right)

    # vertical transport of momentum
    ducov = U
    vanleer = Transport.VanLeerScheme(:scalar, limiter, 1, 2)
    Transport.slopes!(backend, vanleer, ducov, ucov)
    zero_bottom_top!(backend, ducov)
    Transport.fluxes!(backend, vanleer, fluxu, mg_e, flux_e, ucov, ducov)
    zero_bottom_top!(backend, fluxu)
    Transport.FV_update!(backend, vanleer, ucov, ucov, fluxu, flux_e, mg_e)

    barrier(backend) #, @__HERE__)

    update_mass!(backend, mg, newmg)

    return nothing
end

@loops function zero_bottom_top!(_, q)
    let range = axes(q,2)
        @vec for i in range
            q[1,i] = 0
            q[end, i] = 0
        end
    end
end

@loops function to_edges!(_, mg_e, mg, left_right)
    let (krange, ijrange) = axes(mg_e)
        for ij in ijrange
            left, right = left_right[1,ij], left_right[2,ij]
            @vec for k in krange
                mg_e[k,ij] = half(mg[k,left] + mg[k,right])
            end
        end
    end
end

@loops function update_mass!(_, mg, newmg)
    let range = eachindex(mg)
        @vec for i in range
            mg[i] = newmg[i]
        end
    end
end
