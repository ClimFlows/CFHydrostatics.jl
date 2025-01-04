@loops function compute_pressure!(_, p, model, mass)
    # mass is already a 0-form, in Pa
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        ptop, nz = model.vcoord.ptop, size(p, 3)
        for j in jrange
            @vec for i in irange
                p[i, j, nz] = ptop + mass[i, j, nz] / 2
                for k in nz:-1:2
                    p[i, j, k - 1] = p[i, j, k] + (mass[i, j, k] + mass[i, j, k - 1]) / 2
                end
            end
        end
    end
end

@loops function compute_geopot!(_, Phi, model, masses, p)
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        invrad2 = model.planet.radius^-2
        vol = model.gas(:p, :consvar).specific_volume
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
end

@loops function compute_ugradp(_, ugradp, model, ux, uy, massx, massy, factor)
    # factor==1 if  (ux, uy) is contravariant and (massx, massy) is covariant
    # the scaling by radius^-2 turns the mass 2-form into a scalar (O-form)
    let (irange, jrange) = (axes(ugradp, 1), axes(ugradp, 2))
        radius, nz = model.planet.radius, size(ugradp, 3)
        half_invrad2 = radius^-2 / 2
        for j in jrange
            @vec for i in irange
                px = half_invrad2 * massx[i, j, nz]
                py = half_invrad2 * massy[i, j, nz]
                ugradp[i, j, nz] = factor * (ux[i, j, nz] * px + uy[i, j, nz] * py)
                for k in (nz - 1):-1:1
                    px += half_invrad2 * (massx[i, j, k + 1] + massx[i, j, k])
                    py += half_invrad2 * (massy[i, j, k + 1] + massy[i, j, k])
                    ugradp[i, j, k] = factor * (ux[i, j, k] * px + uy[i, j, k] * py)
                end
            end
        end
    end
end

@loops function compute_ugradPhi(_, ugradPhi, model, ux, uy, Phi_x, Phi_y, factor)
    # factor==1 if  (ux, uy) is contravariant and (Phi_x, Phi_y) is covariant
    let (irange, jrange, krange) = axes(ugradPhi)
        for j in jrange, k in krange
            @vec for i in irange
                gradx = Phi_x[i, j, k] + Phi_x[i, j, k + 1]
                grady = Phi_y[i, j, k] + Phi_y[i, j, k + 1]
                ugradPhi[i, j, k] = 0.5 * factor *
                                    (ux[i, j, k] * gradx + uy[i, j, k] * grady)
            end
        end
    end
end

@loops function compute_vertical_velocities(_,
                                            model,
                                            (Omega, Phi_dot, dp_mid),
                                            (masses, dmasses, ugradp, ugradPhi, pressure))
    # dmass is a scalar (O-form in kg/m²/s)
    # consvar is a scalar (O-form in kg/m²/s)
    let (irange, jrange) = (axes(ugradp, 1), axes(ugradp, 2))
        volume = model.gas(:p, :consvar).volume_functions
        nz = size(ugradp, 3)
        for j in jrange
            @vec for i in irange
                # top_down: dp, Omega
                dp_top = zero(Omega[i, j, 1])
                for k in nz:-1:1
                    dp_bot = dp_top + dmasses.air[i, j, k]
                    dp_mid[i, j, k] = (dp_top + dp_bot) / 2
                    Omega[i, j, k] = dp_mid[i, j, k] + ugradp[i, j, k]
                    dp_top = dp_bot
                end
                # bottom-up: Phi_dot
                dPhi = zero(Phi_dot[i, j, 1])
                for k in 1:nz
                    consvar = masses.consvar[i, j, k] / masses.air[i, j, k]
                    mass_dconsvar = dmasses.consvar[i, j, k] -
                                    consvar * dmasses.air[i, j, k]
                    v, dv_dp, dv_dconsvar = volume(pressure[i, j, k], consvar)
                    ddPhi = v * dmasses.air[i, j, k] +
                            dv_dconsvar * mass_dconsvar +
                            dv_dp * masses.air[i, j, k] * dp_mid[i, j, k]
                    Phi_dot[i, j, k] = (dPhi + ddPhi / 2) + ugradPhi[i, j, k]
                    dPhi += ddPhi
                end
            end
        end
    end
end

@loops function compute_pressure_tendency!(_, model, dp_mid, mass)
    let (irange, jrange) = (axes(mass, 1), axes(mass, 2))
        nz = size(mass, 3)
        for j in jrange
            @vec for i in irange
                # top_down: dp_mid
                dp_top = zero(dp_mid[i, j, 1]) # could be SIMD
                for k in nz:-1:1
                    dp_bot = dp_top + dmasses.air[i, j, k]
                    dp_mid[i, j, k] = (dp_top + dp_bot) / 2
                    dp_top = dp_bot
                end
            end
        end
    end
end

@loops function compute_geopot_tendency!(_,
                                         model,
                                         dPhi_l,
                                         (masses, dmasses, pressure, dp_mid))
    let (irange, jrange) = (axes(masses.air, 1), axes(masses.air, 2))
        mass = masses.air
        nz = size(mass, 3)
        volume = model.gas(:p, :consvar).volume_functions
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
end

# to be removed

@loops function compute_ugradPhi_l(_, ugradPhi, model, (ux, uy), (Phi_x, Phi_y), factor)
    # factor==1 if (ux, uy) is contravariant and (Phi_x, Phi_y) is covariant
    let (irange, jrange) = (axes(ugradPhi, 1), axes(ugradPhi, 2))
        krange = axes(ux, 3)
        nz = length(krange)
        for j in jrange
            let l = 1, k = 1
                @vec for i in irange
                    u, v = ux[i, j, k], uy[i, j, k]
                    ugradPhi[i, j, l] = factor * (u * Phi_x[i, j, l] + v * Phi_y[i, j, l])
                end
            end
            for l in 2:nz
                @vec for i in irange
                    u = ux[i, j, l - 1] + ux[i, j, l]
                    v = uy[i, j, l - 1] + uy[i, j, l]
                    ugradPhi[i, j, l] = (factor / 2) *
                                        (u * Phi_x[i, j, l] + v * Phi_y[i, j, l])
                end
            end
            let l = nz + 1, k = nz
                @vec for i in irange
                    u, v = ux[i, j, k], uy[i, j, k]
                    ugradPhi[i, j, l] = factor * (u * Phi_x[i, j, l] + v * Phi_y[i, j, l])
                end
            end
        end
    end
end


@loops function compute_NH_momentum(_,
                                    model,
                                    (W, ux, uy),
                                    ((Phi_x, Phi_y), masses, dmasses, ugradPhi, pressure),
                                    (metric, jac, factor))
    # masses, dmasses are scalar (O-forms with units X/m² and X/m^2/s)
    # we multiply them by jac to obtain densities on the unit sphere (2-forms with units X and X/s)
    let (irange, jrange) = (axes(masses.air, 1), axes(masses.air, 2))
        mass = masses.air
        nz = size(mass, 3)
        dp_mid = W # use W[:,:,2:nz+1] as a buffer to compute dp/dt at full layers
        Phi_dot = W # use W dPhi/dt
        volume = model.gas(:p, :consvar).volume_functions
        for j in jrange
            @vec for i in irange
                # top_down: dp_mid
                dp_top = zero(dp_mid[i, j, 1])
                for k in nz:-1:1
                    dp_bot = dp_top + dmasses.air[i, j, k]
                    dp_mid[i, j, k + 1] = (dp_top + dp_bot) / 2
                    dp_top = dp_bot
                end
                # bottom-up: Phi_dot
                dPhi = zero(W[i, j, 1])
                Phi_dot[i, j, 1] = ugradPhi[i, j, 1]
                for k in 1:nz
                    consvar = masses.consvar[i, j, k] / mass[i, j, k]
                    mass_dconsvar = dmasses.consvar[i, j, k] -
                                    consvar * dmasses.air[i, j, k]
                    v, dv_dp, dv_dconsvar = volume(pressure[i, j, k], consvar)
                    ddPhi = v * dmasses.air[i, j, k] +
                            dv_dconsvar * mass_dconsvar +
                            dv_dp * mass[i, j, k] * dp_mid[i, j, k + 1]
                    dPhi += ddPhi
                    Phi_dot[i, j, k + 1] = dPhi + ugradPhi[i, j, k + 1]
                end

                # mass-weighted vertical momentum W
                W[i, j, 1] = (jac * factor / 2) * Phi_dot[i, j, 1] * mass[i, j, 1]
                for l in 2:nz
                    W[i, j, l] = (jac * factor / 2) * Phi_dot[i, j, l] *
                                 (mass[i, j, l - 1] + mass[i, j, l])
                end
                W[i, j, nz + 1] = (jac * factor / 2) * Phi_dot[i, j, nz + 1] *
                                  mass[i, j, nz]

                # horizontal NH momentum (covariant)
                for (ui, Phi_i) in ((ux, Phi_x), (uy, Phi_y))
                    ui[i, j, 1] = metric * ui[i, j, 1] +
                                  (Phi_i[i, j, 1] * 2W[i, j, 1] +
                                   Phi_i[i, j, 2] * W[i, j, 2]) / (2 * jac * mass[i, j, 1])
                    for k in 2:(nz - 1)
                        ui[i, j, k] = metric * ui[i, j, k] +
                                      (Phi_i[i, j, k] * W[i, j, k] +
                                       Phi_i[i, j, k + 1] * W[i, j, k + 1]) /
                                      (2 * jac * mass[i, j, k])
                    end
                    ui[i, j, nz] = metric * ui[i, j, nz] +
                                   (Phi_i[i, j, nz] * W[i, j, nz] +
                                    Phi_i[i, j, nz + 1] * 2W[i, j, nz + 1]) /
                                   (2 * jac * mass[i, j, nz])
                end
            end
        end
    end
end
