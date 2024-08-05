@loops function compute_pressure!(_, p, model, mass)
    # mass is already a 0-form, in Pa
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        ptop, nz = model.vcoord.ptop, size(p, 3)
        for j in jrange
            @vec for i in irange
                p[i, j, nz] = ptop + mass[i, j, nz, 1] / 2
                for k = nz:-1:2
                    p[i, j, k-1] = p[i, j, k] + (mass[i, j, k, 1] + mass[i, j, k-1, 1]) / 2
                end
            end
        end
    end
end

@loops function compute_geopot!(_, Phi, model, mass, p)
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        invrad2 = model.planet.radius^-2
        vol = model.gas(:p, :consvar).specific_volume
        for j in jrange
            @vec for i in irange
                Phi[i, j, 1] = model.Phis[i, j]
            end
            for k in axes(p, 3)
                @vec for i in irange
                    consvar_ijk = mass[i, j, k, 2] / mass[i, j, k, 1]
                    v = vol(p[i, j, k], consvar_ijk)
                    dPhi = mass[i, j, k, 1] * v
                    Phi[i, j, k+1] = Phi[i, j, k] + dPhi
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
                for k = nz-1:-1:1
                    px += half_invrad2 * (massx[i, j, k+1] + massx[i, j, k])
                    py += half_invrad2 * (massy[i, j, k+1] + massy[i, j, k])
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
                gradx = Phi_x[i, j, k] + Phi_x[i, j, k+1]
                grady = Phi_y[i, j, k] + Phi_y[i, j, k+1]
                #                gradx = Phi_x[i, j, 1] + Phi_x[i, j, 1]
                #                grady = Phi_y[i, j, 1] + Phi_y[i, j, 1]
                ugradPhi[i, j, k] =
                    0.5 * factor * (ux[i, j, k] * gradx + uy[i, j, k] * grady)
            end
        end
    end
end

@loops function compute_Omega(_, Omega, model, dmass, ugradp)
    # dmass is a scalar (O-form)
    let (irange, jrange) = (axes(ugradp, 1), axes(ugradp, 2))
        radius, nz = model.planet.radius, size(ugradp, 3)
        for j in jrange
            @vec for i in irange
                dp = dmass[i, j, nz] / 2
                Omega[i, j, nz] = dp + ugradp[i, j, nz]
                for k = nz-1:-1:1
                    dp += (dmass[i, j, k+1] + dmass[i, j, k]) / 2
                    Omega[i, j, k] = dp + ugradp[i, j, k]
                end
            end
        end
    end
end

@loops function compute_vertical_velocities(
    _, model,
    (Omega, Phi_dot, dp, dthickness),
    (mass, dmass, ugradp, ugradPhi, pressure),
)
    # dmass is a scalar (O-form in kg/m²/s)
    # consvar is a scalar (O-form in kg/m²/s)
    let (irange, jrange) = (axes(ugradp, 1), axes(ugradp, 2))
        volume = model.gas(:p, :consvar).volume_functions
        nz = size(ugradp, 3)
        for j in jrange
            @vec for i in irange
                # top_down: dp, Omega
                dp_top = zero(dp[i, j, 1])
                for k = nz:-1:1
                    dp_bot = dp_top + dmass[i, j, k, 1]
                    dp[i, j, k] = (dp_top + dp_bot) / 2
                    Omega[i, j, k] = dp[i, j, k] + ugradp[i, j, k]
                    dp_top = dp_bot
                end
                # bottom-up: Phi_dot
                dPhi = zero(Phi_dot[i, j, 1])
                for k = 1:nz
                    consvar = mass[i, j, k, 2] / mass[i, j, k, 1]
                    mass_dconsvar = dmass[i, j, k, 2] - consvar * dmass[i, j, k, 1]
                    v, dv_dp, dv_dconsvar = volume(pressure[i, j, k], consvar)
                    ddPhi =
                        v * dmass[i, j, k, 1] +
                        dv_dconsvar * mass_dconsvar +
                        dv_dp * mass[i, j, k, 1] * dp[i, j, k]
                    dthickness[i,j,k] = ddPhi
                    Phi_dot[i, j, k] = (dPhi + ddPhi / 2) + ugradPhi[i, j, k]
                    dPhi += ddPhi
                end
            end
        end
    end
end
