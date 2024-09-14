module Diagnostics

using CookBooks: CookBook
using CFDomains: primal_lonlat_from_cov!
using CFPlanets: lonlat_from_cov

function diagnostics()
    return CookBook(; ulonlat_i, ulon, ulat)
end

function ulonlat_i(model, state)
    domain, planet, ucov = model.domain.layer, model.planet, state.ucov
    lon, lat = domain.lon_i, domain.lat_i
    coslon, sinlon = cos.(lon), sin.(lon)
    coslat, sinlat = cos.(lat), sin.(lat)
    ulon, ulat = similar(ucov), similar(ucov)
    args = domain.primal_deg, domain.primal_edge, domain.primal_perot_cov, coslon, sinlon, coslat, sinlat
    primal_lonlat_from_cov!(ulon, ulat, ucov, args...) do ij, ulon_ij, ulat_ij
        lonlat_from_cov(ulon_ij, ulat_ij, lon[ij], lat[ij], planet)
    end
    return ulon, ulat
end

ulon(ulonlat_i, interp) = interp(ulonlat_i[1])
ulat(ulonlat_i, interp) = interp(ulonlat_i[2])

end # module
