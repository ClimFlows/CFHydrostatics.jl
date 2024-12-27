var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = CFHydrostatics","category":"page"},{"location":"#CFHydrostatics","page":"Home","title":"CFHydrostatics","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for CFHydrostatics.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [CFHydrostatics]","category":"page"},{"location":"#CFHydrostatics.diagnostics-Tuple{CFHydrostatics.HPE}","page":"Home","title":"CFHydrostatics.diagnostics","text":"diags = diagnostics(model::HPE)\n\nReturn diags::CookBook containing recipes to compute standard diagnostics. This object is to be used as follows:\n\nsession = open(diags; model, state)\ntemp = session.temperature\n...\n\nwith state the current state, obtained for instance from initial_HPE.\n\n\n\n\n\n","category":"method"},{"location":"#CFHydrostatics.initial_HPE-Tuple{Any, Any}","page":"Home","title":"CFHydrostatics.initial_HPE","text":"state = initial_HPE(case, model)\n\nReturn an initial state for the hydrostatic model. case is expected to be a function with two methods:\n\nsurface_pressure, surface_geopotential = case(longitude, latitude)\ngeopotential, ulon, ulat = case(longitude, latitude, pressure)\n\nIf the hydrostatic model is for a binary fluid, the second method should return instead:\n\ngeopotential, ulon, ulat, q = case(longitude, latitude, pressure)\n\n\n\n\n\n","category":"method"}]
}