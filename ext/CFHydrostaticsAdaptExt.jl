module CFHydrostaticsAdaptExt

using CFHydrostatics: HPE
using Adapt: Adapt, adapt, @adapt_structure 

@adapt_structure HPE

end
