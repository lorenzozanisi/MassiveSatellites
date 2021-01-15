from MCMC import *

gausspriors = {'gamma11':[0,0.25], 'M11':[0.5,0.3], 'N11':[-0.015, 0.02]}
Sampler = sampler(gausspriors, prior_type='gaussian')
S = Sampler.run_chain(n_walkers=24,n_steps=10000,reset=False, filename='MCMC.h5' )
pp = postprocess(S)
pp.plot_chain() 
pp.check_model()