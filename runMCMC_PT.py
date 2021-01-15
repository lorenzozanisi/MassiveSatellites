from MCMC_temp import *
flatpriors = {'gamma11':[-5,5], 'M11':[-5,5], 'N11':[-0.2,0.1]}
gausspriors = {'gamma11':[0,0.25], 'M11':[0.5,0.3], 'N11':[-0.015, 0.02]}
Sampler = sampler(gausspriors, prior_type='gaussian')
S = Sampler.run_chain(n_walkers=24,n_steps=1000,ntemps=10)
pp = postprocess(S)
pp.plot_chain() 
pp.check_model()
