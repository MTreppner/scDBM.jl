# This file implements methods from the paper
# "Boltzmann Encoded Adversarial Machines" of Fisher et al. (2018)


function gibbssample_gamma!(particles::Particles, gbrbm::GaussianBernoulliRBM2, nsteps::Int = 10;
      autocorcoeff::Float64 = 0.9, betasd::Float64 = 0.9)

   nsamples = size(particles[1], 1)
   vs = map(i -> view(particles[1], i, :), 1:nsamples)
   hs = map(i -> view(particles[2], i, :), 1:nsamples)

   for k = 1:nsamples
      v = vs[k]
      h = hs[k]
      beta = initgammaprocess(autocorcoeff, betasd)

      for i = 1:nsteps
         beta = samplegammaprocess(beta, autocorcoeff, betasd)
         gbrbm2 = GaussianBernoulliRBM2(gbrbm.weights, gbrbm.visbias, gbrbm.hidbias,
               copy(gbrbm.sd) / sqrt(beta))
         samplevisible!(v, gbrbm2, h)
         samplehidden!(h, gbrbm, v, 1/beta)
      end
   end
   particles
end

"""
Try Temperature Driven Sampling for NegativeBinomialBernoulliRBM
"""

#function gibbssample_gamma!(particles::Particles, rbm::NegativeBinomialBernoulliRBM, nsteps::Int = 10;
#   autocorcoeff::Float64 = 0.9, betasd::Float64 = 0.9)

#nsamples = size(particles[1], 1)
#vs = map(i -> view(particles[1], i, :), 1:nsamples)
#hs = map(i -> view(particles[2], i, :), 1:nsamples)

#for k = 1:nsamples
#   v = vs[k]
#   h = hs[k]
#   beta = initgammaprocess(autocorcoeff, betasd)
#
#   for i = 1:nsteps
#      beta = samplegammaprocess(beta, autocorcoeff, betasd)
#      samplevisible!(v, rbm, h)
#      samplehidden!(h, rbm, v, 1/beta)
#   end
#end
#particles
#end


function nearestneighbourcritic(xdata::M, xmodel::M, k::Int,
      distanceweighted::Bool = true) where {M <: AbstractArray{Float64,2}}

   ndatasamples = size(xdata, 1)
   if ndatasamples != size(xmodel, 1)
      error("Number of samples in `xdata` and `xmodel` must be equal.")
   end

   x = vcat(xdata, xmodel)

   nsamples = 2 * ndatasamples

   # Distancematrix is filled with the distances
   # between all combinations of samples in x.
   distancematrix = Matrix{Float64}(nsamples, nsamples)
   for i = 1:nsamples
      for j = (i+1):nsamples
         distancematrix[i, j] = distancematrix[j, i] = norm(x[j, :] - x[i, :])
      end
      distancematrix[i, i] = 0.0
   end

   ret = Vector{Float64}(ndatasamples)
   eps = 0.1
   if distanceweighted
      for i = 1:ndatasamples
         knearestindices = selectperm(distancematrix[i, :], (1:k)+1)
         nearestdatasamplesindices = knearestindices[knearestindices .<= ndatasamples]
         datasamplesdistances = distancematrix[i, nearestdatasamplesindices]
         ret[i] = 2.0 * ( sum(1.0 ./ (datasamplesdistances .+ eps)) /
               sum(1.0 ./ (distancematrix[i, knearestindices] .+ eps))) - 1
      end
   else
      for i = 1:ndatasamples
         knearestindices = selectperm(distancematrix[i, :], (1:k)+1)
         nnearestdatasamples = sum(knearestindices .<= ndatasamples)
         ret[i] = 2 * nnearestdatasamples / k - 1
      end
   end

   ret
end


"""
Update beta with an autoregressive Gamma process.
beta_0 ~ Gamma(nu,c/(1-phi)) = Gamma(1/var, var)
h_t ~ Possion( phi/c * h_{t-1})
beta_t ~ Gamma(nu + z_t, c)
Achieves a stationary distribution with mean 1 and variance var:
Gamma(nu, var) = Gamma(1/var, var)
"""
function initgammaprocess(autocorr_coeff::Float64,betasd::Float64)

   var = betasd.^2
   nu = 1 ./ var
   c = (1-autocorr_coeff).*var

   # beta_0 ~ Gamma(nu,c/(1-phi)) = Gamma(1/var, var)
   beta = Gamma(nu,c ./ (1-autocorr_coeff))
   beta_sample = rand(beta)
end

function samplegammaprocess(beta_sample::Float64, autocorr_coeff::Float64,
      beta_std::Float64)

   var = beta_std^2

   nu = 1/var
   c = (1-autocorr_coeff)*var

   # h_t ~ Possion( phi/c * h_{t-1})
   lambda = beta_sample*autocorr_coeff/c
   z = Poisson(lambda)
   z_sample = rand(z)

   # beta_t ~ Gamma(nu + z_t, c)
   a = nu + z_sample
   b = c
   beta = Gamma(a,b)
   beta_sample = rand(beta)

   beta_sample
end
