module scDBM

using Distributions
using Distributed
using LinearAlgebra
using Random
using Statistics
using DataFrames
using SpecialFunctions
using LsqFit
using GLM
using Gadfly
using Loess
using Distances


export
   AbstractBM,
      aislogimpweights, aisprecision, aisstandarddeviation,
      empiricalloglikelihood, energy, exactloglikelihood,
      exactlogpartitionfunction, loglikelihood,
      logpartitionfunction, logpartitionfunctionzeroweights,
      logproblowerbound, reconstructionerror,
      sampleparticles, samples,
      AbstractRBM,
         BernoulliRBM,
         BernoulliGaussianRBM,
         Binomial2BernoulliRBM,
         GaussianBernoulliRBM,
         GaussianBernoulliRBM2,
         NegativeBinomialBernoulliRBM,
         PartitionedRBM,
         fitrbm, freeenergy, initrbm,
         joinrbms, joindbms,
         trainrbm!,
         samplehidden, samplehidden!,
         samplevisible, samplevisible!,
         hiddenpotential, hiddenpotential!,
         hiddeninput, hiddeninput!,
         visiblepotential, visiblepotential!,
         visibleinput, visibleinput!,
      MultimodalDBM,
         BasicDBM,
         AbstractTrainLayer, AbstractTrainLayers,
         TrainLayer, TrainPartitionedLayer,
         addlayer!, stackrbms,
         initparticles, gibbssample!,gibbssamplenegativebinomial!,
         meanfield, fitdbm, traindbm!,DBindex,
   Particle, Particles,
   AbstractOptimizer,
      LoglikelihoodOptimizer,
      initialized, computegradient!, updateparameters!,
      loglikelihoodoptimizer,
      beamoptimizer,
   Monitor, MonitoringItem, DataDict,
      monitorexactloglikelihood, monitorexactloglikelihood!,
      monitorfreeenergy, monitorfreeenergy!,
      monitorlogproblowerbound, monitorlogproblowerbound!,
      monitorloglikelihood, monitorloglikelihood!,
      monitorreconstructionerror, monitorreconstructionerror!,
      monitorweightsnorm, monitorweightsnorm!,
      propagateforward,
   crossvalidation,
   barsandstripes, logit, splitdata, plotevaluation, crossvalidationcurve, scatterhidden

include("bmtypes.jl")
include("gibbssampling.jl")
include("samplers.jl")
include("optimizers.jl")
include("rbmtraining.jl")
include("rbmstacking.jl")
include("dbmtraining.jl")
include("weightsjoining.jl")
include("evaluating.jl")
include("monitoring.jl")
include("beam.jl")
include("misc.jl")
include("NegativeBinomialRBMPlots.jl")
include("BoltzmannMachinesPlots.jl")


end # of module scDBM
