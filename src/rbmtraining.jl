function assert_enoughvaluesforepochs(vname::String, v::Vector, epochs::Int)
   if length(v) < epochs
      error("Not enough `$vname` (vector of length $(length(v))) " .*
         "for training epochs ($epochs)")
   end
end


function assertinitoptimizers(optimizer::AbstractOptimizer,
      optimizers::Vector{<:AbstractOptimizer}, bm::BM,
      learningrates::Vector{Float64}, sdlearningrates::Vector{Float64},
      epochs::Int
      ) where {BM<:AbstractBM}

   if isempty(optimizers)
      if optimizer === NoOptimizer()
         # default optimization algorithm
         optimizers = map(
               i -> defaultoptimizer(bm, learningrates[i], sdlearningrates[i]),
               1:epochs)
      else
         optimizers = fill(initialized(optimizer, bm), epochs)
      end
   else
      optimizers = map(opt -> initialized(opt, bm), optimizers)
      assert_enoughvaluesforepochs("optimizers", optimizers, epochs)
   end

   optimizers
end


function defaultoptimizer(rbm::R,
      learningrate::Float64, sdlearningrate::Float64) where {R<: AbstractRBM}

   loglikelihoodoptimizer(rbm; learningrate = learningrate,
         sdlearningrate = sdlearningrate)
end

function defaultoptimizer(dbm::MultimodalDBM, learningrate::Float64,
      sdlearningrate::Float64)

   StackedOptimizer(map(
         rbm -> defaultoptimizer(rbm, learningrate, sdlearningrate),
         dbm))
end


"""
    fitrbm(x; ...)
Fits an RBM model to the data set `x`, using Stochastic Gradient Descent (SGD)
with Contrastive Divergence (CD), and returns it.

# Optional keyword arguments (ordered by importance):
* `rbmtype`: the type of the RBM that is to be trained
   This must be a subtype of `AbstractRBM` and defaults to `BernoulliRBM`.
* `nhidden`: number of hidden units for the returned RBM
* `epochs`: number of training epochs
* `learningrate`/`learningrates`: The learning rate for the weights and biases
   can be specified as single value, used throughout all epochs, or as a vector
   of `learningrates` that contains a value for each epoch. Defaults to 0.005.
* `batchsize`: number of samples that are used for making one step in the
   stochastic gradient descent optimizer algorithm. Default is 1.
* `pcd`: indicating whether Persistent Contrastive Divergence (PCD) is to
   be used (true, default) or simple CD that initializes the Gibbs Chain with
   the training sample (false)
* `cdsteps`: number of Gibbs sampling steps for (persistent)
   contrastive divergence, defaults to 1
* `monitoring`: a function that is executed after each training epoch.
   It takes an RBM and the epoch as arguments.
* `upfactor`, `downfactor`: If this function is used for pretraining a part of
   a DBM, it is necessary to multiply the weights of the RBM with factors.
* `sdlearningrate`/`sdlearningrates`: learning rate(s) for the
   standard deviation if training a `GaussianBernoulliRBM` or
   `GaussianBernoulliRBM2`. Ignored for other types of RBMs.
   It usually must be much smaller than the learning rates for
   the weights. By default it is 0.0, which means that the standard deviation
   is not learned.
* `startrbm`: start training with the parameters of the given RBM.
   If this argument is specified, `nhidden` and `rbmtype` are ignored.
* `optimizer`/`optimizers`: an object of type `AbstractOptimizer` or a vector of
   them for each epoch. If specified, the optimization is performed as implemented
   by the given optimizer type. By default, the `LoglikelihoodOptimizer`
   with the `learningrate`/`learningrates` and `sdlearningrate`/`sdlearningrates`
   is used. For other types of optimizers, the learning rates must be specified
   in the `optimizer`. For more information on how to write your own optimizer,
   see `AbstractOptimizer`.
"""
function fitrbm(x::Matrix{Float64};
      nhidden::Int = size(x,2),
      inversedispersion::AbstractArray = ones(size(x,2)) .* 0.5,
      epochs::Int = 10,
      upfactor::Float64 = 1.0,
      downfactor::Float64 = 1.0,
      learningrate::Float64 = 0.005,
      learningrates::Vector{Float64} = fill(learningrate, epochs),
      pcd::Bool = true,
      cdsteps::Int = 1,
      batchsize::Int = 1,
      fisherscoring::Int = 1,
      dispersionquantile::Float64 = 0.5,
      estimatedispersion::String = "gene",
      lambda::Float64 = 100.0,
      zeroinflation::Bool = true,
      dropoutmid::Float64 = 0.5,
      dropoutshape::Float64 = 0.5,
      limit::Int = 10,
      epsilon::Float64 = 0.0001,
      rbmtype::DataType = BernoulliRBM,
      startrbm::AbstractRBM = NoRBM(),
      monitoring::Function = nomonitoring,

      # these arguments are only relevant for GaussianBernoulliRBMs:
      sdlearningrate::Float64 = 0.0,
      sdlearningrates::Vector{Float64} = fill(sdlearningrate, epochs),
      sdinitfactor::Float64 = 0.0,

      optimizer::AbstractOptimizer = NoOptimizer(),
      optimizers::Vector{<:AbstractOptimizer} = Vector{AbstractOptimizer}(),
      sampler::AbstractSampler = (cdsteps == 1 ? NoSampler() : GibbsSampler(cdsteps - 1)))

   if startrbm === NoRBM()
      rbm = initrbm(x, nhidden, inversedispersion, fill(dropoutmid, size(x,2)), fill(dropoutshape, size(x,2)), rbmtype)
   else
      rbm = deepcopy(startrbm)
      nhidden = nhiddennodes(startrbm)
   end

   assert_enoughvaluesforepochs("learningrates", learningrates, epochs)
   assert_enoughvaluesforepochs("sdlearningrates", sdlearningrates, epochs)

   if sdinitfactor > 0 &&
         (rbmtype == GaussianBernoulliRBM || rbmtype == GaussianBernoulliRBM2)
      rbm.sd .*= sdinitfactor
   end

   optimizers = assertinitoptimizers(optimizer, optimizers, rbm,
         learningrates, sdlearningrates, epochs)

   if pcd
      chainstate = rand(batchsize, nhidden)
   else
      chainstate = Matrix{Float64}(undef, 0, 0)
   end

   # allocate space for trainrbm!
   nvisible = size(x, 2)
   nsamples = size(x,1)
   h = Matrix{Float64}(undef, batchsize, nhidden)
   hmodel = Matrix{Float64}(undef, batchsize, nhidden)
   vmodel = Matrix{Float64}(undef, batchsize, nvisible)
   
   for epoch = 1:epochs
      if epoch % 200 == 0
         println("Epoch = ", epoch)
      end
      # Train RBM on data set
      trainrbm!(rbm, x, cdsteps = cdsteps, chainstate = chainstate,
            upfactor = upfactor, downfactor = downfactor,
            learningrate = learningrates[epoch],
            sdlearningrate = sdlearningrates[epoch],
            fisherscoring = fisherscoring,
            zeroinflation = zeroinflation,
            optimizer = optimizers[epoch],
            sampler = sampler,
            batchsize = batchsize,
            h = h, hmodel = hmodel, vmodel = vmodel)

      # Evaluation of learning after each training epoch
      monitoring(rbm, epoch)

      # Estimate inverse dispersion using Fisher scoring algorithm
      nfisherscoring = fisherscoring
      if fisherscoring != 0 && epoch % 5 == 0
      
         if estimatedispersion == "gene"
            genewisedispersion = zeros(0)
            wts = zeros(0)
               for i in 1:size(x,2)
                  push!(genewisedispersion,mle_for_θ(x[:,i], estimatedmean[:,i], wts, rbm, lambda; maxIter=20))
               end

            rbm.inversedispersion .= genewisedispersion

         elseif estimatedispersion == "gene-cell"

            genewisedispersion = zeros(size(x))
            wts = zeros(0)
            
            for j in 1:size(x,1)
               for i in 1:size(x,2)
                  genewisedispersion[j,i] = BMs.mle_for_θ([x[j,i]], [estimatedmean[j,i]], wts, rbm, lambda; maxIter=20)
               end
            end

            global genewisedispersion = genewisedispersion
            rbm.inversedispersion .= genewisedispersion

         else
            genewisedispersion = zeros(0)
            wts = zeros(0)
            for i in 1:size(x,2)
               push!(genewisedispersion,mle_for_θ(x[:,i], estimatedmean[:,i], wts, rbm, lambda; maxIter=20))
            end

            geneorder = collect(1:size(genewisedispersion,1))
            genewisedispersion = DataFrame(hcat(genewisedispersion, geneorder))
            genewisedispersion = sort(genewisedispersion, :x1)

            lowestdispersion = Array{Int64,1}(genewisedispersion[findall(genewisedispersion[1] .> quantile(genewisedispersion[1],dispersionquantile)),2])

            globaldispersion = mle_for_θ_global(x[:,lowestdispersion], estimatedmean[:,lowestdispersion], wts, rbm, lambda; maxIter=20)
            rbm.inversedispersion .= globaldispersion
            if epoch % 30 == 0
               println("Dispersion = ", 1 ./ rbm.inversedispersion)
            end
         end

      end
   end

   rbm
end

function libsizenormalization!(x::Array{Float64,2})

   # Library size normalization (see Splatter Package Bioconductor [https://github.com/Oshlack/splatter/blob/382e22e6b22f32f2bb86a4152af369eda19aaba3/R/splat-estimate.R])
   libsize = mapslices(sum, x, dims = 2)
   libmedian = median(libsize)
   x .= (x ./ libsize) .* libmedian
   x
end

function libsizenormalization(x::Array{Float64,2})

   # Library size normalization (see Splatter Package Bioconductor [https://github.com/Oshlack/splatter/blob/382e22e6b22f32f2bb86a4152af369eda19aaba3/R/splat-estimate.R])
   libsize = mapslices(sum, x, dims = 2)
   libmedian = median(libsize)
   normalizedcounts = (x ./ libsize) .* libmedian
   normalizedcounts
end

function normalization!(x::Array{Float64,2})

   # Normalization (Eraslan et al. [https://doi.org/10.1038/s41467-018-07931-2] pp. 11-12)
   libsize = mapslices(sum, x, dims = 2)
   medianlibsize = median(libsize)
   sizefactors = libsize ./ medianlibsize
   sizefactors = Diagonal(sizefactors[:,1])

   x .= log.((inv(sizefactors) * x) .+ 1)
   x
end

function normalization(x::Array{Float64,2})

   # Normalization (Eraslan et al. [https://doi.org/10.1038/s41467-018-07931-2] pp. 11-12)
   libsize = mapslices(sum, x, dims = 2)
   medianlibsize = median(libsize)
   sizefactors = libsize ./ medianlibsize
   sizefactors = Diagonal(sizefactors[:,1])

   normalizedcounts = log.((inv(sizefactors) * x) .+ 1)
   normalizedcounts
end

function logisticfunction(x::Array{Float64,1}, x0::Array{Float64,1}, k::Array{Float64,1})
   1 ./ (1 .+ exp.(-k .* (x .- x0)))
end

function estimatedropout!(x::Matrix{Float64}, rbm::NegativeBinomialBernoulliRBM)
 
   normalizedcounts = x
  
   means = mapslices(mean, normalizedcounts, dims =1)
   logcounts = log1p.(means)
   logcounts = logcounts[1,:]
   zeroproportion = mapslices(sum, normalizedcounts .== 0, dims = 1) / size(normalizedcounts,1)
   zeroproportion = zeroproportion[1,:]

   dataset = DataFrame(hcat(zeroproportion,logcounts))
   rename!(dataset, :x1 => :zeroproportion, :x2 => :logcounts)

   tmp = fit(GeneralizedLinearModel, @formula(zeroproportion ~ logcounts), dataset, Binomial());
   rbm.dropoutmid .= coef(tmp)[1]
   rbm.dropoutshape .= coef(tmp)[2]
end

function sampledropout(x::Matrix{Float64}, rbm::NegativeBinomialBernoulliRBM)
   
   dropoutmask = Array{Float64,2}(undef,size(x))
   for i in 1:size(x,2)
      
      dropoutmask[:,i] = logisticfunction(x[:,i], rbm.dropoutmid, rbm.dropoutshape)
   end

   bernoulli!(dropoutmask)
   1 .- dropoutmask
end

function sampledropout(x::Matrix{Float64}, dbm::MultimodalDBM)

   @. model(x, p) = 1 / (1 + exp(-p[2] * (x - p[1])))
   dropoutmask = Array{Float64,2}(undef,size(x))
   for i in 1:size(x,2)
      
      dropoutmask[:,i] = bernoulli!(model(log1p.(x[:,i]),[dbm[1].dropoutmid[i], dbm[1].dropoutshape[i]]))
   end

   1 .- dropoutmask
end

function fittedlogistic(x::Matrix{Float64}, rbm::NegativeBinomialBernoulliRBM)

   @. model(x, p) = 1 / (1 + exp(-p[2] * (x - p[1])))
   logisticestimate = Array{Float64,2}(undef,size(x))
   for i in 1:size(x,2)
      
      logisticestimate[:,i] = model(log1p.(x[:,i]),[rbm.dropoutmid[i], rbm.dropoutshape[i]])
   end

   logisticestimate
end

"""
    initrbm(x, nhidden)
    initrbm(x, nhidden, rbmtype)
Creates a RBM with `nhidden` hidden units and initalizes its weights for
training on dataset `x`.
`rbmtype` can be a subtype of `AbstractRBM`, default is `BernoulliRBM`.
"""
function initrbm(x::Array{Float64,2}, nhidden::Int, 
   inversedispersion::AbstractArray, dropoutmid::Array{Float64,1}, dropoutshape::Array{Float64,1},
      rbmtype::DataType = BernoulliRBM)

   nsamples, nvisible = size(x)
   weights = randn(nvisible, nhidden)/sqrt(nvisible)
   hidbias = zeros(nhidden)

   if rbmtype == BernoulliRBM
      visbias = initvisiblebias(x)
      return BernoulliRBM(weights, visbias, hidbias)

   elseif rbmtype == GaussianBernoulliRBM
      visbias = vec(mean(x, dims = 1))
      sd = vec(std(x, dims = 1))
      GaussianBernoulliRBM(weights, visbias, hidbias, sd)

   elseif rbmtype == GaussianBernoulliRBM2
      visbias = vec(mean(x, dims = 1))
      sd = vec(std(x, dims = 1))
      weights .*= sd
      return GaussianBernoulliRBM2(weights, visbias, hidbias, sd)

   elseif rbmtype == BernoulliGaussianRBM
      visbias = initvisiblebias(x)
      return BernoulliGaussianRBM(weights, visbias, hidbias)

   elseif rbmtype == Binomial2BernoulliRBM
      visbias = initvisiblebias(x/2)
      return Binomial2BernoulliRBM(weights, visbias, hidbias)

   elseif rbmtype == NegativeBinomialBernoulliRBM
      weights .= -rand(Uniform(0.0005,0.002),nvisible,nhidden)
      inversedispersion = inversedispersion 
      visbias = initvisiblebiasnegativebinomial(x,inversedispersion)
      dropoutmid = [0.5]
      dropoutshape = [1.0]
      return NegativeBinomialBernoulliRBM(weights, visbias, hidbias, inversedispersion, dropoutmid, dropoutshape)

   else
      error(string("Datatype for RBM is unsupported: ", rbmtype))
   end
end


"""
    initvisiblebias(x)
Returns sensible initial values for the visible bias for training an RBM
on the data set `x`.
"""
function initvisiblebias(x::Array{Float64,2})
   nvisible = size(x,2)
   initbias = zeros(nvisible)
   for j=1:nvisible
      empprob = mean(x[:,j])
      if empprob > 0
         initbias[j] = log(empprob/(1-empprob))
      end
   end
   initbias
end

function initvisiblebiasnegativebinomial(x::Array{Float64,2}, inversedispersion::Array{Float64,1})

   nvisible = size(x,2)
   means = vec(mapslices(mean,x, dims = 1))
   
   theta = log.(means ./ (means .+ inversedispersion))
   theta
end

"""
    nomonitoring
Accepts a model and a number of epochs and returns nothing.
"""
function nomonitoring(bm, epoch)
end


"""
    randombatchmasks(nsamples, batchsize)
Returns BitArray-Sets for the sample indices when training on a dataset with
`nsamples` samples using minibatches of size `batchsize`.
"""
function randombatchmasks(nsamples::Int, batchsize::Int)
   if batchsize > nsamples
      error("Batchsize ($batchsize) exceeds number of samples ($nsamples).")
   end

   nfullbatches, nremainingbatch = divrem(nsamples, batchsize)
   batchsizes = fill(batchsize, nfullbatches)
   if nremainingbatch > 0
      push!(batchsizes, nremainingbatch)
   end
   randomsampleindices = randperm(nsamples)
   map(batchrange -> randomsampleindices[batchrange],
         ranges(batchsizes))
end


function sdupdateterm(gbrbm::GaussianBernoulliRBM,
         v::Matrix{Float64}, h::Matrix{Float64})
   vec(mean((v .- gbrbm.visbias').^2 ./ (gbrbm.sd' .^ 3) -
         h * gbrbm.weights' .* (v ./ (gbrbm.sd' .^ 2)), dims = 1))
end


function assert_enoughvaluesforepochs(vname::String, v::Vector, epochs::Int)
   if length(v) < epochs
      error("Not enough `$vname` (vector of length $(length(v))) " .*
         "for training epochs ($epochs)")
   end
end


function assertinitoptimizers(optimizer::AbstractOptimizer,
      optimizers::Vector{<:AbstractOptimizer}, bm::BM,
      learningrates::Vector{Float64}, sdlearningrates::Vector{Float64},
      epochs::Int
      ) where {BM<:AbstractBM}

   if isempty(optimizers)
      if optimizer === NoOptimizer()
         # default optimization algorithm
         optimizers = map(
               i -> defaultoptimizer(bm, learningrates[i], sdlearningrates[i]),
               1:epochs)
      else
         optimizers = fill(initialized(optimizer, bm), epochs)
      end
   else
      optimizers = map(opt -> initialized(opt, bm), optimizers)
      assert_enoughvaluesforepochs("optimizers", optimizers, epochs)
   end

   optimizers
end


function defaultoptimizer(rbm::R,
      learningrate::Float64, sdlearningrate::Float64) where {R<: AbstractRBM}

   loglikelihoodoptimizer(rbm; learningrate = learningrate,
         sdlearningrate = sdlearningrate)
end

function defaultoptimizer(dbm::MultimodalDBM, learningrate::Float64,
      sdlearningrate::Float64)

   StackedOptimizer(map(
         rbm -> defaultoptimizer(rbm, learningrate, sdlearningrate),
         dbm))
end

"""
    trainrbm!(rbm, x)
Trains the given `rbm` for one epoch using the data set `x`.
(See also function `fitrbm`.)

# Optional keyword arguments:
* `learningrate`, `cdsteps`, `sdlearningrate`, `upfactor`, `downfactor`,
   `optimizer`:
   See documentation of function `fitrbm`.
* `chainstate`: a matrix for holding the states of the RBM's hidden nodes. If
   it is specified, PCD is used.
"""
function trainrbm!(rbm::AbstractRBM, x::Array{Float64,2};
      chainstate::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
      upfactor::Float64 = 1.0,
      downfactor::Float64 = 1.0,
      learningrate::Float64 = 0.005,
      cdsteps::Int = 1,
      batchsize::Int = 1,
      sdlearningrate::Float64 = 0.0,
      fisherscoring::Int = 1,
      zeroinflation::Bool = true,
      limit::Int = 10,
      epsilon::Float64 = 0.0001,
      optimizer::AbstractOptimizer = LoglikelihoodOptimizer(rbm;
            learningrate = learningrate, sdlearningrate = sdlearningrate),
      sampler::AbstractSampler = (cdsteps == 1 ? NoSampler() : GibbsSampler(cdsteps - 1)),

      # write-only arguments for reusing allocated space:
      v::Matrix{Float64} = Matrix{Float64}(undef, batchsize, length(rbm.visbias)),
      h::Matrix{Float64} = Matrix{Float64}(undef, batchsize, length(rbm.hidbias)),
      hmodel::Matrix{Float64} = Matrix{Float64}(undef, batchsize, length(rbm.hidbias)),
      vmodel::Matrix{Float64} = Matrix{Float64}(undef, batchsize, length(rbm.visbias)))

   nsamples = size(x, 1)

   # perform PCD if a chain state is provided as parameter
   pcd = !isempty(chainstate)

   batchmasks = randombatchmasks(nsamples, batchsize)
   nbatchmasks = length(batchmasks)

   normalbatchsize = true

   # Zero-Inflation
   if zeroinflation && typeof(rbm) == NegativeBinomialBernoulliRBM
   
      estimatedropout!(x,rbm)
   end

   global estimatedmean = zeros(0,size(x, 2))
   # go through all samples or thorugh all batches of samples
   for batchindex in eachindex(batchmasks)
      batchmask = batchmasks[batchindex]

      normalbatchsize = (batchindex < nbatchmasks || nsamples % batchsize == 0)

      if normalbatchsize
         v .= view(x, batchmask, :)
      else
         v = x[batchmask, :]
         thisbatchsize = nsamples % batchsize
         h = Matrix{Float64}(undef,thisbatchsize, nhiddennodes(rbm))
         if !pcd
            vmodel = Matrix{Float64}(undef,size(v))
            hmodel = Matrix{Float64}(undef,size(h))
         end # in case of pcd, vmodel and hmodel are not downsized
      end

      # Calculate potential induced by visible nodes, used for update
      hiddenpotential!(h, rbm, v, upfactor)
      
      # In case of CD, start Gibbs chain with the hidden state induced by the
      # sample. In case of PCD, start Gibbs chain with
      # previous state of the Gibbs chain.
      if pcd
         hmodel = chainstate # note: state of chain will be visible by the caller
      else
         copyto!(hmodel, h)
      end

      samplehiddenpotential!(hmodel, rbm)
      
      # Additional sampling steps, may be customized
      sample!(vmodel, hmodel, sampler, rbm, upfactor, downfactor)
      
      # Do not sample in last step to avoid unnecessary sampling noise
      visiblepotential!(vmodel, rbm, hmodel, downfactor)
      
      estimatedmean = vcat(estimatedmean, vmodel)
      global estimatedmean = estimatedmean
      
      hiddenpotential!(hmodel, rbm, vmodel, upfactor)
      
      if !normalbatchsize && pcd
         # remove additional samples
         # that are unnecessary for computing the gradient
         vmodel = vmodel[1:thisbatchsize, :]
         hmodel = hmodel[1:thisbatchsize, :]
      end
      computegradient!(optimizer, v, vmodel, h, hmodel, rbm)
      updateparameters!(rbm, optimizer)     
   end

   rbm
end

function linkfunction(v::M, rbm::NegativeBinomialBernoulliRBM,
   ) where{M <: AbstractArray{Float64,1}}

   
   tmp = zeros(size(v))
   tmp .= log1p.(v ./ (rbm.inversedispersion .+ v))
   tmp
end

function linkfunction(v::M, rbm::NegativeBinomialBernoulliRBM,
   ) where{M <: AbstractArray{Float64,2}}

   tmp = zeros(size(v))
   tmp .= log1p.(v ./ (rbm.inversedispersion .+ v))
   tmp
end

function inverselinkfunction(v::M, rbm::NegativeBinomialBernoulliRBM,
   ) where{M <: AbstractArray{Float64,1}}

   tmp = zeros(size(v))
   tmp = (exp.(v) .* rbm.inversedispersion) ./ (1 .- exp.(v))
   tmp
end

function inverselinkfunction(v::M, rbm::NegativeBinomialBernoulliRBM,
   ) where{M <: AbstractArray{Float64,2}}

   v = (exp.(v) .* rbm.inversedispersion') ./ (1 .- exp.(v))
   v
end

function mu(rbm::NegativeBinomialBernoulliRBM)
   tmp = (exp.(rbm.visbias) .* rbm.inversedispersion) ./ (1 .- exp.(rbm.visbias))
   tmp
end

function mu(dbm::MultimodalDBM)
   tmp = (exp.(dbm[1].visbias) .* dbm[1].inversedispersion) ./ (1 .- exp.(dbm[1].visbias))
   tmp
end

function nbvariance(rbm::NegativeBinomialBernoulliRBM)
   nbmean = mu(rbm)
   nbvariance = nbmean .+ (nbmean.^2 ./ rbm.inversedispersion)

   nbvariance
end

function nbvariance(dbm::MultimodalDBM)
   nbmean = mu(dbm)
   nbvariance = nbmean .+ (nbmean.^2 ./ dbm[1].inversedispersion)

   nbvariance
end

function successprob(rbm::NegativeBinomialBernoulliRBM)
   tmp = mu(rbm) ./ (mu(rbm) .+ rbm.inversedispersion')
   tmp
end

function successprob(dbm::MultimodalDBM)
   tmp = dbm[1].inversedispersion' ./ (mu(dbm) .+ dbm[1].inversedispersion')   
   tmp
end

function negativebinomialparameters(rbm::NegativeBinomialBernoulliRBM)
   nbmean,successprobability = mu(rbm),successprob(rbm)
   tmp = hcat(nbmean,successprobability)
   estimtatedparameters = DataFrame(tmp)
   rename!(estimtatedparameters, :x1 => :Mu, :x2 => :SuccessProb)
   estimtatedparameters
end

function negativebinomialparameters(dbm::MultimodalDBM)
   nbmean,successprobability = mu(dbm),successprob(dbm)
   tmp = hcat(nbmean,successprobability)
   estimtatedparameters = DataFrame(tmp)
   rename!(estimtatedparameters, :x1 => :Mu, :x2 => :SuccessProb)
   estimtatedparameters
end

function mle_for_θ(y::AbstractVector, μ::AbstractVector, wts::AbstractVector, rbm::NegativeBinomialBernoulliRBM, lambda::Float64;
   maxIter=3, convTol=1.e-6) 

   function first_derivative(θ::Real)
      tmp(yi, μi) = -((yi+θ)/(μi+θ) + log(μi+θ) - 1 - log(θ) - digamma(θ+yi) + digamma(θ))
      unit_weights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
      sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
   end

   function second_derivative(θ::Real)
      tmp(yi, μi) = -(yi+θ)/(μi+θ)^2 + 2/(μi+θ) - 1/θ - trigamma(θ+yi) + trigamma(θ)
      unit_weights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
      sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
   end

   unit_weights = length(wts) == 0
   if unit_weights
      n = length(y)
      θ = n / sum((yi/μi - 1)^2 for (yi, μi) in zip(y, μ))
   else
      n = sum(wts)
      θ = n / sum(wti * (yi/μi - 1)^2 for (wti, yi, μi) in zip(wts, y, μ))
   end
   δ, converged = one(θ), false

   for t = 1:maxIter
      θ = max(θ,0.01)
   
      # Add regularization term lambda*(1/theta^2) to the loglikelihood
      # of Fisher scoring algorithm. In this case lambda = 10 and the
      # first derivative is lambda*2/theta^3. The second derivative is
      # lambda*6/theta^4.
      score = first_derivative(θ)
      fisher = second_derivative(θ)
      δ = (score + lambda*2/θ^3) / (fisher + lambda*6/θ^4)
      if abs(δ) <= convTol
         converged = true
      break
   end
   θ = θ + δ
   
   end
   converged
   θ
end

function mle_for_θ_global(y::Array{Float64,2}, μ::Array{Float64,2}, wts::AbstractVector, rbm::NegativeBinomialBernoulliRBM;
   maxIter=10, convTol=1.e-16) 

   function first_derivative(θ::Real)
      tmp(yi, μi) = (yi+θ)/(μi+θ) + log(μi+θ) - 1 - log(θ) - digamma(θ+yi) + digamma(θ)
      unit_weights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
      sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
   end

   function second_derivative(θ::Real)
      tmp(yi, μi) = -(yi+θ)/(μi+θ)^2 + 2/(μi+θ) - 1/θ - trigamma(θ+yi) + trigamma(θ)
      unit_weights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
      sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
   end

   unit_weights = length(wts) == 0
   if unit_weights
      n = length(y)
      θ = n / sum((yi/μi - 1)^2 for (yi, μi) in zip(y, μ))
   else
      n = sum(wts)
      θ = n / sum(wti * (yi/μi - 1)^2 for (wti, yi, μi) in zip(wts, y, μ))
   end
   δ, converged = one(θ), false

   for t = 1:maxIter
      θ = abs(θ)
      δ = first_derivative(θ) / second_derivative(θ)
      
      if abs(δ) <= convTol
         converged = true
      break
   end
   θ = θ - δ
   
   end
   converged
   θ
end

function mle_for_θ_start(y::AbstractVector, μ::AbstractVector, wts::AbstractVector, rbm::NegativeBinomialBernoulliRBM;
   maxIter=30, convTol=1.e-6)

   function first_derivative(θ::Real)
      tmp(yi, μi) = (yi+θ)/(μi+θ) + log(μi+θ) - 1 - log(θ) - digamma(θ+yi) + digamma(θ)
      unit_weights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
      sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
   end

   function second_derivative(θ::Real)
      tmp(yi, μi) = -(yi+θ)/(μi+θ)^2 + 2/(μi+θ) - 1/θ - trigamma(θ+yi) + trigamma(θ)
      unit_weights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
      sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
   end

   unit_weights = length(wts) == 0
   if unit_weights
      n = length(y)
      
      θ = n / sum((yi/μi - 1)^2 for (yi, μi) in zip(y, μ))
   else
      n = sum(wts)
      θ = n / sum(wti * (yi/μi - 1)^2 for (wti, yi, μi) in zip(wts, y, μ))
   end
   δ, converged = one(θ), false

   for t = 1:maxIter
      θ = abs(θ)
      δ = first_derivative(θ) / second_derivative(θ)
      if abs(δ) <= convTol
         converged = true
      break
   end
   θ = θ - δ
   end
   converged || throw(ConvergenceException(maxIter))
   θ
end


"""
    ConvergenceException(iters::Int, lastchange::Real=NaN, tol::Real=NaN)
The fitting procedure failed to converge in `iters` number of iterations,
i.e. the `lastchange` between the cost of the final and penultimate iteration was greater than
specified tolerance `tol`.
"""
struct ConvergenceException{T<:Real} <: Exception
    iters::Int
    lastchange::T
    tol::T
    function ConvergenceException{T}(iters, lastchange::T, tol::T) where T<:Real
        if tol > lastchange
            throw(ArgumentError("Change must be greater than tol."))
        else
            new(iters, lastchange, tol)
        end
    end
end

ConvergenceException(iters, lastchange::T=NaN, tol::T=NaN) where {T<:Real} =
    ConvergenceException{T}(iters, lastchange, tol)

function Base.showerror(io::IO, ce::ConvergenceException)
    print(io, "failure to converge after $(ce.iters) iterations.")
    if !isnan(ce.lastchange)
        print(io, " Last change ($(ce.lastchange)) was greater than tolerance ($(ce.tol)).")
    end
end


"""Standardizes a data matrix x """
function standardize!(x;standardize::Bool=true)
   xmean = mean(x,dims=1)
   xstd = std(x;dims=1)

   for j in 1:size(x)[2]
      for i in 1:size(x)[1]
         if standardize
            x[i,j] = (x[i,j] - xmean[j])/xstd[j]
         else
            x[i,j] = (x[i,j] - xmean[j])
         end
      end
   end
   nothing
end

""" Calculates univariate betas, given an explanatory matrix x and a response y """
function calcunibeta(x,y)
   n, p = size(x)
   unibeta = zeros(p)

   for i=1:p
      denom = 0.0
       for j=1:n
         unibeta[i] += x[j,i]*y[j]
         denom += x[j,i]*x[j,i]
      end
      unibeta[i] /= denom
   end
   unibeta
end

""" Calculates univariate betas, between a selected variable (candidate) in x and all other variables in x. Betas that have been calculated in advance are stored in the matrix covcache."""
function calcunibeta(x,candidate,covcache)
   n, p = size(x)
   unibeta = zeros(p-1)

   for j=1:(p-1)
      targetj = j < candidate ? j : j+1

      if isnan(covcache[candidate,targetj])
          unibeta[j] = sum(x[:,targetj].*x[:,candidate])
          denom = sum(x[:,targetj].*x[:,targetj])

         covcache[candidate,targetj] = covcache[targetj,candidate] = unibeta[j]
      else
          unibeta[j] = covcache[candidate,targetj]
          denom = sum(x[:,targetj].*x[:,targetj])
      end
      unibeta[j] /= denom 
   end
   unibeta
end


""" For each variable "candidate" in inmat, the variables of inmat are selected that best explain the state of "candidate", given stepno steps."""
function partboost(inmat;
                   stepno::Int=10,nu::Float64=0.1,csf::Float64=0.9)
   n, inp = size(inmat)
   p = inp - 1
   betamat = zeros(Float64,inp,inp)

   covcache = fill(NaN,inp,inp)

   indenom = var(inmat;dims=1).*(n-1)

   for candidate=1:inp

       y = view(inmat,:,candidate)
       x = view(inmat,:,[1:(candidate-1); (candidate+1):inp]) 
      denom = view(indenom,[1:(candidate-1); (candidate+1):inp])

      nuvec = fill(nu,p)
      penvec = fill(n*(1/nu-1),p)
     
      actualnom = calcunibeta(inmat,candidate,covcache)
      beta = view(betamat,candidate,[1:(candidate-1); (candidate+1):inp])

      actualupdate = 0.0
      actualsel = -1

      for step in 1:stepno
          if step > 1
            for j=1:p
               inj = j < candidate ? j : j+1
               insel = actualsel < candidate ? actualsel : actualsel+1

               if isnan(covcache[inj,insel])
                   covval = sum(x[:,j] .* x[:,actualsel])
                   covcache[inj,insel] = covcache[insel,inj] = covval
               else
                  covval = covcache[inj,insel]
               end
               actualnom[j] -= actualupdate*covval/denom[j] 
            end
         end

          actualsel = argmax(((actualnom.*denom) ./ (denom.+penvec)).^2)
         actualupdate = nuvec[actualsel]*actualnom[actualsel]
         beta[actualsel] += actualupdate
         nuvec[actualsel] = 1-(1-nuvec[actualsel])^csf
         penvec[actualsel] = n*(1/nuvec[actualsel]-1)

      end
   end

   betamat
end


""" Aggregates distances in distmat using aggfun."""
function clusterdist(distmat,indices1,indices2,aggfun=mean)
    aggfun(view(distmat,indices1,indices2))
end

""" Hierarchically clusters variables based on their pairwise dissimilarity stored in distmat. Minimum and maximum cluster size is given by minsize and maxsize respectively. By default, clustering is performed based on average linkage (aggfun=mean). Returns a dictionary of clusters, where each cluster is represented by a vector containing the cluster mebmbers."""
function clusterbeta(distmat,aggfun=mean;minsize::Int=2 ,maxsize::Int=size(distmat)[2])
   p = size(distmat)[2]
   maxdist = maximum(distmat) + 1.0 

   cldist = copy(distmat) 
   cldict = Dict{Int,Vector{Int}}()
   for i=1:p
      cldict[i] = [i]
   end
 
   global test = cldict
   while true
      cllen = map(key -> length(cldict[key]),collect(keys(cldict))) # added collect() function for julia 1.0.3
     
      global test = cllen
      if minimum(cllen) >= minsize
         break
      end
       
       _keys = sort(collect(keys(cldict)))
       source, target, actualmin = findmindist(cldist,_keys)
      
      if (length(cldict[source]) + length(cldict[target]) > maxsize &&
        
          length(cldict[source]) >= minsize &&
          length(cldict[target]) >= minsize)
      
          cldist[source,target] = cldist[target,source] = maxdist
         continue
      end

      append!(cldict[target],cldict[source]) 
      delete!(cldict,source) 
      for key in keys(cldict)
         if key != target 
            cldist[target,key] = cldist[key,target] =
               clusterdist(distmat,cldict[target],cldict[key],aggfun) 
         end
      end
   end

    cldict
end

"""Makes an asymmetric similarity matrix symmetric. The distance (D) between i and j or j and i is replaced by max(|D_ij|,|D_ji|). """
function calcbetawish(betamat)
   p = size(betamat)[2]
   wishmat = zeros(p,p)
   for i=1:(p-1)
      for j=(i+1):p
         wishmat[i,j] = wishmat[j,i] = max(abs(betamat[i,j]),abs(betamat[j,i]))
      end
   end
   wishmat
end

""" Identifies the variables besti and bestjj with the lowest distance curmin given in the distance matrix distmat."""
function findmindist(distmat,indices=1:size(distmat,2))
  
   curmin = -1
   besti = bestj = -1

   for i=1:(length(indices)-1)
      for j=(i+1):length(indices)
         if (i == 1 && j == 2) || distmat[indices[i],indices[j]] < curmin
            besti = indices[i]
            bestj = indices[j]
            curmin = distmat[indices[i],indices[j]]
         end
      end
   end

   besti, bestj, curmin
end

""" Identifies clusters of related SNPs using stagewise regression. SNP clusters are allowed to have a maximum of maxsize SNPs while at least are required to have a minimum of minsize SNPs. stepno indicates the number of steps in stagewise regression."""
function findpartition(mat;minsize::Int=5,maxsize::Int=20,stepno::Int=100)
    dat = deepcopy(mat)
    standardize!(dat,standardize=true)
    betamat = partboost(dat,stepno=stepno)
    betawish = calcbetawish(betamat)
    cldict = clusterbeta(-betawish,minsize=minsize,maxsize=maxsize)
    mvisibleindex = Array{Array{Int,1}}(undef,length(cldict))
    clkeys = collect(keys(cldict))
    for i=1:length(clkeys)
        mvisibleindex[i] = cldict[clkeys[i]]
    end
    mvisibleindex
end

""" Identifies adequate size of hidden units for each separate DBM"""
function partitionhiddens(partitions,nhiddens)
    nparts = length(partitions)
    p= length(cat(partitions..., dims=1))
    nhiddensmat = zeros(Int,nparts,length(nhiddens))
    for i=1:(nparts-1)
        nhiddensmat[i,:] = floor.(nhiddens ./ (p/length(partitions[i])))
    end
    nhiddensmat[nparts,:] = nhiddens .- vec(sum(nhiddensmat,dims=1))
    nhiddensmat
end

""" Given the partitions, identified by findpartition, a separate DBM is trained on each partition. The overall DBM architecture is given by nhiddens. The units in nhiddens are distributed across the partitions, such that each partition has at least p units in hidden layer 1 and at least a single unit in hidden layer 2."""
function trainpartdbm(mat;partitions::Array{Array{Int64,1},1}=[collect(1:size(mat)[2])],
                      nhiddens::Array{Int64,1}=[size(mat)[2],div(size(mat)[2],10)],
                      epochs::Int64=20,
                      nparticles::Int64=100,
                      learningrate::Float64=0.005)

    partDBMS = Vector{BasicDBM}()
    nhiddensmat = partitionhiddens(partitions,nhiddens)
    println(nhiddensmat)
     for i=1:length(partitions)
        partx = mat[:,partitions[i]]
        pdbm = fitdbm(partx,
                                        nhiddens=nhiddensmat[i,:],
                                        epochs =epochs,
                                        nparticles=nparticles,
                                        learningrate=learningrate
                                        )
        push!(partDBMS,pdbm)
    end
    jointdbm = joindbms(partDBMS,partitions)
    jointdbm,partDBMS
end

