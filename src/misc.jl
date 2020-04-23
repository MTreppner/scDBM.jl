"""
    barsandstripes(nsamples, nvariables)
Generates a test data set. To see the structure in the data set, run e. g.
`reshape(barsandstripes(1, 16), 4, 4)` a few times.

Example from:
MacKay, D. (2003). Information Theory, Inference, and Learning Algorithms
"""
function barsandstripes(nsamples::Int, nvariables::Int)
   squareside = sqrt(nvariables)
   if squareside != floor(squareside)
      error("Number of variables must be a square number")
   end
   squareside = round(Int, squareside)
   x = zeros(nsamples, nvariables)
   for i in 1:nsamples
      sample = hcat([(rand() < 0.5 ? ones(squareside) : zeros(squareside))
            for j in 1:squareside]...)
      if rand() < 0.5
         sample = transpose(sample)
      end
      x[i,:] .= sample[:]
      fill!(sample, 0.0)
   end
   x
end


"""
    batchparallelized(f, n, op)
Distributes the work for executing the function `f` `n` times
on all the available workers and reduces the results with the operator `op`.
`f` is a function that gets a number (of tasks) to execute the tasks.

# Example:
    batchparallelized(n -> aislogimpweights(dbm; nparticles = n), 100, vcat)
"""
function batchparallelized(f::Function, n::Int, op::Function)
   batches = mostevenbatches(n)
   if length(batches) > 1
      return @sync @distributed (op) for batch in batches
         f(batch)
      end
   else
      return f(n)
   end
end


"""
    crossvalidation(x, monitoredfit; ...)
Performs k-fold cross-validation, given
* the data set `x` and
* `monitoredfit`: a function that fits and evaluates a model.
  As arguments it must accept:
  - a training data data set
  - a `DataDict` containing the evaluation data.


The return values of the calls to the `monitoredfit` function
are concatenated with `vcat`.
If the monitoredfit function returns `Monitor` objects,
`crossvalidation` returns a combined `Monitor` object that can be displayed
by creating a cross-validation plot via
`BoltzmannMachinesPlots.crossvalidationplot`.

# Optional named argument:
* `kfold`: specifies the `k` in "`k`-fold" (defaults to 10).

    crossvalidation(x, monitoredfit, pars; ...)
If additionaly a vector of parameters `pars` is given, `monitoredfit`
also expects an additional parameter from the parameter set.
"""
function crossvalidation(x::Matrix{Float64}, monitoredfit::Function, pars...;
      kfold::Int = 10)

   vcat(pmap(monitoredfit, crossvalidationargs(x, pars...)...)...)
end


"""
    crossvalidationargs(x, pars...; )

Returns a tuple of argument vectors containing the parameters for a function
such as the `monitoredfit` argument in `crossvalidation`.

Usage example:
    map(monitoredfit, crossvalidationargs(x)...)

# Optional named argument:
* `kfold`: see `crossvalidation`.
"""
function crossvalidationargs(x::Matrix{Float64}, pars ...; kfold::Int = 10)

   nsamples = size(x, 1)
   batchranges = BMs.ranges(BMs.mostevenbatches(nsamples, kfold))

   args_data = Vector{Tuple{Matrix{Float64}, BMs.DataDict}}(undef, kfold)
   for i in eachindex(batchranges)
      rng = batchranges[i]
      trainingdata = x[[!(j in rng) for j in 1:nsamples], :]
      evaluationdata = x[rng, :]
      datadict = BMs.DataDict(string(i) => evaluationdata)
      args_data[i] = (trainingdata, datadict)
   end

   # this implementation is inefficient and overly complicated

   if isempty(pars)
      args = map(arg -> (arg,), args_data)
   else
      args = (args_data, map(p -> collect(p), pars)...)
      args = vec(collect(Iterators.product(args...)))
   end

   # vector of tuples to tuples of vectors
   (
      map(arg -> arg[1][1], args), # x
      map(arg -> arg[1][2], args), # datadict
      [map(arg -> arg[i], args) for i in 2:length(args[1])]...
   )
end


"""
A function with no arguments doing nothing.
Usable as default argument for functions as arguments.W
"""
function emptyfunc
end


"""
    log1pexp(x)
Calculates log(1+exp(x)). For sufficiently large values of x, the approximation
log(1+exp(x)) ≈ x is used. This is useful to prevent overflow.
"""
function log1pexp(x::Float64)
   if x > 34
      return x
   else
      return log1p(exp(x))
   end
end

function logit(p::Float64)
   -log.(1.0 ./ p .- 1.0)
end

"""
    mostevenbatches(ntasks)
    mostevenbatches(ntasks, nbatches)
Splits a number of tasks `ntasks` into a number of batches `nbatches`.
The number of batches is by default `min(nworkers(), ntasks)`.
The returned result is a vector containing the numbers of tasks for each batch.
"""
function mostevenbatches(ntasks::Int, nbatches::Int = min(nworkers(), ntasks))

   minparticlesperbatch, nbatcheswithplus1 = divrem(ntasks, nbatches)
   batches = Vector{Int}(undef, nbatches)
   for i = 1:nbatches
      if i <= nbatcheswithplus1
         batches[i] = minparticlesperbatch + 1
      else
         batches[i] = minparticlesperbatch
      end
   end
   batches
end


"""
    curvebundles(...)
Generates an example dataset that can be visualized as bundles of trend curves
with added noise. Additional binary columns with labels may be added.

## Optional named arguments:
* `nbundles`: number of bundles
* `nperbundle`: number of sequences per bundle
* `nvariables`: number of variables in the sequences
* `noisesd`: standard deviation of the noise added on all sequences
* `addlabels`: add leading columns to the resulting dataset, specifying the
   membership to a bundle
* `pbreak`: probability that an intermediate point in a sequence is a
   breakpoint, defaults to 0.2.
* `breakval`: a function that expects no input and generates a single
  (random) value for a defining point of a piecewise linear sequence.
  Defaults to `rand`.

# Example:
To quickly grasp the idea, plot generated samples against the variable index:

    x = BMs.curvebundles(nvariables = 10, nbundles = 3,
                       nperbundle = 4, noisesd = 0.03,
                       addlabels = true)
    BoltzmannMachinesPlots.plotcurvebundles(x)
"""
function curvebundles(;
      nbundles::Int = 3,
      nperbundle::Int = 50,
      nvariables::Int = 10,
      noisesd::Float64 = 0.05,
      addlabels::Bool = false,
      pbreak::Float64 = 0.2,
      breakval::Function = rand)

   x = piecewiselinearsequences(nbundles, nvariables;
         pbreak = pbreak, breakval = breakval)
   x = repeat(x, nperbundle)
   nsamples = size(x, 1)
   x .+= noisesd * randn(nsamples, nvariables)

   if addlabels
      nlabelvars = Int(ceil(log(nbundles)/log(2)))
      labels = Matrix{Float64}(nbundles, nlabelvars)
      labels[1,:] .= 0.0
      for j = 2:nbundles
         labels[j,:] .= labels[j-1,:]
         next!(view(labels, j, :))
      end
      labels = repeat(labels, nperbundle)
      x = hcat(labels, x)
   end

   x = x[randperm(nsamples), :]
   x
end


"""
    piecewiselinearsequences(nsequences, nvariables; ...)
Generates a dataset consisting of samples with values that
are piecewise linear functions of the variable index.

Optional named arguments: `pbreak`, `breakval`,
see `piecewiselinearsequencebundles`.
"""
function piecewiselinearsequences(nsequences::Int, nvariables::Int;
      pbreak::Float64 = 0.2, breakval::Function = rand)

   inbetweenvariables = 2:(nvariables-1)
   x = Matrix{Float64}(undef, nsequences, nvariables)
   for i = 1:nsequences
      breakpointindexes = [1; randsubseq(inbetweenvariables, pbreak); nvariables]
      breakpointvalues = [breakval() for i = 1:length(breakpointindexes)]
      lastbreakpoint = 1
      x[i, breakpointindexes] .= breakpointvalues
      for breakpoint in breakpointindexes[2:end]
         valdiff = (x[i, breakpoint] - x[i, lastbreakpoint]) /
               (breakpoint - lastbreakpoint)
         for j = (lastbreakpoint + 1):(breakpoint - 1)
            x[i, j] = x[i, j - 1] + valdiff
         end
         lastbreakpoint = breakpoint
      end
   end
   x
end


"""
    splitdata(x, ratio)
Splits the data set `x` randomly in two data sets `x1` and `x2`, such that
the ratio `n2`/`n1` of the numbers of lines/samples in `x1` and `x2` is
approximately equal to the given `ratio`.
"""
function splitdata(x::Matrix{Float64}, ratio::Float64)
   nsamples = size(x,1)
   testidxs = randperm(nsamples)[1:(round(Int, nsamples*ratio))]
   xtest = x[testidxs,:]
   trainidxs = setdiff(1:nsamples, testidxs)
   xtraining = x[setdiff(1:nsamples, testidxs),:]
   xtraining, xtest, trainidxs, testidxs
end

"""
    DBindex(x, cl; p, q)
Computes the Davies-Bouldin Index for cluster validation.
"""
function DBindex(x::Array{Float64,2}, cl::Array{Int64,1}; p::Int = 1, q::Int = 2)

   n = size(cl,1)
   k = sort(unique(cl))

   centers = Array{Float64,2}(undef,(size(k,1),size(x,2)))
   for i = 1:size(k,1)
      for j = 1:size(x,2)
         centers[i,j] =  mean(x[cl .== k[i], j])
      end
   end

   # S is the within-cluster dispersion
   S = zeros(size(k,1))
   for i = 1:size(k,1)
      ind = (cl .== k[i])
      if sum(ind) > 1
         centerI = centers[i,:]
         centerI = reshape(repeat(centerI', sum(ind)), (sum(ind), size(x,2)))
         tmp = (x[ind,:] .- centerI).^2
         S[i] = mean(sqrt.(mapslices(sum, tmp, dims = 2)).^q).^(1/q)
      else
         S[i] = 0.0
      end
   end

   # R is the between-cluster similarity as a function of the two within-cluster dispersions
   # M is the distance between cluster centroids
   M = pairwise(Euclidean(), centers, dims=1)
   R = Array{Float64,2}(undef,(size(k,1),size(k,1)))
   r = zeros(size(k,1))
   for i = 1:size(k,1)
      for j = 1:size(k,1)
         R[i,j] = (S[i] + S[j]) / M[i,j]
      end
      r[i] = maximum(R[i,:][isfinite.(R[i,:])])
   end

   DB = mean(r[isfinite.(r)])
   DB
end

# Number of parameters for Negative Binomial DBM

function nparams(dbm::Array{BoltzmannMachines.AbstractXBernoulliRBM,1})

   params = zeros()
   for i in 1:size(dbm,1)
      if  typeof(dbm[i]) == BoltzmannMachines.NegativeBinomialBernoulliRBM
         tmp = (size(dbm[i].weights,1) * size(dbm[i].weights,2)) + size(dbm[i].visbias,1) + size(dbm[i].hidbias,1) + size(dbm[i].inversedispersion,1)
         params .+= tmp
      else
         tmp = (size(dbm[i].weights,1) * size(dbm[i].weights,2)) + size(dbm[i].visbias,1) + size(dbm[i].hidbias,1)
         params .+= tmp
      end
   end
   params[1]
end
