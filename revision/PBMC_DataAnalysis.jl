using scDBM
using Pkg, LsqFit, Distributions, Random, Clustering, StatsBase, DelimitedFiles, 
Plots, Gadfly, GLM, LinearAlgebra, SpecialFunctions, DataFrames, CSV, Cairo, Fontconfig, 
Compose, TSne, MultivariateStats, UMAP, Distances, Loess, StatsPlots, PyCall, Conda, Distributed, Serialization
set_default_plot_size(30cm, 20cm)

#################################################################################################################
#################################################################################################################
############################################## Fit scDBM ########################################################
#################################################################################################################
#################################################################################################################
global Iter = ["384", "768", "1152", "1536", "1920", "2304"]
for t in Iter
   gridsearch = true
   if gridsearch
      addprocs(2)
      @everywhere using Random, CSV, DataFrames, scDBM
      @everywhere t = $t

      @everywhere function trainandeval(repititions, learningratedbm, epochsfirstlayer, dispinit, epochsdbm, nhiddennegbin, nhiddenbernoulli, lambdareg)
         println(repititions)
         Random.seed!(repititions)
         global i = 1
         while i <= 1
            @show i
            countmatrix = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_HVG.csv")
            genenames = countmatrix[:,1]
            # 1000 highly variable genes
            countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])');
            cells = parse(Int64, t)
            global seedset = repititions
            random_cells = rand(1:size(countmatrix,1),cells)

               @show sum(mapslices(sum, countmatrix[random_cells,:], dims=1) .> 1) == size(genenames,1)
               countmatrix_df = DataFrame(hcat(genenames,countmatrix[random_cells,:]'))

               data, datatest, trainidxs, testidxs = splitdata(countmatrix[random_cells,:], 0.40);
               # Convert indices to python compatible indices
               trainidxs = trainidxs .- 1
               testidxs = testidxs .- 1
               datadict = DataDict("Training data" => data, "Test data" => datatest);

               trainidxs = DataFrame(hcat(trainidxs, collect(1:size(trainidxs,1))));
               testidxs = DataFrame(hcat(testidxs, collect(1:size(testidxs,1))));

                  @show sum(mapslices(sum, data, dims=1) .!= 0) == size(genenames,1) & sum(mapslices(sum, datatest, dims=1) .!= 0) == size(genenames,1)

                  CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/cell_samples/PBMC/PBMC_cells",t,"_seed",seedset,".csv"), countmatrix_df, writeheader = true)
                  CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/cell_samples/PBMC/PBMC_cells",t,"_trainidxs_seed",seedset,".csv"), trainidxs, writeheader = true)
                  CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/cell_samples/PBMC/PBMC_cells",t,"_testidxs_seed",seedset,".csv"), testidxs, writeheader = true)

                  lr_rbm = [0.000001*ones(3500);0.00005*ones(1000);0.00001*ones(10000)];
                     global dbm = fitdbm(data, 
                           epochs = epochsdbm,
                           learningrate = learningratedbm,
                           batchsizepretraining = 32, 
                           pretraining = [
                              # Negative-Binomial RBM
                              TrainLayer(nhidden = nhiddennegbin, learningrates = lr_rbm, pcd = true, zeroinflation = false, epochs = epochsfirstlayer,
                              rbmtype = NegativeBinomialBernoulliRBM, inversedispersion = (ones(size(data,2)) .* dispinit), fisherscoring = 1, estimatedispersion = "gene", lambda = lambdareg)
                              # Bernoulli RBM
                              TrainLayer(nhidden = nhiddenbernoulli, learningrate = 0.0001, epochs = 500)]
                     )

                  global lplb = logproblowerbound(dbm, datatest)
                  global recerror = reconstructionerror(dbm[1], datatest)

                  @show repititions, learningratedbm, epochsfirstlayer, dispinit, epochsdbm, nhiddennegbin, nhiddenbernoulli, recerror, lplb
                  global i += 1
         end
         repititions, learningratedbm, epochsfirstlayer, dispinit, epochsdbm, nhiddennegbin, nhiddenbernoulli, lambdareg, dbm, recerror, lplb, seedset
      end
      epochsfirstlayerlist = [750]
      learningratedbmlist = [0.0001]
      dispinitlist = [1.0]
      epochsdbmlist = [200]
      nhiddennegbinlist = [12]
      nhiddenbernoullilist = [4]
      lambdareglist = [220.0]
      repititions = collect(4:10:300)

      @time trainingresult = vcat(pmap(params -> trainandeval(params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]),
            Iterators.product(repititions, learningratedbmlist, epochsfirstlayerlist, dispinitlist, epochsdbmlist, nhiddennegbinlist, nhiddenbernoullilist, lambdareglist))...);
   end
   serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/scDBM_",t,"Cells_PBMC"), trainingresult)
end

# Interrupt procs

interrupt()
rmprocs(workers())

#################################################################################################################
#################################################################################################################
############################## scDBM Upsample DBI, ARI, Cluster Proportions #####################################
#################################################################################################################
#################################################################################################################

global countmatrix = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_HVG.csv")
global countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])')
Random.seed!(321);
global umap_original = umap(countmatrix',2; n_neighbors=30, metric=CosineDist(), min_dist=0.3)'

Iter = ["384", "768", "1152", "1536", "1920", "2304"]
for t in Iter
   trainingresult = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/scDBM_",t,"Cells_PBMC"))

   dbi = DataFrame(Array{Float64,2}(undef, 1, 2))
   ari = DataFrame(Array{Float64,2}(undef, 1, 1))
   cluster_prop = DataFrame(Array{Int64,2}(undef, 1, 4))
   cd("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/cell_samples/PBMC/")
   filenames = readdir()[occursin.(string("cells",t,"_seed"), readdir())];

   for i = 1:size(filenames,1)
      
      if t == "384" || t == "768"
         seedset = filenames[i][19:end-4]
      else
         seedset = filenames[i][20:end-4]
      end

      for j = 1:size(trainingresult,1)
         if string(trainingresult[j][12]) == seedset
            global dbm = trainingresult[j][9]
         end
      end
      ngensamples = size(countmatrix,1);
      particles = initparticles(dbm, ngensamples);
      gibbssamplenegativebinomial!(countmatrix,particles, dbm, 50; zeroinflation=false)
      gen_data_dbm = particles[1]

      ########## Test Seurat Clustering ##########
      orig_labels = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_Seurat_Clustering.csv")
      orig_labels = Array{Int64,1}(orig_labels[:x])
      orig_labels = orig_labels .+ 1
      
      umap_original_plot = DataFrame(hcat(umap_original[:,1],umap_original[:,2], string.(orig_labels)))
      x = ["UMAP 1", "UMAP 2", "Cluster"]
      names!(umap_original_plot, Symbol.(x))

      # NB-DBM UMAP
      maplab(x1,x2,y) = map(arg -> y[findmin(sum((x1 .- x2[arg:arg,:]).^2,dims=2))[2][1]],1:size(x2,1))  
      genlab_dbm = maplab(countmatrix,gen_data_dbm,orig_labels)

      # Save mapped clusters
      cellnumber_tmp = collect(1:1:4182)
      generated_labels = DataFrame(hcat(genlab_dbm,cellnumber_tmp))
      CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/PBMC_Gen_Labels_",t,"_seed",seedset,".csv"), generated_labels, writeheader = true)

      Random.seed!(321);
      umap_dbm = umap(gen_data_dbm',2; n_neighbors=30, metric=CosineDist(), min_dist=0.3)'
      umap_dbm_plot = DataFrame(hcat(umap_dbm[:,1],umap_dbm[:,2], string.(genlab_dbm)))
      names!(umap_dbm_plot, Symbol.(x))

      # DBI
      orig_dbindex = BMs.DBindex(countmatrix, orig_labels);
      dbm_dbindex = BMs.DBindex(gen_data_dbm, genlab_dbm);
      combined_dbi = vec([orig_dbindex, dbm_dbindex])

      # ARI
      ari_index = Clustering.randindex(orig_labels,genlab_dbm)[2]

      push!(dbi, combined_dbi)
      push!(ari, ari_index)

      countmap_dbm = countmap(genlab_dbm)
      countmap_orig = countmap(orig_labels)
      for l in 1:size(unique(orig_labels),1)
         if in(l, keys(countmap_dbm)) == true
            nothing
         else
            countmap_dbm[l] = 0
         end
      end

      # Cluster Proportions
      tmp = DataFrame(hcat(collect(values(countmap_orig)),collect(values(countmap_dbm)), collect(keys(countmap_orig)), parse.(Int64, repeat(["$i"], size(unique(orig_labels),1)))))
      for k in 1:size(tmp,1)
         push!(cluster_prop, tmp[k,:])
      end
      println(i)
   end

   # DBI
   dbi = dbi[2:end,:]
   rename!(dbi, :x1 => :Original, :x2 => :NB_DBM);
   d = stack(dbi, 1:2)

   serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/dbi_",t,"cells_dbm_PBMC_Seurat"),d)

   # ARI
   ari = ari[2:end,:]
   rename!(ari, :x1 => :ARI);

   serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/ari_",t,"cells_dbm_PBMC_Seurat"),ari)

   # Cluster Proportions
   cluster_prop = cluster_prop[2:end,:]
   rename!(cluster_prop, :x1 => :Original, :x2 => :NB_DBM, :x3 => :Cluster, :x4 => :Rep);

   serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/cluster_prop_",t,"Cells_dbm_PBMC_Seurat"),cluster_prop)
end

#################################################################################################################
#################################################################################################################
############################ scVI Prior Upsample DBI, ARI, Cluster Proportions ##################################
#################################################################################################################
#################################################################################################################

global countmatrix = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_HVG.csv")
global countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])')
Random.seed!(321);
global umap_original = umap(countmatrix',2; n_neighbors=30, metric=CosineDist(), min_dist=0.3)';

Iter = ["384", "768", "1152", "1536", "1920", "2304"]
for t in Iter

dbi_scvi_prior = DataFrame(Array{Float64,2}(undef, 1, 2));
ari = DataFrame(Array{Float64,2}(undef, 1, 1));
cluster_prop_prior = DataFrame(Array{Int64,2}(undef, 1, 4));
for i = 1:30

py"""
n_epochs_all = None
save_path = 'C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/cell_samples/PBMC/'
show_plot = True
"""

py"""
import os
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, CsvDataset
from scvi.models import *
from scvi.inference import *
import torch
import urllib.request
"""

cd("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/cell_samples/PBMC/");
filenames = readdir()[occursin.(string("cells",t,"_seed"), readdir())];
tmp = filenames[i];
py"""
countmatrix = CsvDataset(str($tmp), save_path = save_path, new_n_genes = 500)
"""

countmatrix = py"CsvDataset(str($tmp), save_path = save_path, new_n_genes = 500)"

# Read train/test split indices
filenames_train = readdir()[occursin.(string("cells",t,"_train"), readdir())];
tmp = filenames_train[i];
trainidxs = CSV.read(tmp)[:,1];
py"""
trainidxs_py = $trainidxs
trainidxs_py = np.array(trainidxs_py)
"""

filenames_test = readdir()[occursin.(string("cells",t,"_train"), readdir())];
tmp = filenames_test[i];
testidxs = CSV.read(tmp)[:,1];
py"""
testidxs_py = $testidxs
testidxs_py = np.array(testidxs_py)
"""

n_cells = parse(Int64, t)
py"""
n_epochs = 100
lr = 1e-3
use_batches = False
use_cuda = True
n_cells = $n_cells
"""

if t == "384" || t == "768"
   seedset = filenames[i][19:end-4]
else
   seedset = filenames[i][20:end-4]
end

# Train the model and output model likelihood every 5 epochs
py"""
random.seed(int($seedset))
vae = VAE(countmatrix.nb_genes, n_batch=countmatrix.n_batches * use_batches, reconstruction_loss='nb', dispersion = 'gene')
trainer = UnsupervisedTrainer(vae,
   countmatrix,
   train_size=0.7,
   trainidxs=trainidxs_py,
   testidxs=testidxs_py,
   use_cuda=use_cuda,
   frequency=5,
)
"""

py"""
trainer.train(n_epochs=n_epochs, lr=lr)
"""

# Obtaining the posterior object and sample latent space from it
py"""
full = trainer.create_posterior(trainer.model, countmatrix, indices=np.arange(len(countmatrix)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()
"""

full =  py"trainer.create_posterior(trainer.model, countmatrix, indices=np.arange(len(countmatrix)))";
latent, batch_indices, labels = py"full.sequential(4182).get_latent()";
batch_indices = py"batch_indices.ravel()";

##########################################
# Generate Data from Prior with upsampling
##########################################
py"""
n_cells = 4182
"""
n_cells = 4182

gen_data_scvi = py"full.generate(batch_size=4182, n_samples=10, sample_prior=True, n_cells=n_cells)";
gen_data_scvi = Array{Float64,2}(gen_data_scvi[1][:,:,4]);

maplab(x1,x2,y) = map(arg -> y[findmin(sum((x1 .- x2[arg:arg,:]).^2,dims=2))[2][1]],1:size(x2,1))  

# Original data

# Read original data PBMC
countmatrix = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_HVG.csv")
countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])')

orig_labels = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_Seurat_Clustering.csv")
orig_labels = Array{Int64,1}(orig_labels[:x])
orig_labels = orig_labels .+ 1

umap_original_plot = DataFrame(hcat(umap_original[:,1],umap_original[:,2], string.(orig_labels)));
x = ["UMAP 1", "UMAP 2", "Cluster"];
names!(umap_original_plot, Symbol.(x));

# scVI Prior
genlab_scvi = maplab(countmatrix,gen_data_scvi,orig_labels);

# Save mapped clusters
cellnumber_tmp = collect(1:1:4182)
generated_labels = DataFrame(hcat(genlab_scvi,cellnumber_tmp))
CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Prior/PBMC/PBMC_scvi_prior_Gen_Labels_",t,"_seed",seedset,".csv"), generated_labels, writeheader = true)
   
   Random.seed!(321);
   umap_scvi = umap(gen_data_scvi',2; n_neighbors=30, metric=CosineDist(), min_dist=0.3)';
   umap_scvi_plot = DataFrame(hcat(umap_scvi[:,1],umap_scvi[:,2], string.(genlab_scvi)));
   names!(umap_scvi_plot, Symbol.(x));

   # DBI
   orig_dbindex = DBindex(countmatrix, orig_labels);
   scvi_dbindex = DBindex(gen_data_scvi, genlab_scvi);
   combined_dbi = vec([orig_dbindex, scvi_dbindex]);

   # ARI
   ari_index = Clustering.randindex(orig_labels,genlab_scvi)[2];

   push!(dbi_scvi_prior, combined_dbi);
   push!(ari, ari_index);

   countmap_scvi = countmap(genlab_scvi)
   countmap_orig = countmap(orig_labels)
   for l in 1:size(unique(orig_labels),1)
      if in(l, keys(countmap_scvi)) == true
         nothing
      else
         countmap_scvi[l] = 0
      end
   end

   # Cluster Proportions
   global tmp2 = DataFrame(hcat(collect(values(countmap_orig)),collect(values(countmap_scvi)), collect(keys(countmap_orig)), parse.(Int64, repeat(["$i"], size(unique(orig_labels),1)))))

for k in 1:size(tmp2,1)
   push!(cluster_prop_prior, tmp2[k,:])
end

end

# DBI Prior
dbi_scvi_prior = dbi_scvi_prior[2:end,:];
rename!(dbi_scvi_prior, :x1 => :Original, :x2 => :scVI_Prior);
d_scvi_prior = stack(dbi_scvi_prior, 1:2);

serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Prior/PBMC/pbmc_dbi_",t,"cells_scvi_prior_Seurat"),dbi_scvi_prior)

# ARI
ari = ari[2:end,:];
rename!(ari, :x1 => :ARI);

serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Prior/PBMC/pbmc_ari_",t,"cells_scvi_prior_Seurat"),ari)

# Cluster Proportions scVI Prior
cluster_prop_prior = cluster_prop_prior[2:end,:]
rename!(cluster_prop_prior, :x1 => :Original, :x2 => :scVI_Prior, :x3 => :Cluster, :x4 => :Rep);

serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Prior/PBMC/cluster_prop_prior_",t,"Cells_dbm_Upsample_Seurat"),cluster_prop_prior)
end

#################################################################################################################
#################################################################################################################
############################ scVI Posterior Upsample DBI, ARI, Cluster Proportions ##############################
#################################################################################################################
#################################################################################################################

global countmatrix = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_HVG.csv")
global countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])')
Random.seed!(321);
global umap_original = umap(countmatrix',2; n_neighbors=30, metric=CosineDist(), min_dist=0.3)';

Iter = ["384", "768", "1152", "1536", "1920", "2304"]
for t in Iter

dbi_scvi_posterior = DataFrame(Array{Float64,2}(undef, 1, 2));
ari = DataFrame(Array{Float64,2}(undef, 1, 1));
cluster_prop_posterior = DataFrame(Array{Int64,2}(undef, 1, 4));
for i = 1:30

py"""
n_epochs_all = None
save_path = 'C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/cell_samples/PBMC/'
show_plot = True
"""

py"""
import os
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, CsvDataset
from scvi.models import *
from scvi.inference import *
import torch
import urllib.request
"""

cd("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/cell_samples/PBMC/");
filenames = readdir()[occursin.(string("cells",t,"_seed"), readdir())];
tmp = filenames[i];
py"""
countmatrix = CsvDataset(str($tmp), save_path = save_path, new_n_genes = 500)
"""

countmatrix = py"CsvDataset(str($tmp), save_path = save_path, new_n_genes = 500)"

# Read train/test split indices
filenames_train = readdir()[occursin.(string("cells",t,"_train"), readdir())];
tmp = filenames_train[i];
trainidxs = CSV.read(tmp)[:,1];
py"""
trainidxs_py = $trainidxs
trainidxs_py = np.array(trainidxs_py)
"""

filenames_test = readdir()[occursin.(string("cells",t,"_train"), readdir())];
tmp = filenames_test[i];
testidxs = CSV.read(tmp)[:,1];
py"""
testidxs_py = $testidxs
testidxs_py = np.array(testidxs_py)
"""

n_cells = parse(Int64, t)
py"""
n_epochs = 100
lr = 1e-3
use_batches = False
use_cuda = True
n_cells = $n_cells
"""

if t == "384" || t == "768"
   seedset = filenames[i][19:end-4]
else
   seedset = filenames[i][20:end-4]
end

# Train the model and output model likelihood every 5 epochs
py"""
random.seed(int($seedset))
vae = VAE(countmatrix.nb_genes, n_batch=countmatrix.n_batches * use_batches, reconstruction_loss='nb', dispersion = 'gene')
trainer = UnsupervisedTrainer(vae,
   countmatrix,
   train_size=0.7,
   trainidxs=trainidxs_py,
   testidxs=testidxs_py,
   use_cuda=use_cuda,
   frequency=5,
)
"""

py"""
trainer.train(n_epochs=n_epochs, lr=lr)
"""

# Obtaining the posterior object and sample latent space from it
py"""
full = trainer.create_posterior(trainer.model, countmatrix, indices=np.arange(len(countmatrix)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()
"""

full =  py"trainer.create_posterior(trainer.model, countmatrix, indices=np.arange(len(countmatrix)))";
latent, batch_indices, labels = py"full.sequential(4182).get_latent()";
batch_indices = py"batch_indices.ravel()";

##############################################
# Generate Data from Posterior with upsampling
##############################################
n_cells = parse(Int64, t)
py"""
n_cells = $n_cells
"""
n_subset = n_cells

# Read original data
countmatrix = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_HVG.csv")
countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])');

# Generate Data
gen_data_scvi = py"full.generate(batch_size=n_cells, n_samples=10, sample_prior=False, n_cells=n_cells)";
n_paste = collect(1:trunc(Int,size(countmatrix,1)/n_subset))
global gen_data_scvi_posterior = gen_data_scvi[1][:,:,1]
for i in 2:size(n_paste,1)

   global gen_data_scvi_posterior = vcat(gen_data_scvi_posterior, gen_data_scvi[1][:,:,i])
end
rest = size(countmatrix,1) % n_subset
gen_data_scvi_posterior = Array{Float64,2}(vcat(gen_data_scvi_posterior,gen_data_scvi[1][1:rest,:,size(n_paste,1) + 1]))

maplab(x1,x2,y) = map(arg -> y[findmin(sum((x1 .- x2[arg:arg,:]).^2,dims=2))[2][1]],1:size(x2,1))  

# Original data
orig_labels = CSV.read("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/PBMC_Seurat_Clustering.csv")
orig_labels = Array{Int64,1}(orig_labels[:x])
orig_labels = orig_labels .+ 1

umap_original_plot = DataFrame(hcat(umap_original[:,1],umap_original[:,2], string.(orig_labels)));
x = ["UMAP 1", "UMAP 2", "Cluster"];
names!(umap_original_plot, Symbol.(x));

# scVI Posterior
genlab_scvi = maplab(countmatrix,gen_data_scvi_posterior,orig_labels);

# Save mapped clusters
cellnumber_tmp = collect(1:1:4182)
generated_labels = DataFrame(hcat(genlab_scvi,cellnumber_tmp))
CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Posterior/PBMC/PBMC_scvi_posterior_Gen_Labels_",t,"_seed",seedset,".csv"), generated_labels, writeheader = true)

Random.seed!(321);
umap_scvi = umap(gen_data_scvi_posterior',2; n_neighbors=30, metric=CosineDist(), min_dist=0.3)';
umap_scvi_plot = DataFrame(hcat(umap_scvi[:,1],umap_scvi[:,2], string.(genlab_scvi)));
names!(umap_scvi_plot, Symbol.(x));

# DBI
orig_dbindex = BMs.DBindex(countmatrix, orig_labels);
scvi_dbindex = BMs.DBindex(gen_data_scvi_posterior, genlab_scvi);
combined_dbi = vec([orig_dbindex, scvi_dbindex]);

# ARI
ari_index = Clustering.randindex(orig_labels,genlab_scvi)[2];

push!(dbi_scvi_posterior, combined_dbi);
push!(ari, ari_index);

countmap_scvi = countmap(genlab_scvi)
countmap_orig = countmap(orig_labels)
for l in 1:size(unique(orig_labels),1)
   if in(l, keys(countmap_scvi)) == true
      nothing
   else
      countmap_scvi[l] = 0
   end
end

   # Cluster Proportions
global tmp1 = DataFrame(hcat(collect(values(countmap_orig)),collect(values(countmap_scvi)), collect(keys(countmap_orig)), parse.(Int64, repeat(["$i"], size(unique(orig_labels),1)))))

for k in 1:size(tmp1,1)
   push!(cluster_prop_posterior, tmp1[k,:])
end

end

# DBI Posterior
dbi_scvi_posterior = dbi_scvi_posterior[2:end,:];
rename!(dbi_scvi_posterior, :x1 => :Original, :x2 => :scVI_Posterior);
d_scvi_posterior = stack(dbi_scvi_posterior, 1:2);

serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Posterior/PBMC/pbmc_dbi_",t,"cells_scvi_posterior_Seurat"),d_scvi_posterior)

# ARI
ari = ari[2:end,:];
rename!(ari, :x1 => :ARI);

serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Posterior/PBMC/pbmc_ari_",t,"cells_scvi_prior_Seurat"),ari)

# Cluster Proportions scVI Posterior
cluster_prop_posterior = cluster_prop_posterior[2:end,:]
rename!(cluster_prop_posterior, :x1 => :Original, :x2 => :scVI_Posterior, :x3 => :Cluster, :x4 => :Rep);

serialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Posterior/PBMC/cluster_prop_posterior_",t,"Cells_dbm_Upsample_Seurat"),cluster_prop_posterior)
end
