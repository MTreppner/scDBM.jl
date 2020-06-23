using Serialization

# Deserialize DBI tables
Iter = ["384", "768", "1152", "1536", "1920", "2304"]
dbi = DataFrame(Array{Float64,2}(undef, 1, 3))
rename!(dbi, :x1 => :variable, :x2 => :value, :x3 => :cells);
for t in Iter
   # scDBM
   dbi_scdbm = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/dbi_",t,"cells_dbm_PBMC_Seurat"));
   tmp = repeat([string(t," Cells")], size(dbi_scdbm,1)); 
   dbi_scdbm = hcat(dbi_scdbm, tmp);
   rename!(dbi_scdbm, :x1 => :cells);

   # scVI Prior
   dbi_scvi_prior = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Prior/PBMC/pbmc_dbi_",t,"cells_scvi_prior_Seurat"));
   dbi_scvi_prior = stack(dbi_scvi_prior)
   tmp = repeat([string(t," Cells")], size(dbi_scvi_prior,1)); 
   dbi_scvi_prior = hcat(dbi_scvi_prior, tmp);
   rename!(dbi_scvi_prior, :x1 => :cells);

   # scVI Posterior
   dbi_scvi_posterior = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Posterior/PBMC/pbmc_dbi_",t,"cells_scvi_posterior_Seurat"));
   tmp = repeat([string(t," Cells")], size(dbi_scvi_posterior,1)); 
   dbi_scvi_posterior = hcat(dbi_scvi_posterior, tmp);
   rename!(dbi_scvi_posterior, :x1 => :cells);

   global dbi = vcat(dbi, dbi_scdbm, dbi_scvi_prior, dbi_scvi_posterior)
end
dbi = dbi[2:end,:]

CSV.write("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/plotting/PBMC/dbi_plotting.csv", dbi, writeheader = true)

# Deserialize ARI tables
Iter = ["384", "768", "1152", "1536", "1920", "2304"]
ari = DataFrame(Array{Float64,2}(undef, 1, 3))
rename!(ari, :x1 => :ARI, :x2 => :cells, :x3 => :variable);
for t in Iter
   # scDBM
   ari_scdbm = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/ari_",t,"cells_dbm_PBMC_Seurat"));
   tmp = repeat([string(t," Cells")], size(ari_scdbm,1));
   tmp1 = repeat(["scDBM"], size(ari_scdbm,1));
   ari_scdbm = hcat(ari_scdbm, tmp, tmp1, makeunique=true);
   rename!(ari_scdbm, :x1 => :cells, :x1_1 => :variable);

   # scVI Prior
   ari_scvi_prior = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Prior/PBMC/pbmc_ari_",t,"cells_scvi_prior_Seurat"));
   tmp = repeat([string(t," Cells")], size(ari_scvi_prior,1)); 
   tmp1 = repeat(["scVI Prior"], size(ari_scvi_prior,1));
   ari_scvi_prior = hcat(ari_scvi_prior, tmp, tmp1, makeunique=true);
   rename!(ari_scvi_prior, :x1 => :cells, :x1_1 => :variable);

   # scVI Posterior
   ari_scvi_posterior = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Posterior/PBMC/pbmc_ari_",t,"cells_scvi_prior_Seurat"));
   tmp = repeat([string(t," Cells")], size(ari_scvi_posterior,1)); 
   tmp1 = repeat(["scVI Posterior"], size(ari_scvi_posterior,1));
   ari_scvi_posterior = hcat(ari_scvi_posterior, tmp, tmp1, makeunique=true);
   rename!(ari_scvi_posterior, :x1 => :cells, :x1_1 => :variable);

   global ari = vcat(ari, ari_scdbm, ari_scvi_prior, ari_scvi_posterior)
end
ari = ari[2:end,:]

CSV.write("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/plotting/PBMC/ari_plotting.csv", ari, writeheader = true)

# Cluster proportions
Iter = ["384", "768", "1152", "1536", "1920", "2304"]
for t in Iter
   prop_scdbm = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/cluster_prop_",t,"Cells_dbm_PBMC_Seurat"))
   CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/plotting/PBMC/prop_scdbm",t,"cells.csv"), prop_scdbm, writeheader = true)

   prop_scvi_prior = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Prior/PBMC/cluster_prop_prior_",t,"Cells_dbm_Upsample_Seurat"))
   CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/plotting/PBMC/prop_scvi_prior",t,"cells.csv"), prop_scvi_prior, writeheader = true)

   prop_scvi_posterior = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Posterior/PBMC/cluster_prop_posterior_",t,"Cells_dbm_Upsample_Seurat"))
   CSV.write(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/plotting/PBMC/prop_scvi_posterior",t,"cells.csv"), prop_scvi_posterior, writeheader = true)
end

# Gene distributions
trainingresult = deserialize("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/scDBM_384Cells_PBMC")

cd("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/")
filenames = readdir()[occursin.(r"cells384_seed", readdir())];
countmatrix = CSV.read(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/",filenames[19]))
countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])')

seedset = filenames[19][19:end-4]
for j = 1:30
   if string(trainingresult[j][12]) == seedset
      global dbm = trainingresult[j][9]
   end
end
ngensamples = 4182;
particles = BMs.initparticles(dbm, ngensamples);
BMs.gibbssamplenegativebinomial!(countmatrix,particles, dbm, 100; zeroinflation=false)
gen_data_dbm = DataFrame(particles[1])

CSV.write("C:/Users/treppner/Dropbox/PhD/scRNA-seq/AdrianData/scRNAseq_data_all/cor_plotting_4182dbm_PBMC.csv", gen_data_dbm, writeheader = true)
