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
   dbi_scvi_posterior = stack(dbi_scvi_posterior)
   tmp = repeat([string(t," Cells")], size(dbi_scvi_posterior,1)); 
   dbi_scvi_posterior = hcat(dbi_scvi_posterior, tmp);
   rename!(dbi_scvi_posterior, :x1 => :cells);

   global dbi = vcat(dbi, dbi_scdbm, dbi_scvi_prior, dbi_scvi_posterior)
end
dbi = dbi[2:end,:]

CSV.write("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/plotting/dbi_plotting.csv", dbi, writeheader = true)

# Deserialize ARI tables
Iter = ["384", "768", "1152", "1536", "1920", "2304"]
ari = DataFrame(Array{Float64,2}(undef, 1, 3))
rename!(ari, :x1 => :variable, :x2 => :value, :x3 => :cells);
for t in Iter
   # scDBM
   ari_scdbm = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/ari_",t,"cells_dbm_PBMC_Seurat"));
   tmp = repeat([string(t," Cells")], size(ari_scdbm,1)); 
   ari_scdbm = hcat(ari_scdbm, tmp);
   rename!(ari_scdbm, :x1 => :cells);

   # scVI Prior
   ari_scvi_prior = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Prior/PBMC/pbmc_ari_",t,"cells_scvi_prior_Seurat"));
   tmp = repeat([string(t," Cells")], size(ari_scvi_prior,1)); 
   ari_scvi_prior = hcat(ari_scvi_prior, tmp);
   rename!(ari_scvi_prior, :x1 => :cells);

   # scVI Posterior
   ari_scvi_posterior = deserialize(string("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scVI_Posterior/PBMC/pbmc_ari_",t,"cells_scvi_posterior_Seurat"));
   ari_scvi_posterior = stack(ari_scvi_posterior)
   tmp = repeat([string(t," Cells")], size(ari_scvi_posterior,1)); 
   ari_scvi_posterior = hcat(ari_scvi_posterior, tmp);
   rename!(ari_scvi_posterior, :x1 => :cells);

   global dbi = vcat(dbi, ari_scdbm, ari_scvi_prior, ari_scvi_posterior)
end
ari = ari[2:end,:]

CSV.write("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/plotting/ari_plotting.csv", ari, writeheader = true)
