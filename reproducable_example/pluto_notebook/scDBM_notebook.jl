### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# ╔═╡ d54abd5a-5f44-11eb-34fe-c96ecbd1327c
begin
	# Load required packages
	using Pkg, HTTP, CSV, DataFrames, Random, Distances, 
        UMAP, Clustering, Gadfly, DelimitedFiles, 
        MultivariateStats, Cairo, Fontconfig, StatsBase

	# Set default plot size
    set_default_plot_size(50cm, 20cm)
	
	# Set working directory
	cd("/Users/martintreppner/Desktop")
end;

# ╔═╡ 59a1670e-5fb6-11eb-0cf1-6fefc45fb083
begin
	# Install and Load single-cell deep Boltzmann machine (scDBM) 
	# Pkg.add(PackageSpec(url="https://github.com/MTreppner/scDBM.jl", rev="master"))
	using scDBM
end;

# ╔═╡ 09c53f44-5fb6-11eb-121b-757954f80b1d
begin
	# Load Python packages using PyCall
	using PyCall
	
	random = pyimport("random")
	os = pyimport("os")
	np = pyimport("numpy")
	pd = pyimport("pandas")
	scvi = pyimport("scvi")
	scvi_dataset = pyimport("scvi.dataset")
	scvi_models = pyimport("scvi.models")
	scvi_inference = pyimport("scvi.inference")
end;

# ╔═╡ d0caecf8-5f44-11eb-26fc-e749e38caaca
md"
##### Synthetic Single-Cell RNA-Sequencing Data from Small Pilot Studies using Deep Generative Models
"

# ╔═╡ db8ff29a-5f44-11eb-1178-076b084421f1
md"
###### Single-Cell Deep Boltzmann Machine (scDBM)
"

# ╔═╡ 0dc3d50e-5f45-11eb-1d35-957a2ad4984a
begin
	# Set hyperparameters
	epochsfirstlayer = 60
    learningratedbm = 1.0e-11
    dispinit = 0.5
    epochsdbm = 10
    nhiddennegbin = 4
    nhiddenbernoulli = 2
    lambdareg = 1.0
    repititions = collect(4:10:300)
end;

# ╔═╡ 641ff96c-5f4c-11eb-0a04-07bc290451a7
# Create mapping function
maplab(x1,x2,y) = map(arg -> y[findmin(sum((x1 .- x2[arg:arg,:]).^2,dims=2))[2][1]],1:size(x2,1));

# ╔═╡ 9c4dee34-5fb5-11eb-1711-3f606be0a97b
md"
Since the running time of the algorithms across all plates and iterations take a considerable amount of time, we only consider a subset of the overall analyses from the manuscript.
"

# ╔═╡ e0f4cab2-5f44-11eb-0471-8f2825d901cc
begin
	# Initialize arrays
	global dbi = DataFrame(Array{Float64,2}(undef, 1, 2));
	global dbi[:cells] = "Init"
	global ari = DataFrame(Array{Float64,2}(undef, 1, 1));
	global ari[:cells] = "Init"
	global cluster_prop = DataFrame(Array{Int64,2}(undef, 1, 4));
	global cluster_prop[:cells] = "Init"
	
	global Iter = ["384", "1152", "1920"]
	for t in Iter
		for i in 1:5
			# Countmatrix with 2000 highly variable genes
			Random.seed!(repititions[i])
			countmatrix = CSV.read("notebook_test/segerstolpe_hvg.csv")
			genenames = countmatrix[:,1]
			countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])');
			
			# Sample t random cells
			cells = parse(Int64, t)
			global seedset = repititions[i]
			random_cells = rand(1:size(countmatrix,1),cells)

			# Save sub-sampled data
			countmatrix_tmp =
			DataFrame(hcat(collect(1:size(countmatrix[random_cells,:]',1)),
					countmatrix[random_cells,:]'))
			CSV.write(string("notebook_test/segerstolpe_cells",
					t,
					"_seed",
					seedset,
					".csv"),
				countmatrix_tmp, 
				writeheader = true)
			
			# Train- test split
			data, datatest = splitdata(countmatrix[random_cells,:], 0.30);
			datadict = DataDict("Training data" => data, "Test data" => datatest);
			lr_rbm = [1.0e-11*ones(3500);0.00005*ones(1000);0.00001*ones(10000)];

			# Train scDBM
			global dbm = fitdbm(data, 
				epochs = epochsdbm,
				learningrate = learningratedbm,
				batchsizepretraining = 64, 
				pretraining = [
				# Negative-Binomial RBM
				TrainLayer(nhidden = nhiddennegbin, 
						learningrates = lr_rbm, 
						pcd = true,
						zeroinflation = false, 
						epochs = epochsfirstlayer,
						rbmtype = NegativeBinomialBernoulliRBM, 
						inversedispersion = (ones(size(data,2)) .* dispinit), 
						fisherscoring = 1, 
						estimatedispersion = "gene", 
						lambda = lambdareg)
				# Bernoulli RBM
				TrainLayer(nhidden = nhiddenbernoulli, 
						learningrate = 1.0e-11, 
						epochs = 40)]
				)
				
			# Generate synthetic samples
			ngensamples = size(countmatrix,1);
			particles = initparticles(dbm, ngensamples);
			gibbssamplenegativebinomial!(countmatrix,
				particles, 
				dbm, 50; 
				zeroinflation=false);
			gen_data_dbm = particles[1];

			# Save generated data for one iteration
			if i == 3
				gen_data_dbm_out = DataFrame(gen_data_dbm)
				CSV.write(string("notebook_test/gen_data_scdbm",
					t,
					"_seed",
					seedset,
					".csv"),
				gen_data_dbm_out, 
				writeheader = true)
			end
			
			# UMAP on synthetic data
			Random.seed!(123);
			umap_dbm = umap(gen_data_dbm',2; 
				n_neighbors=30, 
				metric=CosineDist(), 
				min_dist=0.3)';

			# Load original cluster labels
			orig_labels = CSV.read("notebook_test/segerstolpe_hvg_clustering.csv");
			orig_labels = Array{Int64,1}(orig_labels[:x]);

			# UMAP on original data
			countmatrix = CSV.read("notebook_test/segerstolpe_hvg.csv");
			countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])');
			Random.seed!(321);
			umap_original = umap(countmatrix',2; 
				n_neighbors=30, 
				metric=CosineDist(), 
				min_dist=0.3)';
			
			# Mapping in UMAP space
			genlab_dbm = maplab(umap_original,umap_dbm,orig_labels);

			# DBI
			orig_dbindex = DBindex(countmatrix, orig_labels);
			dbm_dbindex = DBindex(gen_data_dbm, genlab_dbm);
			combined_dbi = DataFrame(vec([orig_dbindex, dbm_dbindex])')
        	combined_dbi[:cells] = t


			# ARI
			ari_index = DataFrame(
				hcat(Clustering.randindex(orig_labels,genlab_dbm)[2], t))
	        rename!(ari_index, :x2 => :cells);

			global dbi = vcat(dbi, combined_dbi)
			global ari = vcat(ari, ari_index)

			# Cluster Proportions
			countmap_dbm = countmap(genlab_dbm)
			countmap_orig = countmap(orig_labels)

			for l in 1:size(unique(orig_labels),1)
				if in(l, keys(countmap_dbm)) == true
					nothing
				else
					countmap_dbm[l] = 0
				end
			end

			tmp =	DataFrame(hcat(collect(values(countmap_orig)),
						collect(values(countmap_dbm)), 
						collect(keys(countmap_orig)), 
						parse.(Int64, repeat(["$i"], size(unique(orig_labels),1)))))
	        tmp[:cells] = t
    	    global cluster_prop = vcat(cluster_prop, tmp)
		end
	end
end;

# ╔═╡ 2c93f912-5f4c-11eb-3873-237144ded866
begin
	# DBI scDBM
	dbi_all = dbi[2:end,:]
	rename!(dbi_all, :x1 => :Original, :x2 => :scDBM);
	dbi_all = DataFrames.stack(dbi_all, 1:2)

	# ARI scDBM
	ari_all = ari[2:end,:]
	rename!(ari_all, :x1 => :ARI);
	
	# Cluster Proportions scDBM
	cluster_prop_all = cluster_prop[2:end,:]
	rename!(cluster_prop_all, 
		:x1 => :Original, 
		:x2 => :proportion, 
		:x3 => :Cluster, 
		:x4 => :Rep);
	cluster_prop_all[:model] = "scDBM"
end;

# ╔═╡ 71528b5c-5f4e-11eb-1c3e-7f62baec183a
md"
###### Single-Cell Variational Inference (scVI)
"

# ╔═╡ 87a19476-5fb6-11eb-36bf-7d0244187077
begin
	# Set fixed number of genes to avoid automatic sampling in scVI
	n_genes = 2000

	cd("notebook_test/");
	
	# Initialize arrays
	global dbi_scvi = DataFrame(Array{Float64,2}(undef, 1, 2));
	global dbi_scvi[:cells] = "Init";
	global ari_scvi = DataFrame(Array{Float64,2}(undef, 1, 1));
	global ari_scvi[:cells] = "Init"
	global cluster_prop_scvi = DataFrame(Array{Int64,2}(undef, 1, 4));
	global cluster_prop_scvi[:cells] = "Init"
	
	for t in Iter
    	for i = 1:5
			
			# Read sub-sampled data
			filenames = readdir()[occursin.(string("cells",t,"_seed"), readdir())];
			tmp = filenames[i];
        
			countmatrix_sample = scvi_dataset.CsvDataset(tmp, 
					save_path = "", 
					new_n_genes = n_genes); 			
			
			# Set hyperparameters
			n_latent=14
			n_hidden=64
			n_layers=2
			n_epochs=200
			lr_scvi=0.01
			use_batches = false
			use_cuda = true

			# Train the model and output model likelihood every epoch
			vae = scvi_models.VAE(countmatrix_sample.nb_genes, 
				n_batch=countmatrix_sample.n_batches * use_batches, 
				n_latent=n_latent, 
				n_hidden=n_hidden,
				n_layers=n_layers, 
				reconstruction_loss="nb", 
				dispersion = "gene"
			);
			
			# Set up unsupervised trainer
			trainer = scvi_inference.UnsupervisedTrainer(vae,
				countmatrix_sample,
				train_size=0.7,
				use_cuda=use_cuda,
				frequency=1,
				n_epochs_kl_warmup=75,
				batch_size=36
        	);
			
			# Train scVI
			trainer.train(n_epochs=n_epochs,lr=lr_scvi)
			
			# Create posterior model
			full = trainer.create_posterior(trainer.model, countmatrix_sample);
			
			batch_size = 32
        	n_cells = 16

			# Generate Data
			gen_data_scvi = full.generate(batch_size=batch_size,
				n_samples=10,
				n_cells=n_cells)
			
			# Paste together generated cells
			n_subset = parse(Int64, t)
			
			n_paste = collect(1:trunc(Int,2090/n_subset))
			global gen_data_scvi_posterior = gen_data_scvi[1][:,:,1]
			for i in 2:size(n_paste,1)
				
				global gen_data_scvi_posterior = vcat(gen_data_scvi_posterior,
					gen_data_scvi[1][:,:,i])
			end
			rest = 2090 % n_subset
			
			gen_data_scvi = Array{Float64,2}(vcat(gen_data_scvi_posterior,
					gen_data_scvi[1][1:rest,:,size(n_paste,1) + 1]))

			# Save generated data for one iteration
			if i == 3
				gen_data_scvi_out = DataFrame(gen_data_scvi)
				CSV.write(string("gen_data_scvi",
					t,
					"_seed",
					seedset,
					".csv"),
				gen_data_scvi_out, 
				writeheader = true)
			end

			# UMAP on synthetic data
			Random.seed!(123);
			umap_scvi = umap(gen_data_scvi',2; 
				n_neighbors=30, 
				metric=CosineDist(), 
				min_dist=0.3)';
			
			# Load original cluster labels
			orig_labels = CSV.read("segerstolpe_hvg_clustering.csv")
			orig_labels = Array{Int64,1}(orig_labels[:x])
		
			# UMAP on original data
			countmatrix = CSV.read("segerstolpe_hvg.csv")
			countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])')

			Random.seed!(321);
			umap_original = umap(countmatrix',2; 
				n_neighbors=30, 
				metric=CosineDist(), 
				min_dist=0.3)';
			
			# Mapping in UMAP space
	        genlab_scvi = maplab(umap_original,umap_scvi,orig_labels);

			# DBI
			orig_dbindex_scvi = DBindex(countmatrix, orig_labels);
			scvi_dbindex = DBindex(gen_data_scvi, genlab_scvi);
			combined_dbi_scvi = DataFrame(vec([orig_dbindex_scvi, scvi_dbindex])')
        	combined_dbi_scvi[:cells] = t

			# ARI
			ari_index_scvi = DataFrame(
				hcat(Clustering.randindex(orig_labels,genlab_scvi)[2], t))
	        rename!(ari_index_scvi, :x2 => :cells);

			global dbi_scvi = vcat(dbi_scvi, combined_dbi_scvi)
			global ari_scvi = vcat(ari_scvi, ari_index_scvi)

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
			global tmp1 = DataFrame(hcat(collect(values(countmap_orig)),
					collect(values(countmap_scvi)), 
					collect(keys(countmap_orig)), parse.(Int64, repeat(["$i"],
							size(unique(orig_labels),1)))))
			global tmp1[:cells] = t
    	    global cluster_prop_scvi = vcat(cluster_prop_scvi, tmp1)
		end
	end
end;

# ╔═╡ bd161ef8-5fb6-11eb-2fb7-335fb40af173
begin
	# DBI scVI
	dbi_scvi_all = dbi_scvi[2:end,:];
	rename!(dbi_scvi_all, :x1 => :Original, :x2 => :scVI);
	dbi_scvi_all = DataFrames.stack(dbi_scvi_all, 1:2);
   
    # ARI scVI
    ari_scvi_all = ari_scvi[2:end,:];
    rename!(ari_scvi_all, :x1 => :ARI);

    # Cluster Proportions scVI
    cluster_prop_scvi_all = cluster_prop_scvi[2:end,:]
    rename!(cluster_prop_scvi_all, 
		:x1 => :Original, 
		:x2 => :proportion, 
		:x3 => :Cluster, 
		:x4 => :Rep);
	cluster_prop_scvi_all[:model] = "scVI"
end;

# ╔═╡ 945354fe-5fe6-11eb-3f0e-2df1f5961fac
md"
###### Combine results for both models
"

# ╔═╡ a242d864-5fe6-11eb-21f6-4b7640b2e816
begin
	dbi_plotting = vcat(dbi_all, dbi_scvi_all)
	ari_all[:model] = "scDBM"
	ari_scvi_all[:model] = "scVI"
	ari_plotting = vcat(ari_all, ari_scvi_all)
	cluster_prop_plotting = vcat(cluster_prop_all, cluster_prop_scvi_all)
end;

# ╔═╡ 961d9ff0-5fe7-11eb-14a4-5bb9ee53054a
begin
	# Save DBI data
	CSV.write(string("dbi_plotting.csv"),
				dbi_plotting, 
				writeheader = true)

	# Save ARI data
	CSV.write(string("ari_plotting.csv"),
				ari_plotting, 
				writeheader = true)

	# Save cluster proportion data
	CSV.write(string("cluster_prop_plotting.csv"),
				cluster_prop_plotting, 
				writeheader = true)
end;

# ╔═╡ Cell order:
# ╟─d0caecf8-5f44-11eb-26fc-e749e38caaca
# ╠═d54abd5a-5f44-11eb-34fe-c96ecbd1327c
# ╠═db8ff29a-5f44-11eb-1178-076b084421f1
# ╠═59a1670e-5fb6-11eb-0cf1-6fefc45fb083
# ╠═0dc3d50e-5f45-11eb-1d35-957a2ad4984a
# ╠═641ff96c-5f4c-11eb-0a04-07bc290451a7
# ╟─9c4dee34-5fb5-11eb-1711-3f606be0a97b
# ╠═e0f4cab2-5f44-11eb-0471-8f2825d901cc
# ╠═2c93f912-5f4c-11eb-3873-237144ded866
# ╟─71528b5c-5f4e-11eb-1c3e-7f62baec183a
# ╠═09c53f44-5fb6-11eb-121b-757954f80b1d
# ╠═87a19476-5fb6-11eb-36bf-7d0244187077
# ╠═bd161ef8-5fb6-11eb-2fb7-335fb40af173
# ╟─945354fe-5fe6-11eb-3f0e-2df1f5961fac
# ╠═a242d864-5fe6-11eb-21f6-4b7640b2e816
# ╠═961d9ff0-5fe7-11eb-14a4-5bb9ee53054a
