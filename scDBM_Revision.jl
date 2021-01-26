using Pkg, HTTP, CSV, DataFrames, Random, Distances, UMAP, Clustering, Gadfly, DelimitedFiles, MultivariateStats, Serialization

# Load single-cell deep Boltzmann machines (scDBM) Package 
# Pkg.add(PackageSpec(url="https://github.com/MTreppner/scDBM.jl", rev="master"))
using scDBM

set_default_plot_size(50cm, 20cm)

# Read PBMC4k Data  
countmatrix = CSV.read("C:/Users/treppner/Dropbox/PhD/scRNA-seq/PBMC/PBMC_HVG.csv")
genenames = countmatrix[:,1]
countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:501])');

# Sample pilot dataset
Random.seed!(111);
cells = 384; # Size of a 384 well plate
random_cells = rand(1:size(countmatrix,1),cells);
pilot_data = countmatrix[random_cells,:];

# Fit scDBM

# Set up train and test set.
Random.seed!(101);
data, datatest = splitdata(pilot_data, 0.3);
datadict = DataDict("Training data" => data, "Test data" => datatest);

# Set initial parameters.
epochs = 750;                                                        # Train for 750 epochs
init_disp = (ones(size(data,2)) .* 1.0);                             # Set inverse dispersion parameter
lr = [0.000001*ones(3500);0.00005*ones(1000);0.00001*ones(10000)];   # Set learning rate
regularization = 220.0;

# Train scDBM.
monitor = Monitor(); monitor1 = Monitor(); monitor2 = Monitor();
Random.seed!(59);
dbm = fitdbm(data, epochs = 200, learningrate = 0.0001, batchsizepretraining = 32,
      monitoringdatapretraining = datadict,
      pretraining = [
            # Negative-Binomial RBM
            TrainLayer(nhidden = 12,
            learningrates = lr, 
            epochs = epochs,
            rbmtype = NegativeBinomialBernoulliRBM, 
            inversedispersion = init_disp,
            fisherscoring = 1,
            lambda = regularization,
               monitoring = (rbm, epoch, datadict) -> begin if epoch % 20 == 0 monitorreconstructionerror!(monitor1, rbm, epoch, datadict) end end);
            # Bernoulli RBM
            TrainLayer(nhidden = 4, 
            learningrate = 0.0001,
            epochs = 500,
               monitoring = (rbm, epoch, datadict) -> begin if epoch % 20 == 0 monitorreconstructionerror!(monitor2, rbm, epoch, datadict) end end)
            ];
monitoring = (dbm, epoch) -> begin 
                              if epoch % 5 == 0 
                                 monitorlogproblowerbound!(monitor, dbm, epoch, datadict)
                              end
                           end);
hstack(
   plotevaluation(monitor1, monitorreconstructionerror),
   plotevaluation(monitor2, monitorreconstructionerror),
   plotevaluation(monitor, monitorlogproblowerbound)
)

# Save trained scDBM
serialize("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM_one_plate", dbm);

# Load trained scDBM
trainingresult = deserialize("C:/Users/treppner/Dropbox/PhD/scDBM.jl/revision/scDBM/PBMC/scDBM_384Cells_PBMC")

# Generate synthetic samples.
number_gensamples = size(pilot_data,1);
synthetic_cells = initparticles(dbm, number_gensamples);
gibbssamplenegativebinomial!(pilot_data,synthetic_cells, dbm, 30);

# Dimensionality reduction, clustering, and Davies-Bouldin index  

# PCA and k-means clustering for pilot data
pca_pilot_data = fit(PCA, pilot_data; maxoutdim=50);
counts_clust_original = kmeans(pca_pilot_data.proj', 10);
orig_labels = counts_clust_original.assignments;
pca_original_plot = DataFrame(hcat(pca_pilot_data.proj[:,1],pca_pilot_data.proj[:,2], string.(orig_labels)));
x = ["PC 1", "PC 2", "Cluster"];
names!(pca_original_plot, Symbol.(x));

# Map cells using Euclidean distance
maplab(x1,x2,y) = map(arg -> y[findmin(sum((x1 .- x2[arg:arg,:]).^2,dims=2))[2][1]],1:size(x2,1));
scDBM_labels = maplab(pilot_data,synthetic_cells[1],orig_labels);

countmap(scDBM_labels)
countmap(counts_clust_original.assignments)

# PCA and k-means clustering for scDBM data
pca_scDBM_data = fit(PCA, synthetic_cells[1]; maxoutdim=50);
pca_scDBM_plot = DataFrame(hcat(pca_scDBM_data.proj[:,1],pca_scDBM_data.proj[:,2], string.(scDBM_labels)));
names!(pca_scDBM_plot, Symbol.(x));

orig_dbindex = DBindex(pilot_data, orig_labels);
dbm_dbindex = DBindex(synthetic_cells[1], scDBM_labels);
