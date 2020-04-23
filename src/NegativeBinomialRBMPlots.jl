function libsizenormalization(x::Array{Float64,2})

   # Library size normalization (see Splatter Package Bioconductor [https://github.com/Oshlack/splatter/blob/382e22e6b22f32f2bb86a4152af369eda19aaba3/R/splat-estimate.R])
   libsize = mapslices(sum, x, dims = 2)
   libmedian = median(libsize)
   normalizedcounts = (x ./ libsize) .* libmedian

   normalizedcounts
end

function counthistograms(x::Array{Float64,2}, particles::Array{Array{Float64,2},1},ncells::Int)
   tmp1 = DataFrame(x);
   tmp1 = DataFrames.melt(tmp1);
   test = repeat(["Ground Truth"], size(tmp1,1));
   tmp1 = hcat(test,tmp1);

   tmp2 = DataFrame(particles[1]);
   tmp2 = DataFrames.melt(tmp2);
   test = repeat(["Generated Data"], size(tmp2,1));
   tmp2 = hcat(test,tmp2);

   data = vcat(tmp1,tmp2)
   data = data[:,[1,3]]

   rename!(data, :x1 => :Setting, :value => :ExpressionCount)

   p1 =  plot(data,   x = "ExpressionCount", 
         xgroup="Setting", 
         Geom.subplot_grid(Geom.histogram,Coord.cartesian(xmin=0.0, ymin=0.0)), 
         Guide.title("Histogram of Expression Counts for n=$ncells Highly Expressed Cells"),
         Guide.xlabel("Expression Count"))
end

function cellwise_meanvarianceplot(x::Array{Float64,2}, particles::Array{Float64,2},ncells::Int;logarithm::Bool)
   
   if logarithm
      # Take means and variances of each row
      variance_rbm = log.(mapslices(var,particles,dims=2) .+ 1.0)
      means_rbm = log.(mapslices(mean,particles,dims=2) .+ 1.0)

      variance_true = log.(mapslices(var,x,dims=2) .+ 1.0)
      means_true = log.(mapslices(mean,x,dims=2) .+ 1.0)
   else
      # Take means and variances of each row
      variance_rbm = mapslices(var,particles,dims=2)
      means_rbm = mapslices(mean,particles,dims=2)

      variance_true = mapslices(var,x,dims=2)
      means_true = mapslices(mean,x,dims=2)
   end

   test = repeat(["Ground Truth"], size(means_true,1))
   mean_var_true = DataFrame(hcat(test,means_true,variance_true))

   test = repeat(["Generated Data"], size(means_rbm,1))
   mean_var_rbm = DataFrame(hcat(test,means_rbm,variance_rbm))

   data = vcat(mean_var_true,mean_var_rbm)
   rename!(data, :x1 => :Setting, :x2 => :log_Mean , :x3 => :log_Var)

   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "log_Mean", y = "log_Var", 
      Guide.title("Cell-Wise Mean-Variance Plot"), 
      Geom.subplot_grid(layer(Geom.point), layer(intercept=[0.0], slope=[1.0], abline)),
      Guide.xlabel("log(Mean)"),
      Guide.ylabel("log(Variance)"))
end

function scvi_cellwise_meanvarianceplot(x::Array{Float64,2}, particles::Array{Float64,2},ncells::Int;logarithm::Bool)
   
   if logarithm
      # Take means and variances of each row
      variance_rbm = log.(mapslices(var,particles,dims=2) .+ 1.0)
      means_rbm = log.(mapslices(mean,particles,dims=2) .+ 1.0)

      variance_true = log.(mapslices(var,x,dims=2) .+ 1.0)
      means_true = log.(mapslices(mean,x,dims=2) .+ 1.0)
   else
      # Take means and variances of each row
      variance_rbm = mapslices(var,particles,dims=2)
      means_rbm = mapslices(mean,particles,dims=2)

      variance_true = mapslices(var,x,dims=2)
      means_true = mapslices(mean,x,dims=2)
   end

   test = repeat(["Ground Truth"], size(means_true,1))
   mean_var_true = DataFrame(hcat(test,means_true,variance_true))

   test = repeat(["scVI Generated Data"], size(means_rbm,1))
   mean_var_rbm = DataFrame(hcat(test,means_rbm,variance_rbm))

   data = vcat(mean_var_true,mean_var_rbm)
   rename!(data, :x1 => :Setting, :x2 => :log_Mean , :x3 => :log_Var)

   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "log_Mean", y = "log_Var", 
      Guide.title("Cell-Wise Mean-Variance Plot"), 
      Geom.subplot_grid(layer(Geom.point), layer(intercept=[0.0], slope=[1.0], abline)),
      Guide.xlabel("log(Mean)"),
      Guide.ylabel("log(Variance)"))
end

function compare_cellwise_meanvarianceplot(x::Array{Float64,2}, dbmgen::Array{Float64,2}, scvigen::Array{Float64,2};logarithm::Bool)
   
   if logarithm
      # Take means and variances of each row
      variance_dbm = log.(mapslices(var,dbmgen,dims=2) .+ 1.0)
      means_dbm = log.(mapslices(mean,dbmgen,dims=2) .+ 1.0)

      variance_scvi = log.(mapslices(var,scvigen,dims=2) .+ 1.0)
      means_scvi = log.(mapslices(mean,scvigen,dims=2) .+ 1.0)

      variance_true = log.(mapslices(var,x,dims=2) .+ 1.0)
      means_true = log.(mapslices(mean,x,dims=2) .+ 1.0)
   else
      # Take means and variances of each row
      variance_dbm = mapslices(var,dbmgen,dims=2)
      means_dbm = mapslices(mean,dbmgen,dims=2)

      variance_scvi = mapslices(var,scvigen,dims=2)
      means_scvi = mapslices(mean,scvigen,dims=2)

      variance_true = mapslices(var,x,dims=2)
      means_true = mapslices(mean,x,dims=2)
   end

   test = repeat(["Ground Truth"], size(means_true,1))
   mean_var_true = DataFrame(hcat(test,means_true,variance_true))

   test = repeat(["NB-DBM Generated Data"], size(means_dbm,1))
   mean_var_dbm = DataFrame(hcat(test,means_dbm,variance_dbm))

   test = repeat(["scVI Generated Data"], size(means_scvi,1))
   mean_var_scvi = DataFrame(hcat(test,means_scvi,variance_scvi))

   data = vcat(mean_var_true,mean_var_dbm, mean_var_scvi)
   rename!(data, :x1 => :Setting, :x2 => :log_Mean , :x3 => :log_Var)

   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "log_Mean", y = "log_Var", 
      Guide.title("Cell-Wise Mean-Variance Plot"), 
      Geom.subplot_grid(layer(Geom.point), layer(intercept=[0.0], slope=[1.0], abline)),
      Guide.xlabel("log(Mean)"),
      Guide.ylabel("log(Variance)"))
end

function genewise_meanvarianceplot(x::Array{Float64,2}, particles::Array{Float64,2};logarithm::Bool)
   
   if logarithm
      # Take means and variances of each row
      variance_rbm = log10.(mapslices(var,particles,dims=1)' .+ 1.0)
      means_rbm = log10.(mapslices(mean,particles,dims=1)' .+ 1.0)

      variance_true = log10.(mapslices(var,x,dims=1)' .+ 1.0)
      means_true = log10.(mapslices(mean,x,dims=1)' .+ 1.0)
   else
      # Take means and variances of each row
      variance_rbm = mapslices(var,particles,dims=1)'
      means_rbm = mapslices(mean,particles,dims=1)'

      variance_true = mapslices(var,x,dims=1)'
      means_true = mapslices(mean,x,dims=1)'
   end
   
   test = repeat(["Ground Truth"], size(means_true,1))
   mean_var_true = DataFrame(hcat(test,means_true,variance_true))

   test = repeat(["Generated Data"], size(means_rbm,1))
   mean_var_rbm = DataFrame(hcat(test,means_rbm,variance_rbm))

   data = vcat(mean_var_true,mean_var_rbm)
   rename!(data, :x1 => :Setting, :x2 => :log_Mean , :x3 => :log_Var)

   ticks = [0, 0.5, 1, 2]
   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "log_Mean", y = "log_Var", 
      Guide.title("Gene-Wise Mean-Variance Plot"),
      Geom.subplot_grid(layer(Geom.point), layer(intercept=[0.0], slope=[1.0], abline)),
      Guide.xlabel("log10(Mean)"),
      Guide.ylabel("log10(Variance)"))
end

function scvi_genewise_meanvarianceplot(x::Array{Float64,2}, particles::Array{Float64,2};logarithm::Bool)
   
   if logarithm
      # Take means and variances of each row
      variance_rbm = log10.(mapslices(var,particles,dims=1)' .+ 1.0)
      means_rbm = log10.(mapslices(mean,particles,dims=1)' .+ 1.0)

      variance_true = log10.(mapslices(var,x,dims=1)' .+ 1.0)
      means_true = log10.(mapslices(mean,x,dims=1)' .+ 1.0)
   else
      # Take means and variances of each row
      variance_rbm = mapslices(var,particles,dims=1)'
      means_rbm = mapslices(mean,particles,dims=1)'

      variance_true = mapslices(var,x,dims=1)'
      means_true = mapslices(mean,x,dims=1)'
   end
   
   test = repeat(["Ground Truth"], size(means_true,1))
   mean_var_true = DataFrame(hcat(test,means_true,variance_true))

   test = repeat(["scVI Generated Data"], size(means_rbm,1))
   mean_var_rbm = DataFrame(hcat(test,means_rbm,variance_rbm))

   data = vcat(mean_var_true,mean_var_rbm)
   rename!(data, :x1 => :Setting, :x2 => :log_Mean , :x3 => :log_Var)

   ticks = [0, 0.5, 1, 2]
   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "log_Mean", y = "log_Var", 
      Guide.title("Gene-Wise Mean-Variance Plot"),
      Geom.subplot_grid(layer(Geom.point), layer(intercept=[0.0], slope=[1.0], abline)),
      Guide.xlabel("log10(Mean)"),
      Guide.ylabel("log10(Variance)"))
end

function compare_genewise_meanvarianceplot(x::Array{Float64,2}, dbmgen::Array{Float64,2}, scvigen::Array{Float64,2}; logarithm::Bool)
   
   if logarithm
      # Take means and variances of each row
      variance_dbm = log10.(mapslices(var,dbmgen,dims=1)' .+ 1.0)
      means_dbm = log10.(mapslices(mean,dbmgen,dims=1)' .+ 1.0)

      variance_scvi = log10.(mapslices(var,scvigen,dims=1)' .+ 1.0)
      means_scvi = log10.(mapslices(mean,scvigen,dims=1)' .+ 1.0)

      variance_true = log10.(mapslices(var,x,dims=1)' .+ 1.0)
      means_true = log10.(mapslices(mean,x,dims=1)' .+ 1.0)
   else
      # Take means and variances of each row
      variance_dbm = mapslices(var,dbmgen,dims=1)'
      means_dbm = mapslices(mean,dbmgen,dims=1)'

      variance_scvi = mapslices(var,scvigen,dims=1)'
      means_scvi = mapslices(mean,scvigen,dims=1)'

      variance_true = mapslices(var,x,dims=1)'
      means_true = mapslices(mean,x,dims=1)'
   end
   
   test = repeat(["Ground Truth"], size(means_true,1))
   mean_var_true = DataFrame(hcat(test,means_true,variance_true))

   test = repeat(["NB-DBM Generated Data"], size(means_dbm,1))
   mean_var_dbm = DataFrame(hcat(test,means_dbm,variance_dbm))

   test = repeat(["scVI Generated Data"], size(means_scvi,1))
   mean_var_scvi = DataFrame(hcat(test,means_scvi,variance_scvi))

   data = vcat(mean_var_true,mean_var_dbm,mean_var_scvi)
   rename!(data, :x1 => :Setting, :x2 => :log_Mean , :x3 => :log_Var)

   ticks = [0, 0.5, 1, 2]
   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "log_Mean", y = "log_Var", 
      Guide.title("Gene-Wise Mean-Variance Plot"),
      Geom.subplot_grid(layer(Geom.point), layer(intercept=[0.0], slope=[1.0], abline)),
      Guide.xlabel("log10(Mean)"),
      Guide.ylabel("log10(Variance)"))
end

function compare_tsne(tsne_original::Array{Float64,2}, tsne_dbm::Array{Float64,2}, tsne_scvi::Array{Float64,2})
    
   test = repeat(["Ground Truth"], size(tsne_original,1))
   tsne_original = DataFrame(hcat(test,tsne_original))

   test = repeat(["NB-DBM Generated Data"], size(tsne_dbm,1))
   tsne_dbm = DataFrame(hcat(test,tsne_dbm))

   test = repeat(["scVI Generated Data"], size(tsne_scvi,1))
   tsne_scvi = DataFrame(hcat(test,tsne_scvi))

   data = vcat(tsne_original,tsne_dbm,tsne_scvi)
   rename!(data, :x1 => :Setting, :x2 => :TSNE1 , :x3 => :TSNE2)

   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "TSNE1", y = "TSNE2", 
      Guide.title("t-SNE Plot"),
      Geom.subplot_grid(layer(Geom.point)),
      Guide.xlabel("t-SNE 1"),
      Guide.ylabel("t-SNE 2"))
end

function compare_pca(pca_original::Array{Float64,2}, pca_dbm::Array{Float64,2}, pca_scvi::Array{Float64,2})
    
   test = repeat(["Ground Truth"], size(pca_original,1))
   pca_original = DataFrame(hcat(test,pca_original))

   test = repeat(["NB-DBM Generated Data"], size(pca_dbm,1))
   pca_dbm = DataFrame(hcat(test,pca_dbm))

   test = repeat(["scVI Generated Data"], size(pca_scvi,1))
   pca_scvi = DataFrame(hcat(test,pca_scvi))

   data = vcat(pca_original,pca_dbm,pca_scvi)
   rename!(data, :x1 => :Setting, :x2 => :PC1 , :x3 => :PC2)

   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "PC1", y = "PC2", 
      Guide.title("Principal Component Analysis"),
      Geom.subplot_grid(layer(Geom.point)),
      Guide.xlabel("PC 1"),
      Guide.ylabel("PC 2"))
end

function meandispersionplot(x::Array{Float64,2}, rbm::NegativeBinomialBernoulliRBM; libsizenormalization::Bool)

   dispersion = 1 ./ rbm.inversedispersion
   if libsizenormalization
      normalizedcounts = libsizenormalization(x)
   else
      normalizedcounts = x
   end
   
   mean_normalizedcounts = mapslices(mean, normalizedcounts, dims=1)'

   data = DataFrame(hcat(dispersion,mean_normalizedcounts))
   rename!(data, :x1 => :Dispersion, :x2 => :MeanNormalizedCounts)

   p = plot(data, x = "MeanNormalizedCounts", y = "Dispersion", Geom.point,
      Guide.title("Mean-Dispersion Plot"), 
      Guide.xlabel("Mean Normalized Counts"),
      Guide.ylabel("Dispersion Estimate"),
      Theme(major_label_font_size=25pt, minor_label_font_size=20pt)) 
   p
end

function genewise_violinplot(x::Array{Float64,2}, particles::Array{Array{Float64,2},1},gene::Int)
   violinplot = DataFrame(hcat(x[:,gene],particles[1][:,gene]));
   rename!(violinplot, :x1 => :Ground_Truth, :x2 => :Generated_Data);
   violinplot = melt(violinplot);
   p_violin = plot(violinplot, x="variable", y="value", Geom.histogram)
end

function cellwise_violinplot(x::Array{Float64,2}, particles::Array{Float64,2},cell::Int)
   violinplot = DataFrame(hcat(x[cell,:],particles[cell,:]));
   rename!(violinplot, :x1 => :Ground_Truth, :x2 => :Generated_Data);
   violinplot = melt(violinplot);
   p_violin = plot(violinplot, x="variable", y="value", Geom.violin)
end

function genewise_varplot(x::Array{Float64,2}, particles::Array{Array{Float64,2},1})
   truevariance = DataFrame(log.(mapslices(var,x,dims=1)'))
   generatedvariance = DataFrame(log.(mapslices(var,particles[1],dims=1)'))
   
   test = repeat(["Ground Truth"], size(truevariance,1))
   var_true = DataFrame(hcat(test,truevariance,makeunique=true))

   test = repeat(["Generated Data"], size(generatedvariance,1))
   var_generated = DataFrame(hcat(test,generatedvariance,makeunique=true))

   data = vcat(var_true,var_generated)
   rename!(data, :x1 => :Setting, :x1_1 => :log_Var)

   abline = Geom.abline(color="red", style=:dash)
   p2 = plot(data, xgroup="Setting", x = "log_Var",
      Guide.title("Gene-Wise Variance Plot"), 
      Geom.subplot_grid(layer(Geom.point), layer(intercept=[0.0], slope=[1.0], abline)),
      Guide.xlabel("log(Mean)"),
      Guide.ylabel("log(Variance)"),
      Theme(major_label_font_size=25pt, minor_label_font_size=20pt)) 
end

# https://hbctraining.github.io/DGE_workshop/lessons/04_DGE_DESeq2_analysis.html
function coefficientofvariation(particles::Array{Array{Float64,2},1}, rbm::NegativeBinomialBernoulliRBM)
   tmp = DataFrame(mapslices(std,particles[1],dims=1)' ./ mu(rbm))
   rename!(tmp, :x1 => :CV)
   tmp
end

function biologicalcoefficientofvariation(particles::Array{Array{Float64,2},1}, rbm::NegativeBinomialBernoulliRBM)
   tmp = DataFrame(mapslices(std,particles[1],dims=1)' ./ mu(rbm))
   rename!(tmp, :x1 => :CV)
   tmp
end


function zeroinflationplot_compare(x::Array{Float64,2}, particles::Array{Array{Float64,2},1}, particles_nozeroinflation::Array{Array{Float64,2},1})

   # genewise mean-expression vs proportion of zero counts for ground truth
   means_true = log.(mapslices(mean,x,dims=1)' .+ 1.0)
   zeroproportion = Array{Float64,2}((mapslices(sum, x .== 0, dims = 1) / size(x,1))')
   
   plotting_true = DataFrame(hcat(means_true,zeroproportion))
   rename!(plotting_true, :x1 => :Means, :x2 => :Zeroproportions)

   # genewise mean-expression vs proportion of zero counts for generated data
   means_generated = log.(mapslices(mean,particles[1],dims=1)' .+ 1.0)
   zeroproportion_generated = Array{Float64,2}((mapslices(sum, particles[1] .== 0, dims = 1) / size(particles[1],1))')
   
   plotting_generated = DataFrame(hcat(means_generated,zeroproportion_generated))
   rename!(plotting_generated, :x1 => :Means, :x2 => :Zeroproportions)

   # genewise mean-expression vs proportion of zero counts for generated data
   means_generated_noZI = log.(mapslices(mean,particles_nozeroinflation[1],dims=1)' .+ 1.0)
   zeroproportion_generated_noZI = Array{Float64,2}((mapslices(sum, particles_nozeroinflation[1] .== 0, dims = 1) / size(particles_nozeroinflation[1],1))')

   plotting_generated_noZI = DataFrame(hcat(means_generated_noZI,zeroproportion_generated_noZI))
   rename!(plotting_generated_noZI, :x1 => :Means, :x2 => :Zeroproportions)

   data_final = vcat(plotting_true, plotting_generated, plotting_generated_noZI)
   
   test_true = repeat(["Ground Truth"], size(plotting_true,1));
   test_generated = repeat(["ZINB-DBM"], size(plotting_generated,1));
   test_generated_noZI = repeat(["NB-DBM"], size(plotting_generated_noZI,1));
   
   tmp_final = vcat(test_true,test_generated,test_generated_noZI)

   data_final = hcat(tmp_final,data_final)
   rename!(data_final, :x1 => :Dataset)

   p2 = plot(data_final, x = "Means", y = "Zeroproportions", 
      color = "Dataset", Geom.point, Stat.smooth(smoothing=0.75),
      Guide.title("Mean-Zero Relationship"), 
      Guide.xlabel("log(Mean)"),
      Guide.ylabel("Proportion of Zeros"),
      Theme(point_size=1.5mm, major_label_font_size=25pt, 
      minor_label_font_size=20pt, 
      point_label_font_size=20pt, 
      key_title_font_size=20pt, 
      key_label_font_size=15pt)) 
   p2
end

function zeroinflationplot(x::Array{Float64,2}, particles::Array{Float64,2})

   # genewise mean-expression vs proportion of zero counts for ground truth
   means_true = log.(mapslices(mean,x,dims=1)' .+ 1.0)
   zeroproportion = Array{Float64,2}((mapslices(sum, x .== 0, dims = 1) / size(x,1))')
   
   plotting_true = DataFrame(hcat(means_true,zeroproportion))
   rename!(plotting_true, :x1 => :Means, :x2 => :Zeroproportions)

   # genewise mean-expression vs proportion of zero counts for generated data
   means_generated = log.(mapslices(mean,particles,dims=1)' .+ 1.0)
   zeroproportion_generated = Array{Float64,2}((mapslices(sum, particles .== 0, dims = 1) / size(particles,1))')
   
   plotting_generated = DataFrame(hcat(means_generated,zeroproportion_generated))
   rename!(plotting_generated, :x1 => :Means, :x2 => :Zeroproportions)

   data_final = vcat(plotting_true, plotting_generated)
   
   test_true = repeat(["Ground Truth"], size(plotting_true,1));
   test_generated = repeat(["NB-DBM"], size(plotting_generated,1));
   
   tmp_final = vcat(test_true,test_generated)

   data_final = hcat(tmp_final,data_final)
   rename!(data_final, :x1 => :Dataset)

   p2 = plot(data_final, x = "Means", y = "Zeroproportions", 
      color = "Dataset", Geom.point, 
      Guide.title("Mean-Zero Relationship"), 
      Guide.xlabel("log10(Mean)"),
      Guide.ylabel("Proportion of Zeros"),
      Theme(major_label_font_size=25pt, minor_label_font_size=20pt, key_title_font_size=20pt, key_label_font_size=15pt))

   p2
end

function scvi_zeroinflationplot(x::Array{Float64,2}, particles::Array{Float64,2})

   # genewise mean-expression vs proportion of zero counts for ground truth
   means_true = log.(mapslices(mean,x,dims=1)' .+ 1.0)
   zeroproportion = Array{Float64,2}((mapslices(sum, x .== 0, dims = 1) / size(x,1))')
   
   plotting_true = DataFrame(hcat(means_true,zeroproportion))
   rename!(plotting_true, :x1 => :Means, :x2 => :Zeroproportions)

   # genewise mean-expression vs proportion of zero counts for generated data
   means_generated = log.(mapslices(mean,particles,dims=1)' .+ 1.0)
   zeroproportion_generated = Array{Float64,2}((mapslices(sum, particles .== 0, dims = 1) / size(particles,1))')
   
   plotting_generated = DataFrame(hcat(means_generated,zeroproportion_generated))
   rename!(plotting_generated, :x1 => :Means, :x2 => :Zeroproportions)

   data_final = vcat(plotting_true, plotting_generated)
   
   test_true = repeat(["Ground Truth"], size(plotting_true,1));
   test_generated = repeat(["scVI"], size(plotting_generated,1));
   
   tmp_final = vcat(test_true,test_generated)

   data_final = hcat(tmp_final,data_final)
   rename!(data_final, :x1 => :Dataset)

   p2 = plot(data_final, x = "Means", y = "Zeroproportions", 
      color = "Dataset", Geom.point, 
      Guide.title("Mean-Zero Relationship"), 
      Guide.xlabel("log10(Mean)"),
      Guide.ylabel("Proportion of Zeros"),
      Theme(major_label_font_size=25pt, minor_label_font_size=20pt, key_title_font_size=20pt, key_label_font_size=15pt))

   p2
end

function zerospergene(x::Array{Float64,2}, particles::Array{Array{Float64,2},1})

   Ground_Truth = Array{Float64,2}((mapslices(sum, x .== 0, dims = 1) / size(x,1))')
   Generated = Array{Float64,2}((mapslices(sum, particles[1] .== 0, dims = 1) / size(particles[1],1))')

   combined_zeroproportion = DataFrame(hcat(Ground_Truth, Generated))
   rename!(combined_zeroproportion, :x1 => :Ground_Truth, :x2 => :Generated)

   tmp = DataFrames.melt(combined_zeroproportion)

   p = plot(tmp, x="variable", y="value" , Geom.boxplot,
         Guide.title("Distribution of Zeros per Gene"),
         Guide.xlabel(""),
         Guide.ylabel("Proportion of Zeros per Gene"))
   p
end

function bicluster_heatmap(M::Array{Float64,2}; k = 2, y_names = Array[])

      # cluster and sort cols
      c = Clustering.kmeans(M,k)
      idy =  sortperm(assignments(c))
      M = M[:,idy]
   
      # cluster and sort rows
      M_t = convert(Array{Float64,2}, M')
      c2 = Clustering.kmeans(M_t,k)
      idx =  sortperm(assignments(c2))
      M_t = M_t[:,idx]
   
      # generate names of y if non were supplied
      if isempty(y_names)
          y_names = [string("y", i) for i = 1:1:size(M)[2]]
      end
   
      Plots.heatmap(1:size(M_t)[2], convert(Array{String,1}, y_names[idy]), M_t)
end
   
function compare_umap(umap_original::DataFrame, umap_dbm::DataFrame, umap_scvi::DataFrame)

   # Generate color palette
   palette = Scale.color_discrete().f(5)

   # Set up data frame for plotting
   groups = vcat(repeat(["Original"], size(umap_original,1)), repeat(["NB-DBM"], size(umap_dbm,1)), repeat(["scVI"], size(umap_scvi,1)))
   plotting_data = hcat(vcat(umap_original, umap_dbm, umap_scvi), groups)
   x = ["UMAP 1", "UMAP 2", "Cluster", "Groups"]
   names!(plotting_data, Symbol.(x))

   # Plot
   p = plot(plotting_data, xgroup="Groups", x="UMAP 1", y="UMAP 2", color="Cluster", Scale.color_discrete_manual(palette..., levels=["1","2","3","4","5"]), 
      Geom.subplot_grid(Geom.point), Guide.title(""), Guide.xlabel("UMAP 1")
   )

   p
end
