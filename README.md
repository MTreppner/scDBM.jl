# scDBM.jl  
This is the implementation for **"Making many out of few: deep generative models for single-cell RNA-sequencing data"**.

## Abstract  
Deep generative models, such as variational autoencoders (VAEs) or deep Boltzmann machines (DBM), can generate anarbitrary number of synthetic observations after being trained on an initial set of samples. This has mainly been investigated for imaging data but could also be useful for single-cell transcriptomics (scRNA-seq). A small pilot study could be used for planning a full-scale study by investigating planned analysis strategies on synthetic data with different sample sizes. It is unclear whether synthetic observations generated based on a small scRNA-seq dataset reflect the properties relevant for subsequent data analysis steps.  
We specifically investigated two deep generative modeling approaches, VAEs and DBMs.  First, we considered single-cell variational inference (scVI) in two variants, generating samples from the posterior distribution, the standard approach, or the prior distribution.  Second, we propose single-cell deep Boltzmann machines (scDBM). When considering the similarity of clustering results on synthetic data to ground-truth clustering, we find that the scVI (posterior) variant resulted in high variability, most likely due to amplifying artifacts of small data sets.  All approaches showed mixed results for cell types with different abundance by overrepresenting highly abundant cell types and missing less abundant cell types. With increasing pilot dataset sizes, the proportions of the cells in each cluster became more similar to that of ground-truth data. We also showed that all approaches learn the univariate distribution of most genes, but problems occurred with bimodality. Overall, the results showed that generative deep learning approaches might be valuable for supporting the design of scRNA-seq experiments.

## Main requirements  
Julia: 1.1.0

## References  

The package is based on the implementation 'BoltzmannMachines.jl'

[1] Lenz, Stefan, Moritz Hess, and Harald Binder. "Unsupervised deep learning on biomedical data with BoltzmannMachines. jl." bioRxiv (2019): 578252.

