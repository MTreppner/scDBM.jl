# scDBM.jl  
This is the implementation for **"Generating Synthetic Single-Cell RNA-Sequencing Data from Small Pilot Studies using Deep Learning"**.

## Abstract  
**Motivation:** When designing experiments, it is often advised to start with a small pilot study for determining
the sample size of subsequent full-scale investigations. Recently, deep learning techniques for singlecell
RNA-sequencing data have become available, that can uncover low-dimensional representations of
expression patterns within cells and could potentially be useful also with pilot data. Here, we specifically
examine the ability of these methods to learn the structure of data from a small pilot study and subsequently
generate synthetic expression datasets useful for planning full-scale experiments.

**Results:** We investigate two deep generative modeling approaches. First, we consider established singlecell
variational inference (scVI) techniques in two variants, either generating samples from the posterior
distribution, which is the approach proposed so far, or from the prior distribution. Second, we propose
a novel single-cell deep Boltzmann machine (scDBM) technique, which might be particularly suitable for
small datasets. When considering the similarity of clustering results on synthetic data to ground-truth
clustering, we find that the scVI (posterior) variant exhibits high variability. In contrast, expression patterns
generated from the scVI (prior) variant and scDBM perform better. All approaches show mixed results
when considering cell types with different abundance by sometimes overrepresenting highly abundant cell
types and missing less abundant cell types. Taking such tradeoffs in performance between approaches
into account, we conclude that for making inference from a small pilot study to a larger experiment, it is
advantageous to use scVI (prior) or scDBM, as scVI (posterior) tends to produce strong signals in synthetic
data that are not justified by the original data. The proposed novel scDBM approach seems to have a slight
advantage for very small pilot datasets. More generally, the results show that generative deep learning
approaches might be valuable for supporting the experimental design stage of scRNA-seq experiments.

## Main requirements  
Julia: 1.1.0

## References  

The package is based on the implementation 'BoltzmannMachines.jl'

[1] Lenz, Stefan, Moritz Hess, and Harald Binder. "Unsupervised deep learning on biomedical data with BoltzmannMachines. jl." bioRxiv (2019): 578252.

