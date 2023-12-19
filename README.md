# 3ASC : Novel Variant Prioritization for Jointly prioritizing SNV and CNV and classifying reportability in rare disease
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

**Figure 1**. Illustrated the architecture of 3ASC leveraging MIL
![3ASC](data/images/figure1_3asc.png) 


**3ASC** is designed to incorporate both Single Nucleotide Variants (SNVs) and Copy Number Variants (CNVs). We initially embedded input variant features into a low-dimensional space with neural network encoders. To address the different dimensionalities of SNV and CNV features, distinct encoders were constructed for each type of variant. Then, we additionally attached the shared embedding layer from SNV and CNV encoders. This transformation projected the encoded both SNVs and CNV features onto the manifold, where they could be semantically equivalent. Since it is essential to aggregate these embedded instance vectors to predict reportability of patients, we applied this using an attention-based pooling method with attention weights derived from a gated attention layer. Then, we devise our model enable to classify the reportability with bag label classifier. For variant prioritization, we leverage these weights which include the importance in predicting patient label with instance label classifier. Plus, to achieve optimal variant prioritization, we additionally adopted RankNet which is a differentiable function of the deep learning 



## Performance: Reportability & Variant Prioritization*
- Reportability classification: **0.92** of AUROC for binary classification
- Joint prioritization for SNV and CNV: Top1 (0.739), Top 3(0.9315), **Top 5(0.9682)** 
- CNV only prioriziation: Top 1 (0.7227), Top 3 (0.8814), **Top 5 (0.9498)**

*Assessed with 5-Fold cross validation

**Figure 2**. Variant prioritization performance. A) AUROC for classification of patient reportability; B) Prioritization performance for casual variant among patients with one of SNVs and CNV or both, C) Prioritization performance for casual variant among patients with CNV only
![Performance](data/images/figure2_performance.png)


## Effectiveness of RankNet 
For performance of variant prioritization for casual SNV and CNV variants, the model with pointwise RankNet statically significantly outperformed (P=0.007; respectively).

**Figure 3**. Ablation test of pointwise RankNet for variant prioritization

<img src="data/images/figure3_ranknet.png" width="500">

## Benchmark performance
Our model overall outperforms than Exomiser, and LIRICAL. At the top 1, 3ASC showed 74.4% of recall for casual variant, compared to 59.9% of recall in Exomiser, and 53.7% of recall in LIRICA respectively

<img src="data/images/figure4_benchmark.png" width="500">

## Feature importance
For both reportability classification, symptom similarity between patient’s phenotypes and disease phenotypes showed most decreased AUROC as 0.191 of AUROC (Figure 5-A)

**Figure 5**. Top 15 of feature importance of 3ASC; A) Difference in AUROC between original data and shuffled data for reportability classification; B) Difference in hit rate at 5 between original data and shuffled data for variant prioritization. (All feature importance is attached in the supplementary document 2).

<img src="data/images/figure5_feature_importance.png" width="800">

