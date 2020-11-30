# Deep Representations of Somatic Mutations in Cancer Samples

This repository describes in detail a methodology for the embedding of somatic mutations profiles in human cancer. With the development of advanced machine learning methods for precision medicine, it is increasingly important to ensure that these novel tools are made availale to researchers whose expertise does not necessarily lie in deep learning and representation learning methods. Deep learning models have the capability to integrate several data sources in a unified framework, and how this data is represented constitutes a significant portion of the experimental design involved.

The present reprository offers a complete implementation of a deep learning model for the embedding of somatic mutations profiles, as well as pre-computed embeddings for a selected set of samples from the COSMIC database, that are readily available.

The following sections describe the approach used to learn representations of cancer samples, the experimental design, and the steps necessary to use or reproduce the results.

## Dataset

The following model is trained using the [Sanger Catalogue of Somatic Mutations in Cancer (COSMIC)](https://cancer.sanger.ac.uk/cosmic) database, which is a curated source of information on somatic mutations in human cancer. We are specifically interested in its vast amount of genome-wide screen data, which is a rich resource for the training of deep learning model.

More specifically, we make use of the "COSMIC Mutation Data (Genome Screens)" dataset that is available in version v92 of the project, which offers information on individual somatic mutations present in cancer samples. To reproduce the results shown in this repository, it is necessary to download and extract this dataset in the root folder of the project.

With the mutation data, we aim to encode each cancer sample as a binary vector where each component signals whether a mutation in the corresponding gene is observed. More accurately, a cancer sample is represented by two such vectors, as we follow the design choices of Palazzo et al. (2019), with which we consider deleterious and non-deleterious mutations separately.

The somatic mutations in the dataset are filtered according to the following quality control criteria:
- labelled as "Confirmed somatic variant"
- they must come from a genome-wide screen study
- mapped onto the GRCh38 reference genome
- the mutations must correspond to a gene (only coding mutations), i.e. there is an HGNC identifier available
- we exclude hypermutated samples (>1000 mutation) according to previous works in the literature (Vogelstein et al. (2013))
- underrepresented genes are removed (gene mutations that appear fewer than 100 times)

The resulting processed dataset includes 5.491.567 somatic mutations in 12.449 unique genes from 26.355 cancer samples.

## Model

The model chosen for the embedding of somatic mutation profiles is a bimodal Information-maximizing Variational Autoencoder (InfoVAE), derived from the work of Zhao et al. (2019). The structure of the model, inspired by the work of Palazzo et al. (2019), considers separately deleterious and non-deleterious mutations in the input stage. Contrary to the original work, we split the mutations into these two categories based on the FATHMM prediction (Shabib et al. (2015)) included in the COSMIC dataset.

<img src="readme_images/MMD_VAE.png"
     alt="Bimodal MMD-VAE model"
     height=439 width=572 />

The InfoVAE model relies om Maximum Mean Discrepancy to regularize the latent representations learned by the autoencoder model, which reportedly improves on the quality of the embeddings compared to standard VAEs.

Additionally, we implemented a beta VAE model that is closer to the original method of Palazzo et al. (2019)

<img src="readme_images/Bimodal VAE.png"
     alt="Bimodal VAE model"
     height=439 width=572 />

We used a Beta Variational Autoencoder with warmup (Sønderby et al. (2016)), as this configuration allows the model to focus on the reconstruction of the data for the first few epochs while gradually introducting the variational objective.

## Phenotype Prediction Evaluation

To evaluate how well the learned embeddings describe the phenotype of the cancer samples, we adopted a similar approach to the work of Amar et al. (2017).

Using information from [Disease Ontology](https://disease-ontology.org/), we constructed a hierarchy of cancer subtypes, based on their site of origin and histology, which results in 56 categories. These categories were chosen so that there would be a significant enough number of samples both in the positive and negative class for each of them. If the overlap between a parent and child category was too large, we considered only the parent class.

<img src="readme_images/subtype_hierarchy.png"
     alt="Hierarchy of cancer subtypes"
     height=500 width=1200 />

Each cancer sample is assigned a positive (1) or negative (0) class for each category, while respecting the hierarchical structure. Some of the categories considered (e.g. adenocarcinoma) consist of purely histological subtypes, while most other categories are related to the primary site of origin of the cancer sample. Of the 26.355 samples, 25.645 were assigned at least one category, while the remaining samples were excluded from this analysis.

The embeddings analysed were obtained through the use of an InfoVAE model with latent dimension of 50, thus achieving a compression factor of 498 (12449+12449 -> 50).

We first used a UMAP projection to visualize and observe the patterns shown by the embeddings according to their categories. The plots, available in the folder "embeddings/plots_mmd_vae", show an interesting distribution of the various cancer subtypes in the embedding manifold. Interestingly, it appears that there is a concentration of "liquid tumours" in the right-hand side of the UMAP representation, while gastrointestinal and thoracic cancers are grouped mostly in the left branch.

<img src="readme_images/embeddings_visualization_mmd.png"
     alt="UMAP embeddings"
     height=400 width=800 />

To evaluate how well the embeddings capture the phenotypic information of the samples, we trained a Support Vector Machine (SVM) classifier for each of the 56 categories to evaluate the prediction accuracy on a validation set. Given the imbalance of categories when using the full dataset, for each cancer subtype, where possible, a set of negative samples equal in size to the positive samples was randomly selected. This smaller subtype-specific dataset was split into training and test sets, and used for the classifier evaluations, by applying a 5-fold cross-validation scheme. The results reported in the following table represent the median value of the ROC-AUC for the SVM classifiers across folds.

As evidenced by the results, the compressed embeddings capture a lot of the information contained in the deleterious mutation profile, while often remaining inferior to the non-deleterious mutations profile for site identification. This suggests that, while tissue-specific mutation pattern might be more effective for site classification, it might not be the determining factor in analysing the characteristics of cancer samples. This is in-line with the paradigm shift in cancer study that prioritizes biomarker information rather than tissue-specific information.
These results show that the learned embedding improves classification performance in 24 out of 56 categories, while mantaining a significant amount of information from the original mutational profile in the other cases. Interestingly, the performance of the embeddings varies significantly between cancer subtypes. Some cancer types (e.g. gastrointestinal system cancer) show a marked improvement when using the embeddings, while others (e.g. atrocytoma) show a decrease in performance.

| Cancer Subtype | N. Pos. Samples | Embeddings AUC | Xdel AUC | Xnd AUC |
| --- | --- | --- | --- | --- |
| hepatocellular_carcinoma | 849 | <b>0.73</b> | 0.69 | 0.65 |
| liver_cancer | 1315 | 0.67 | 0.67 | <b>0.68</b> |
| colon_adenocarcinoma | 575 | <b>0.78</b> | 0.77 | 0.76 |
| colon_carcinoma | 582 | <b>0.78</b> | 0.75 | 0.76 |
| colon_cancer | 633 | <b>0.79</b> | 0.75 | 0.76 |
| colorectal_cancer | 633 | 0.79 | <b>0.8</b> | 0.75 |
| intestinal_cancer | 1814 | <b>0.81</b> | 0.77 | 0.74 |
| esophageal_carcinoma | 969 | <b>0.8</b> | 0.79 | 0.76 |
| stomach_cancer | 906 | <b>0.81</b> | 0.73 | 0.75 |
| biliary_tract_cancer | 613 | 0.62 | 0.61 | <b>0.65</b> |
| gastrointestinal_system_cancer | 6255 | <b>0.8</b> | 0.67 | 0.66 |
| integumentary_system_cancer | 668 | 0.63 | <b>0.77</b> | 0.55 |
| diffuse_large_B_cell_lymphoma | 256 | 0.68 | <b>0.81</b> | 0.75 |
| lymphoma | 760 | <b>0.74</b> | <b>0.74</b> | <b>0.74</b> |
| acute_myeloid_leukaemia | 942 | 0.84 | <b>0.87</b> | 0.84 |
| T_cell_acute_lymphoblastic_leukemia | 359 | 0.81 | <b>0.83</b> | 0.76 |
| B_cell_acute_lymphoblastic_leukemia | 251 | <b>0.78</b> | 0.75 | 0.65 |
| lymphoid_leukemia | 778 | 0.75 | <b>0.81</b> | 0.7 |
| chronic_lymphocytic_leukemia | 798 | <b>0.83</b> | 0.6 | 0.56 |
| leukemia | 3059 | <b>0.81</b> | 0.73 | 0.74 |
| immune_system_cancer | 3847 | <b>0.78</b> | 0.75 | 0.74 |
| breast_ductal_carcinoma | 122 | 0.63 | <b>0.77</b> | 0.69 |
| breast_cancer | 1819 | 0.62 | <b>0.69</b> | 0.66 |
| lung_adenocarcinoma | 778 | 0.57 | 0.63 | <b>0.65</b> |
| lung_squamous_cell_carcinoma | 506 | 0.8 | 0.79 | <b>0.84</b> |
| lung_cancer | 1639 | 0.6 | <b>0.65</b> | <b>0.65</b> |
| respiratory_system_cancer | 2157 | <b>0.69</b> | 0.63 | 0.66 |
| thoracic_cancer | 3976 | 0.63 | 0.63 | <b>0.64</b> |
| ovarian_cancer | 643 | 0.65 | <b>0.67</b> | 0.65 |
| cervical_cancer | 253 | 0.69 | 0.77 | <b>0.79</b> |
| endometrial_cancer | 399 | 0.71 | <b>0.82</b> | 0.68 |
| female_reproductive_organ_cancer | 1297 | 0.67 | <b>0.71</b> | 0.64 |
| prostate_adenocarcinoma | 817 | 0.64 | 0.63 | <b>0.65</b> |
| male_reproductive_organ_cancer | 1135 | 0.65 | <b>0.66</b> | 0.63 |
| reproductive_organ_cancer | 2432 | 0.62 | <b>0.63</b> | 0.6 |
| pancreatic_ductal_carcinoma | 737 | 0.78 | <b>0.87</b> | 0.57 |
| pancreatic_cancer | 1052 | 0.72 | <b>0.81</b> | 0.61 |
| thyroid_gland_cancer | 943 | <b>0.82</b> | 0.81 | 0.55 |
| adrenal_gland_cancer | 318 | <b>0.88</b> | 0.81 | 0.76 |
| endocrine_gland_cancer | 2403 | <b>0.73</b> | 0.69 | 0.63 |
| clear_cell_renal_cell_carcinoma | 1161 | 0.75 | <b>0.8</b> | 0.71 |
| kidney_cancer | 1938 | <b>0.72</b> | <b>0.72</b> | 0.68 |
| urinary_bladder_cancer | 501 | 0.75 | 0.83 | <b>0.84</b> |
| urinary_system_cancer | 2439 | 0.63 | <b>0.71</b> | 0.68 |
| brain_glioma | 1451 | 0.64 | <b>0.79</b> | 0.66 |
| brain_cancer | 1820 | 0.66 | <b>0.71</b> | 0.61 |
| central_nervous_system_cancer | 2171 | 0.67 | <b>0.73</b> | 0.62 |
| nervous_system_cancer | 2703 | <b>0.71</b> | 0.69 | 0.67 |
| connective_tissue_cancer | 505 | <b>0.69</b> | 0.53 | 0.58 |
| squamous_cell_carcinoma | 2273 | <b>0.81</b> | 0.69 | 0.66 |
| adenocarcinoma | 4079 | <b>0.73</b> | 0.65 | 0.66 |
| carcinoma | 16377 | <b>0.79</b> | 0.75 | 0.71 |
| astrocytoma | 972 | 0.59 | <b>0.74</b> | 0.65 |
| malignant_glioma | 1656 | 0.63 | <b>0.75</b> | 0.64 |
| cancer | 25233 | <b>0.56</b> | 0.54 | 0.54 |
| benign_neoplasm | 416 | 0.55 | 0.63 | <b>0.64</b> |

Overall, the performance shown and the almost 500-fold compression of the data suggest that the use of deep representations of somatic mutation profiles is beneficial for downstream analyses.


## Reproducing the Results

To reproduce the results of this project, download and extract the "COSMIC Mutation Data (Genome Screens)" dataset (version v92) in the root folder of the project, and then execute the following files in order:
1. _COSMIC Data Processing.ipynb_ notebook
2. _mmd_train_model.py_
3. _embed_cancer_samples.py_
4. _plot_embeddings.py_

The files require a python 3 setup with the packages listed in _requirements.txt_

## Embeddings Availability

Precomputed embeddings are available in the _embeddings/mmd_vae_embeddings.csv_ and _embeddings/vae_embeddings.csv_ files for the InfoVAE and BetaVAE respectively, which contain the 50-dimensional representations for the 25.645 samples for which a cancer subtype was assigned.

## Files Description
- _COSMIC Data Processing.ipynb_: this jupyter  notebook contains all the proedures to process the raw Mutation Data dataset into the formats required by the VAE model

## References
- Amar, D., Izraeli, S., & Shamir, R. (2017). _Utilizing somatic mutation data from numerous studies for cancer research: proof of concept and applications._ Oncogene, 36(24), 3375-3383
- Palazzo, M., Beauseroy, P., & Yankilevich, P. (2019). _A pan-cancer somatic mutation embedding using autoencoders._ BMC bioinformatics, 20(1), 655
- Shihab, H. A., Rogers, M. F., Gough, J., Mort, M., Cooper, D. N., Day, I. N., ... & Campbell, C. (2015). _An integrative approach to predicting the functional effects of non-coding and coding sequence variation._ Bioinformatics, 31(10), 1536-1543.
- Sønderby, C. K., Raiko, T., Maaløe, L., Sønderby, S. K., & Winther, O. (2016). _Ladder variational autoencoders._ In Advances in neural information processing systems (pp. 3738-3746).
- Vogelstein, B., Papadopoulos, N., Velculescu, V. E., Zhou, S., Diaz, L. A., and Kinzler, K. W. (2013). _Cancer genome landscapes_
- Zhao, S., Song, J., & Ermon, S. (2019, July). _Infovae: Balancing learning and inference in variational autoencoders._ In Proceedings of the aaai conference on artificial intelligence (Vol. 33, pp. 5885-5892).