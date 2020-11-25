# Deep Representations of Somatic Mutations in Cancer Samples

This repository describes in detail a methodology for the embedding of somatic mutations profiles in human cancer. Its aim specifically is to offer this tool to any researcher that might want to include such a procedure in their work.
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

The resulting processed dataset includes 5.491.567 somatic mutations from 26.355 cancer samples

## Model


## Phenotype evaluation



## Reproducing the Results


## Files Description


## References
- Palazzo, M., Beauseroy, P., & Yankilevich, P. (2019). _A pan-cancer somatic mutation embedding using autoencoders._ BMC bioinformatics, 20(1), 655
- Vogelstein, B., Papadopoulos, N., Velculescu, V. E., Zhou, S., Diaz, L. A., and Kinzler, K. W. (2013). _Cancer genome landscapes_