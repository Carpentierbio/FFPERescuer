
# FFPERescuer

## **Introduction**

The repository provides a deep learning framework for reconstructing formalin-fixed paraffin-embedded (FFPE)-derived gene expression data from RNA-sequencing.

FFPE tissue samples often show RNA degradation due to fixation and storage processes; consequently, the RNA-sequencing data from FFPE samples for molecular analysis of cancer are not comparable to those obtained from fresh frozen (FF) tissues. We developed a deep learning (DL)-based framework that used 9568 FF primary tumor samples from The Cancer Genome Atlas across 28 cancer types and the given dataset to reconstruct gene expression profiles in a given FFPE dataset.

<img src="https://github.com/Carpentierbio/FFPERescuer/blob/main/image/FFPERescuer.jpg" alt="workflow" width="100%">

**Figure 1 Unsupervised domain adaptation for FFPE-derived RNA-seq profile reconstruction.** 

Schematic diagram of reconstructing FFPE-derived gene expression profile including training two networks and combining them. In the first step, the pan-cancer gene expression profile from FF tissue (source domain) in TCGA was used to train an CNN-based autoencoder network (comprising FF-encoder and FF-decoder). Then, a part the FF pan-cancer gene expression profile with fewer number of genes was used to train the partial FF-encoder. In the third step, partial FF-encoder and FF-decoder were combined to create FFPERescuer, amis for reconstructing FFPE-derived gene expression profile (target domain). FFPERescuer takes a small number of input genes while generates high-dimensional gene expression outputs. CNN, convolutional neural network.

## **Prerequisites**

Python >= 3.5
