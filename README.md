# FFPE-rescuer

FFPE-rescuer is a deep learning-based framework developed for recovery of FFPE-derived RNA-sequencing data.

Figure 1 Constructing a recovery network using CNN for FFPE-derived RNA-seq data. Schematic diagram of recovering FFPE-derived gene expression including training two networks and combining them. In the first step, the pan-cancer gene expression profile from FF tissue in TCGA was used to train an autoencoder network (comprising FF-encoder and FF-decoder). Then, a sub-set of the FF pan-cancer gene expression profile with fewer number of genes was used to train the partial FF-encoder. In the third step, partial FF-encoder and FF-decoder were combined to create FFPE-rescuer, which was used for recovering FFPE-derived gene expression profile. FFPE-rescuer takes a small number of input genes while output a profile with a large number of genes.
