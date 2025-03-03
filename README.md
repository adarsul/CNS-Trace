# CNS-Trace Transcription Factor Binding Evolution Model

This model implements [Stormo's binding model](https://doi.org/10.1093/bioinformatics/16.1.16) to estimate transcription factor (TF) affinity to conserved non-coding sequences (CNS) across evolutionary time. By integrating phylogenetics and sequence analysis, it reconstructs ancestral CNS sequences using FastML, allowing the inference of TF binding site gain, loss, or retention throughout plant evolution.

The approach combines TF binding energy modeling with a neutral evolution framework to assess whether observed binding changes are due to selection or drift. This enables a deeper understanding of how regulatory elements evolve and whether transcription factor binding imposes constraints on non-coding sequence divergence. 

The model is designed to analyze CNS data from multiple plant species and provides insights into the evolutionary forces shaping gene regulation.
