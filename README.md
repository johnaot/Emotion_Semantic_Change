Emotion Semantic Change
======

This repository contains replication code and data that accompany the following
paper: Xu, A., Stellar, J.E., and Xu, Y. (2021) Evolution of
emotion semantics. Cognition.

## Dependencies

### Software

Required `jupyter`
Required `python>=3.7`
Required packages:
```
numpy
scipy
pandas
scikit-learn
statsmodel
seaborn
matplotlib
```

### Data

- HistWords, available from https://nlp.stanford.edu/projects/histwords/

To run the following notebooks, 
1. In `constants.py`, modify
    - `npy_path_eng` with the path to the folder containing npy files of English sgns vectors from HistWords
    - `vocab_path_eng` with the path to the folder containing the pkl file of English vocabulary from HistWords
    - `pos_path_eng` with the path to the folder containing pkl files of English pos from HistWords
    - `freq_path_eng` with the path to the pkl file of English frequency data from HistWords
    - repeat the above bullet point for French

## Execution

### Illustrating the hypothesis

To generate the illustration for our main hypothesis, run `jupyter notebook hypothesis`

### Analyses of emotion semantic change and bird names

To run regression analyses on semantic change of emotion words and create figure 3, run `jupyter notebook analyses`

### Historical analysis of prototypicality judgement

To perform the analysis described in Supplementary Information (SI) section 1, run `jupyter notebook SI_section_1`

### Evaluation of the nearest-neighbour measure

To perform analyses described in SI section 2, run `jupyter notebook SI_section_2`

### Category-bounded analyses of emotion semantic change

To perform analyses described in SI section 3.1, run `jupyter notebook analyses`

### Other factors of emotion semantic change

To perform analyses described in SI section 3,2, run `jupyter notebook analyses_other`

### Analysis using empirical emotion prototypicality ratings

To perform the analysis described in SI section 3.3, run `jupyter notebook analyses_other`

