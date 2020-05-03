# Medicine Recommender in Python

## Aim:
Recommend medicine by wound properties.

## Data:
The wound properties and the corresponding purchased medicine.

## Method:
1. Generating a unique ID for wound properties
2. Applying matrix factorization method (WALS algorithm) to decompose generated ID and medicine-ID

## Steps:
Following [this google recommender example](https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-deploy)
N.B:
1. Remember to `export PATH=/home/rachel/miniconda2/bin:$PATH` when install miniconda2
2. After hypertuning,
``` bash
./mltrain.sh local ../data/MSE_OAE_Cat_ML.csv \
--use-optimized --output-dir ${BUCKET} --delimiter ,
```

