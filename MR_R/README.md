# Medicine Recommender in R

## Aim:
Recommend medicine by wound properties.

## Data:
The wound properties and the corresponding purchased medicine.

## Method:
1. Generating a unique ID for wound properties
2. Applying matrix factorization method (WALS algorithm) to decompose generated ID and medicine-ID

Adopted from [this google recommender example](https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-deploy)

## File struture:
- data
- savedmodel:
- [preprocess_data.R](preprocess_data.R): preprocessing data
- [bestModel.R](bestModel.R):
  - Training
  - saving the best model under [savedmodel](savedmodel) directory
  - generating [pred.json](pred.json) for predicting

## Notes
Gcloud commond for predicting is: 
```bash
gcloud ml-engine predict --model="collaborative_filter_rec"\
 --version="collaborative_filter_rec_2"\
 --json-instances='pred.json'\
 --format=json
```
