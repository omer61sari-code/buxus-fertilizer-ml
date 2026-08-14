[README.md](https://github.com/user-attachments/files/31089162/README.md)
# Buxus sempervirens Fertilizer Classification — Python Analysis

Python workflow for the revised machine-learning analysis of *Buxus sempervirens*
fertilizer-treatment groups.

## Analysis design
- 70 plant-level observations (7 treatment classes × 10 observations)
- Six aboveground morphological predictors
- Stratified 70:30 split: 49 training + 21 untouched hold-out observations
- Five-fold stratified cross-validation within the training subset
- Classifiers: IBk, PART-like rule learner, J48-style decision tree, MLP, and Gaussian Naive Bayes
- Random Forest used separately for feature importance and SHAP interpretation

## Important
The revised analysis is **Python-only**; WEKA/Java is not required.

PART is implemented as a reproducible Python sequential rule learner inspired by
the PART partial-tree/rule approach; it is not a byte-for-byte reproduction of
Weka PART. The J48-style model is implemented with scikit-learn using an entropy
criterion and should not be described as the exact Weka J48 implementation.

## Input
Put the validated dataset in the repository root as:

`ML_70_current_input.csv`

Required columns:

`Applications, Plant_height, Plant_width, Number_of_shoots, Shoot_length, Leaf_width, Leaf_length`

The script expects 70 observations and 10 observations in each class:
Control, Potassium, Nitrogen, Vermicompost, Bacteria, Mycorrhizal, Sheep wool.

The raw manuscript dataset is not included unless the authors decide to make it
publicly available. An optional Excel file may be placed at:
`input/Birlesik_veriler.xlsx`

## Run
```bash
pip install -r requirements.txt
python buxus_fertilizer_ml.py
```

The script generates `Results_ML_Revised/` and `Figures_ML_Revised/`.

Outputs include training accuracy, five-fold CV accuracy, hold-out accuracy and
balanced accuracy, weighted precision/recall/F1, Cohen's kappa, ROC-AUC,
hold-out predictions, Random Forest feature importance, and SHAP outputs.

A fixed random seed (42) is used for reproducibility.
