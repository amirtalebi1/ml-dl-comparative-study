# Data Size, Model Complexity, and the Classical ML vs. Deep Learning Trade-off

A controlled comparative study of classical machine learning algorithms and neural networks
(including CNNs and transfer learning), examining how each performs as training data size grows,
on both tabular and image data.

## Research questions

1. As training set size increases, how does the performance of classical, regression-based ML
   models compare to neural networks on the *same* data, with the *same* fixed validation/test
   sets?
2. Do the same patterns hold for image data (CNNs) as for tabular data (dense networks)?
3. In a low-data regime, does transfer learning from a pretrained model outperform training a CNN
   from scratch?
4. What are the compute-cost trade-offs (training time) between model families?


## Datasets

- **Tabular:** UCI Adult Census Income (~48,800 rows, binary classification: income >$50K or not)
- **Image:** CIFAR-10 (60,000 32×32 color images, 10 classes)

## Method

For each data type, training data is split into three nested, stratified subsets — **small**,
**medium**, **large** — while validation and test sets stay fixed across every experiment. This
makes "performance vs. training set size" a controlled, fair comparison.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_eda_and_preprocessing.ipynb` | Load Adult dataset, EDA, preprocessing, create small/medium/large splits |
| 02 | `02_classical_ml_tabular.ipynb` | Logistic Regression (plain + polynomial-feature variant) across all sizes |
| 03 | `03_neural_net_from_scratch.ipynb` | Feedforward NN implemented from scratch in NumPy (forward/backprop) |
| 04 | `04_neural_net_keras_tabular.ipynb` | Keras dense NN with regularization/optimization tuning, across all sizes |
| 05 | `05_cnn_image_classification.ipynb` | CNN from scratch on CIFAR-10, across all sizes |
| 06 | `06_transfer_learning.ipynb` | MobileNetV2 transfer learning vs. from-scratch CNN on the small subset |
| 07 | `07_final_comparison.ipynb` | Aggregates all results, headline charts, discussion |

## Concepts applied

| Concept | Notebook |
|---|---|
| Logistic regression, L2 regularization | 02 |
| Polynomial/interaction feature engineering for non-linear decision boundaries | 02 |
| Bias-variance diagnosis via learning curves, train/val/test methodology | 02 |
| Neural network forward/backward propagation from scratch | 03 |
| Dropout, batch normalization, Adam optimizer, hyperparameter tuning, early stopping | 04 |
| Bias-variance diagnosis via training curves, error analysis | 04, 05 |
| CNN architecture, transfer learning | 05, 06 |




## Results

See `results/` for saved metrics (`.csv`) and figures (`.png`), and `report.md` for the
full write-up and discussion.

