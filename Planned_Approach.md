*Planned ML Approach* 
    * Baselines: Logistic and Linear Regression to establish performance benchmarks and test for linear separability in genre and popularity tasks respectively.
    * Random Forest: Captures non-linear feature interactions (e.g., high "energy" but low "danceability"), while offering noise robustness and feature importance for interpretability.
    It offers robustness against noise and intrinsic feature importance scoring for interpretability. 
    * Support Vector Machines(SVM): For classification, we will test SVMs with RBF kernels to separate genres that may overlap such as “Pop” and “Dance”.
    * XGBoost: We will implement Gradient Boosting to minimize bias and further improve prediction accuracy on the regression task by sequentially correcting the errors of previous weak learners. 