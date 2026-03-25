# Spotify Track Analysis: Predicting Genre & Popularity from Raw Audio  
**MSE 446 Final Project – Team 27**  

---

## The "Why" Behind the Project
Music recommendations usually take the easy route by relying on artist fame, metadata, or user behavior.  
[cite_start]For this project, we wanted to see how far we could get using *only* intrinsic audio features—like acousticness, tempo, and energy [cite: 273-275].

We started with a Kaggle dataset of 114,000 tracks, but quickly realized that 125 hyper-specific subgenres create too much noise and class overlap.  
[cite_start]We built a data pipeline to remap these into 10 distinct macro-genres [cite: 277-284], balanced the classes, and pushed the data through everything from simple linear baselines to a highly tuned XGBoost pipeline.

**Our main takeaway:** While rigid genre boundaries are highly subjective (Pop, Rock, and Electronic share massive overlap), our tree-based models proved that raw audio features are incredibly effective at predicting a track's "vibe" via Top-3 accuracy (83.5%).

---

## How the Codebase is Organized
[cite_start]To make grading straightforward, the entire execution flow is orchestrated from a single Jupyter Notebook [cite: 196-200]. The underlying logic is broken out into modular Python scripts.

- `MSE446_Final_Project.ipynb` ➔ **Start here.** This notebook imports our modules, runs the models sequentially, and generates the exact visualizations used in our presentation.
- `ml_common.py` ➔ The backbone. Handles feature column definitions, data loading, and consistent train/test splits across all models.
- `build_dataset.py` ➔ The data prep script.  
  *(Note: You do not need to run this; the cleaned `.csv` datasets are already included in the repo so you don't need Spotify API credentials).*

### The Models:
- `baseline_models.py` → Logistic & Linear Regression (to demonstrate non-linearity)
- `random_forest_model.py` → Random Forest Classifier & Regressor
- `svm_model.py` → SVM-RBF Classifier (evaluated on a downsampled dataset)
- `xgboost_model.py` → Baseline Gradient Boosting
- `high_performance_genre_pipeline.py` → Tuned XGBoost pipeline optimizing Top-3 accuracy

- `/model_outputs/` ➔ Output folder for confusion matrices and feature importance CSVs

---

## Environment Setup
We used `uv` for package management to keep the environment reproducible. The core stack relies on Python 3.13+, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, and `seaborn`.

### 1. Install `uv` (if needed)
- **Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- **macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Sync dependencies
Open your terminal in the project root and run:
```bash
uv sync
```

---

## Reproducing Our Results
Once the environment is synced, you can recreate the entire pipeline in a few steps.

### 1. Launch Jupyter Lab
```bash
uv run jupyter lab
```

### 2. Open the notebook
`MSE446_Final_Project.ipynb`

### 3. Run everything
- Click **Run → Run All Cells**

---

## What to Expect
- The notebook loads the dataset and displays class balance.
- Runs linear baselines.
- Trains Random Forest and SVM models.
- Executes the high-performance XGBoost pipeline.
- Outputs:
  - Confusion matrix heatmap  
  - Feature importance bar charts (inline)