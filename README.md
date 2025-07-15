# Wind Turbine Failure Prediction using Neural Networks – ReneWind Project

## Problem Statement

The objective of this project is to build and tune neural network classification models to predict **failures in wind turbines**. Early detection of failures can reduce maintenance costs by enabling timely **repairs** and avoiding **replacements**.

### Nature of Predictions:

- **True Positives (TP)**: Actual failures correctly predicted – **Repair Cost**
- **False Negatives (FN)**: Failures missed by model – **Replacement Cost**
- **False Positives (FP)**: Incorrect failure predictions – **Inspection Cost**

**Cost Hierarchy:**  
*Inspection Cost* < 🔧 *Repair Cost* < 💣 *Replacement Cost*

---

## Data Dictionary

- **Predictors**: 40 continuous sensor features (not named explicitly)
- **Target Variable**:
  - `1` – Failure occurred
  - `0` – No failure

Data is split into **train.csv** and **test.csv**.

---

## Libraries & Tools Used

- **Python**
- **EDA & Preprocessing**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- **Neural Networks**: `TensorFlow`, `Keras`
- **Evaluation Metrics**: `confusion_matrix`, `recall_score`, `precision_score`, `f1_score`, `classification_report`
- **Regularization & Optimization**:
  - Dropout, Batch Normalization
  - L1, L2 Regularization
  - Adam, SGD, Adagrad Optimizers
  - Weight Initialization: He, Glorot, RandomNormal

---

## Best Performing Model: **Model 13**

| Metric     | Training | Testing |
|------------|----------|---------|
| **Recall** | 0.9062   | 0.9064  |

### Configuration:

- **Architecture**:
  - Input Layer → Dense(80, ReLU) → BatchNorm → Dropout(0.3)
  - Dense(160, ReLU) → BatchNorm
  - Output Layer: Sigmoid
- **Optimizer**: Adam (`learning_rate = 1e-5`)
- **Loss Function**: Binary Crossentropy
- **Regularization**: L1 & L2
- **Weight Initialization**:
  - Layer 1: HeNormal  
  - Layer 2: GlorotUniform  
  - Output: RandomNormal
- **Training**:
  - Epochs: 50
  - Batch Size: 32
  - Class Weights: Used to handle imbalance

---

## Actionable Insights

- Gradual tuning (increasing epochs before batch size) helped smoothen metric curves.
- Adam Optimizer performed better than SGD and Adagrad, especially with a low learning rate.
- Techniques like **Dropout**, **Batch Normalization**, and **Regularization** effectively reduced overfitting.
- Weight initialization impacted convergence and final metrics.

---

## Business Recommendations

- Focus on monitoring the following **key sensor features**: `V7`, `V11`, `V15`, `V16`, `V21`, `V28`, `V34`
- Improving accuracy in detecting failures through these features can reduce high replacement costs.
- Prioritize **preventive maintenance** over post-failure replacement by implementing real-time monitoring based on this model.
- Use model output as a **risk classification tool** for scheduling inspections efficiently.

---

## Files Included

- `rene_wind_nn.ipynb` – Jupyter Notebook with complete modeling pipeline
- `rene_wind_nn.html` – Exported version for viewing
- `README.md` – Documentation file
-  Input Data - Train.csv and Test.csv

---

## How to Run

1. Clone or download this repo
2. Open `rene_wind_nn.ipynb` in Google Colab or Jupyter Notebook
3. Install dependencies (if needed):  
   `pip install pandas numpy scikit-learn tensorflow matplotlib seaborn`
4. Run cells in sequence

---

## Future Scope

- Explore **CNN or LSTM** models if time-series nature of sensors is included.
- Integrate the model with **real-time turbine data streams**.
- Build a **dashboard** (Streamlit or PowerBI) to alert and visualize failure risks.
