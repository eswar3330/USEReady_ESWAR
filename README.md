# USEReady_ESWAR
# Load Type Prediction Model for Power Systems (Assignment 2)

## Presented by Eswar Reddy

## 1. Objective

The primary objective of this project is to develop a robust machine learning model capable of predicting the "Load_Type" of a power system based on historical data. The "Load_Type" categorization includes "Light_Load", "Medium_Load", and "Maximum_Load". This is a multi-class classification problem.

This project demonstrates skills in:
* Comprehensive data preprocessing and cleaning
* Advanced exploratory data analysis (EDA)
* Strategic feature engineering
* Robust model selection and hyperparameter tuning
* In-depth model evaluation and interpretation (including SHAP)
* Adherence to time-series validation principles

## 2. Methodology & Solution Approach

The solution followed a structured Machine Learning pipeline designed to build a high-performing and interpretable model.

### Phase 1: Comprehensive Data Understanding & Preprocessing

* **Data Loading & Initial Inspection:** Loaded `load_data.csv` and performed initial checks (`.head()`, `.info()`, `.describe()`) to understand data types, missing values, and problematic column names (e.g., `Lagging_Current_Reactive.Power_kVarh`, `CO2(tCO2)`).
* **Column Renaming:** Renamed columns with special characters for easier access.
* **Datetime Conversion & Sorting:** Converted `Date_Time` to datetime objects, set it as the DataFrame index, and sorted the data to ensure chronological order crucial for time-series analysis.
* **Missing Value Imputation:** Applied `ffill()` (forward fill) followed by `bfill()` (backward fill) for numerical features with missing values. This method is preferred over mean/median for time-series data to maintain continuity.
* **Load Type Encoding:** Encoded the categorical target variable (`Load_Type`) into numerical labels (`Light_Load`: 0, `Maximum_Load`: 1, `Medium_Load`: 2).

### Phase 2: Advanced Exploratory Data Analysis (EDA) & Feature Engineering

This phase focused on extracting meaningful features and understanding data patterns relevant to `Load_Type`.

* **Temporal Feature Engineering:** Extracted granular time-based features from `Date_Time`, including:
    * `Hour`, `DayOfWeek`, `DayOfMonth`, `Month`, `Year`, `WeekOfYear`, `Quarter`
    * `IsWeekend` (binary indicator)
    * `TimeOfDay` (categorical bins like 'Night', 'Morning', 'Day', 'Evening', 'Late_Night') - critical for capturing daily load cycles.
* **Domain-Specific Feature Engineering:**
    * `Total_Reactive_Power_kVarh`: Sum of lagging and leading reactive power.
    * `Leading_to_Lagging_Ratio`, `Lagging_to_Leading_Ratio`: Ratios providing insights into load characteristics.
    * `Calculated_Apparent_Power_kVAh` and `Calculated_Power_Factor`: Derived an estimated power factor.
* **CRITICAL DATA ANOMALY HANDLING:**
    * **`NSM` (Number of Seconds from Midnight) Correction:** Identified that `NSM` values exceeding 86399 (seconds in 24 hours) were anomalous if the feature represents a daily cycle. These values were robustly capped at 86399.
    * **Power Factor Capping:** Observed and capped `Lagging_Current_Power_Factor` and `Leading_Current_Power_Factor` at 100.0, as values exceeding this are generally considered outliers or misinterpretations of percentage.
* **In-depth EDA Visualizations:** Utilized count plots (Load Type distribution), time series plots (Usage_kWh over time), correlation heatmaps, and box plots (feature distribution by Load Type) to gain deep insights into data relationships and patterns. The NSM analysis confirmed its daily cyclical behavior.

### Phase 3: Model Development & Validation

This phase focused on selecting, training, and validating robust classification models.

* **Feature Selection & One-Hot Encoding:** Selected a comprehensive set of features and applied one-hot encoding to `TimeOfDay`.
* **Time-Series Split (Crucial for Time-Series Data):** The dataset was split into training and test sets using a **time-based cutoff**, with the **last month of data (December 2018) reserved strictly for testing**. This approach prevents data leakage and accurately simulates a real-world scenario where the model predicts future unseen load types based on historical data.
* **Feature Scaling:** Applied `StandardScaler` to numerical features (fit on training data, transformed on both train and test).
* **Model Selection & Hyperparameter Tuning:**
    * **Baseline:** Logistic Regression (with `class_weight='balanced'` for imbalance).
    * **Main Models:** Random Forest Classifier and LightGBM Classifier (Gradient Boosting). These tree-based models excel at capturing non-linear relationships.
    * **Tuning:** `GridSearchCV` with `StratifiedKFold` (essential for maintaining class distribution in folds) was used to optimize hyperparameters. `f1_weighted` was used as the scoring metric for tuning, which is more appropriate than accuracy for imbalanced datasets.
    * **Class Imbalance Handling:** `class_weight='balanced'` was consistently applied to all classifiers to mitigate the impact of the uneven distribution of load types.

### Phase 4: Comprehensive Model Evaluation & Interpretation

This phase involved rigorously evaluating the best model and interpreting its decision-making process.

* **Best Model Identification:** The **Random Forest Classifier** consistently showed the best overall performance, demonstrating superior balance across precision, recall, and F1-scores for all classes.
* **Performance Metrics:**
    * **Classification Report:** Provided detailed Precision, Recall, and F1-scores for each Load Type, and overall Accuracy (achieved **0.95 (95%) accuracy** on the unseen test set).
    * **Confusion Matrix:** Visualized misclassifications, showing that errors were rare and primarily occurred at the boundaries between load types (e.g., Light_Load vs. Medium_Load).
    * **ROC AUC Curves (One-vs-Rest):** Demonstrated the model's excellent discriminatory power for all classes (AUC values near 1.00), indicating high confidence in its predictions.
* **Model Interpretation (Explainable AI - XAI):**
    * **Feature Importance Plot:** Identified the most influential features.
    * **SHAP (SHapley Additive exPlanations) Summary Plot:** Provided deeper insights into *how* each feature (high vs. low values) impacts the prediction for each class, explaining the model's decisions beyond simple importance scores.
* **Qualitative Error Analysis:** Sampled and analyzed misclassified examples, focusing on original feature values, to understand common patterns in prediction errors.

## 3. Evaluation Criteria & Results

The model's performance was evaluated based on Accuracy, Precision, Recall, and F1-score.

**Calculated Per-Field Scores (Based on Random Forest Classifier Predictions):**
- Light_Load:
  - Recall: 0.95 (TP: 1658, FN: 87)
  - Precision: 0.98 (TP: 1658, FP: 39)
  - F1-Score: 0.97
- Maximum_Load:
  - Recall: 0.89 (TP: 471, FN: 57)
  - Precision: 0.94 (TP: 471, FP: 30)
  - F1-Score: 0.92
- Medium_Load:
  - Recall: 0.98 (TP: 688, FN: 16)
  - Precision: 0.87 (TP: 688, FP: 104)
  - F1-Score: 0.92

**Overall Scores:**
- Accuracy: 0.95
- Macro Avg F1-Score: 0.93
- Weighted Avg F1-Score: 0.95

**Discussion of Results:**
The Random Forest Classifier demonstrated **excellent performance**, achieving a high overall accuracy of 95% on unseen data. Critically, its high F1-scores for all load types, including the less frequent 'Maximum_Load' and 'Medium_Load', confirm its robustness in handling class imbalance. The interpretability analysis (Feature Importance, SHAP) validated that the model learned from expected physical drivers (e.g., `NSM`, `Hour`, `Usage_kWh`), aligning with domain intuition. Misclassifications were minimal and largely confined to borderline cases, reflecting the inherent ambiguity at class transitions.

## 4. Submission Requirements

### Codebase
- The entire solution is provided as a Notebook (`Useready_AIML_Assignment_2_Eswar.ipynb`) in this GitHub repository.
- It contains all code from data preprocessing, feature engineering, model selection, training, and comprehensive evaluation.

### Instructions to Run
1.  **Clone the Repository:** `git clone https://github.com/eswar3330/USEReady_ESWAR
2.  **Navigate to Notebook:** Open `Useready_AIML_Assignment_2_Eswar.ipynb` in Google Colab.
3.  **Upload Data:** In the Colab environment, create a `data` folder at the root.
    * Upload `load_data.csv` into the `data` folder.
4.  **Run Cells Sequentially:** Execute each code cell in the notebook from top to bottom. Ensure Colab runtime is set to GPU (Runtime -> Change runtime type -> T4 GPU or equivalent) for faster model training.

### Predictions for Test Set
The predictions for the files in the test set were generated by the Random Forest Classifier and are implicitly part of the model evaluation results presented in the notebook's outputs and classification reports. The `y_pred_rf` variable holds these predictions.

### Per Field Recall Score
The calculated Per-Field Recall, Precision, and F1-Scores are presented above in section 3.

## 4. Future Work & Enhancements

To further enhance this system for production-grade robustness and accuracy:

* **Add External Data Sources:**
    * Weather data (temperature, humidity, cloud cover, etc.)
    * Calendar data (holidays, weekends, special events)
    * Economic or regional demand shifts
    * Population/housing trends
* **Better Time-Series Features:**
    * Lag features (e.g., load from yesterday same time)
    * Rolling windows (e.g., moving averages, volatility)
    * Fourier series to model seasonality more explicitly
* **Deep Error Analysis:**
    * Focus on edge cases
    * Use fuzzy thresholds or prediction confidence zones for ambiguous samples
* **Explore Model Ensembles:**
    * Blend multiple models (e.g., RF + XGBoost + LightGBM)
    * Potential performance boost and stability
* **Adaptive Learning + Retraining:**
    * Monitor for concept drift in production
    * Set up a retraining pipeline that updates the model monthly/quarterly
* **Productionizing It:**
    * Serve via REST API (Flask/FastAPI)
    * Add logging, monitoring (e.g., Prometheus), and alerting
* **Explainability Improvements:**
    * SHAP force plots for specific prediction audits
    * Counterfactuals for 'what-if' explanations
* **Cost-Sensitive Learning:**
    * Not all mistakes are equal — penalize critical ones more (e.g., misclassifying Maximum_Load as Light_Load)
