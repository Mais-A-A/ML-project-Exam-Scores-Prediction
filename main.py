"""
===============================================================================
EXAM SCORE PREDICTION - MACHINE LEARNING PROJECT
Enhanced Version with Hyperparameter Tuning and Advanced Models
===============================================================================
Dataset: Kaggle - Exam Score Prediction Dataset
Authors: [Your Names]
Date: December 2025
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

import kagglehub
import os

print("=" * 100)
print(" " * 30 + "EXAM SCORE PREDICTION PROJECT")
print(" " * 22 + "Enhanced with Optimized Hyperparameter Tuning")
print("=" * 100)

# ============================================
# PART 1: DATA LOADING
# ============================================
print("\n" + "=" * 100)
print("STEP 1: LOADING DATASET")
print("=" * 100)

path = kagglehub.dataset_download("kundanbedmutha/exam-score-prediction-dataset")

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    csv_file_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    print(f"\nSuccessfully loaded: {csv_files[0]}")
    print(f"Dataset shape: {df.shape}")
else:
    print("Error: No CSV file found!")
    raise SystemExit

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(f"  Total samples: {len(df)}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Columns: {df.columns.tolist()}")

print("\nTarget Variable (exam_score) Statistics:")
print(df['exam_score'].describe())

# ============================================
# PART 2: DATA PREPROCESSING
# ============================================
print("\n" + "=" * 100)
print("STEP 2: DATA PREPROCESSING")
print("=" * 100)

print("\n[1/4] Dropping student_id column...")
df_clean = df.drop('student_id', axis=1)

print("[2/4] Separating features and target...")
X = df_clean.drop('exam_score', axis=1)
y = df_clean['exam_score']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols.tolist()}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols.tolist()}")

print("\n[3/4] Encoding categorical variables...")
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

    print(f"\n  {col} encoding:")
    for i, label in enumerate(le.classes_):
        print(f"    {label} -> {i}")

print("\n[4/4] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_encoded)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_encoded)*100:.1f}%)")

# ============================================
# PART 3: FEATURE SCALING
# ============================================
print("\n" + "=" * 100)
print("STEP 3: FEATURE SCALING")
print("=" * 100)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled successfully (mean=0, std=1)")

# ============================================
# PART 4: EVALUATION FUNCTION
# ============================================
def evaluate_model(model, X_eval, y_eval, model_name):
    y_pred = model.predict(X_eval)
    r2 = r2_score(y_eval, y_pred)
    mse = mean_squared_error(y_eval, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_eval, y_pred)

    return {
        'model': model_name,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'predictions': y_pred
    }

def print_results(result):
    print(f"\n{result['model']}:")
    print(f"  R2 Score: {result['R2']:.4f} ({result['R2']*100:.2f}% variance explained)")
    print(f"  RMSE: {result['RMSE']:.2f} points")
    print(f"  MAE: {result['MAE']:.2f} points")

# ============================================
# PART 5: HYPERPARAMETER TUNING
# ============================================
print("\n" + "=" * 100)
print("STEP 4: HYPERPARAMETER TUNING FOR EACH MODEL (OPTIMIZED)")
print("=" * 100)

tuning_results = []

# ============================================
# 5.1: LINEAR REGRESSION (BASELINE)
# ============================================
print("\n" + "-" * 100)
print("MODEL 1: LINEAR REGRESSION (Baseline - No Tuning Needed)")
print("-" * 100)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_result = evaluate_model(lr, X_test_scaled, y_test, "Linear Regression")
print_results(lr_result)

tuning_results.append({
    'Model': 'Linear Regression',
    'Best Params': 'Default (no meaningful tuning)',
    'R2': lr_result['R2'],
    'RMSE': lr_result['RMSE'],
    'MAE': lr_result['MAE']
})

# ============================================
# 5.2: K-NEAREST NEIGHBORS (OPTIMIZED GRID)
# ============================================
print("\n" + "-" * 100)
print("MODEL 2: K-NEAREST NEIGHBORS - HYPERPARAMETER TUNING (Optimized)")
print("-" * 100)

knn_param_grid = {
    'n_neighbors': [7, 10, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

print(f"\nTotal combinations to test: {len(knn_param_grid['n_neighbors']) * len(knn_param_grid['weights']) * len(knn_param_grid['metric'])}")

knn_grid = GridSearchCV(
    KNeighborsRegressor(),
    knn_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

print("Running Grid Search...")
knn_grid.fit(X_train_scaled, y_train)

print("\nGrid Search completed!")
print(f"Best parameters found: {knn_grid.best_params_}")
print(f"Best cross-validation R2: {knn_grid.best_score_:.4f}")

best_knn = knn_grid.best_estimator_
knn_result = evaluate_model(best_knn, X_test_scaled, y_test, "K-Nearest Neighbors (Tuned)")
print_results(knn_result)

tuning_results.append({
    'Model': 'K-Nearest Neighbors',
    'Best Params': str(knn_grid.best_params_),
    'R2': knn_result['R2'],
    'RMSE': knn_result['RMSE'],
    'MAE': knn_result['MAE']
})

cv_results_knn = pd.DataFrame(knn_grid.cv_results_)
top_5_knn = cv_results_knn.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 5 parameter combinations:")
print(top_5_knn.to_string(index=False))

# ============================================
# 5.3: NEURAL NETWORK (OPTIMIZED GRID)
# ============================================
print("\n" + "-" * 100)
print("MODEL 3: NEURAL NETWORK - HYPERPARAMETER TUNING (Optimized)")
print("-" * 100)

nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu'],
    'alpha': [0.0001, 0.001]
}

print(f"\nTotal combinations to test: {len(nn_param_grid['hidden_layer_sizes']) * len(nn_param_grid['activation']) * len(nn_param_grid['alpha'])}")

nn_grid = GridSearchCV(
    MLPRegressor(random_state=42, max_iter=800, early_stopping=True),
    nn_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

print("Running Grid Search...")
nn_grid.fit(X_train_scaled, y_train)

print("\nGrid Search completed!")
print(f"Best parameters found: {nn_grid.best_params_}")
print(f"Best cross-validation R2: {nn_grid.best_score_:.4f}")

best_nn = nn_grid.best_estimator_
nn_result = evaluate_model(best_nn, X_test_scaled, y_test, "Neural Network (Tuned)")
print_results(nn_result)

tuning_results.append({
    'Model': 'Neural Network',
    'Best Params': str(nn_grid.best_params_),
    'R2': nn_result['R2'],
    'RMSE': nn_result['RMSE'],
    'MAE': nn_result['MAE']
})

cv_results_nn = pd.DataFrame(nn_grid.cv_results_)
top_5_nn = cv_results_nn.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 5 parameter combinations:")
print(top_5_nn.to_string(index=False))
print(f"Training completed in {best_nn.n_iter_} iterations")

# ============================================
# 5.4: RANDOM FOREST - REDUCED GRID (SAFE)
# ============================================

print("\n" + "-" * 100)
print("MODEL 4: RANDOM FOREST - HYPERPARAMETER TUNING (Reduced Grid)")
print("-" * 100)

rf_param_grid = {
    'n_estimators': [150, 250],        # بدل 100,200,300
    'max_depth': [10, 20],             # أهم قيم فقط
    'min_samples_split': [5],          # قيمة واحدة مدروسة
    'min_samples_leaf': [4],           # قيمة واحدة مدروسة
    'criterion': ['squared_error']     # الافتراضي والأسرع
}

total_combinations = (
    len(rf_param_grid['n_estimators']) *
    len(rf_param_grid['max_depth']) *
    len(rf_param_grid['min_samples_split']) *
    len(rf_param_grid['min_samples_leaf']) *
    len(rf_param_grid['criterion'])
)

print(f"\nTotal combinations to test: {total_combinations}")

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

print("Running Grid Search for Random Forest...")
rf_grid.fit(X_train_scaled, y_train)

print("\nGrid Search completed!")
print(f"Best parameters found: {rf_grid.best_params_}")
print(f"Best cross-validation R2: {rf_grid.best_score_:.4f}")

best_rf = rf_grid.best_estimator_
rf_result = evaluate_model(best_rf, X_test_scaled, y_test, "Random Forest (Tuned)")
print_results(rf_result)

tuning_results.append({
    'Model': 'Random Forest',
    'Best Params': str(rf_grid.best_params_),
    'R2': rf_result['R2'],
    'RMSE': rf_result['RMSE'],
    'MAE': rf_result['MAE']
})

cv_results_rf = pd.DataFrame(rf_grid.cv_results_)
top_5_rf = cv_results_rf.nlargest(5, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score']
]

print("\nTop parameter combinations:")
print(top_5_rf.to_string(index=False))

# ============================================
# 5.5: XGBOOST (OPTIMIZED GRID)
# ============================================
if XGBOOST_AVAILABLE:
    print("\n" + "-" * 100)
    print("MODEL 5: XGBOOST - HYPERPARAMETER TUNING (Optimized)")
    print("-" * 100)

    xgb_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    print(f"\nTotal combinations to test: {len(xgb_param_grid['n_estimators']) * len(xgb_param_grid['max_depth']) * len(xgb_param_grid['learning_rate']) * len(xgb_param_grid['subsample'])}")

    xgb_grid = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        xgb_param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )

    print("Running Grid Search...")
    xgb_grid.fit(X_train_scaled, y_train)

    print("\nGrid Search completed!")
    print(f"Best parameters found: {xgb_grid.best_params_}")
    print(f"Best cross-validation R2: {xgb_grid.best_score_:.4f}")

    best_xgb = xgb_grid.best_estimator_
    xgb_result = evaluate_model(best_xgb, X_test_scaled, y_test, "XGBoost (Tuned)")
    print_results(xgb_result)

    tuning_results.append({
        'Model': 'XGBoost',
        'Best Params': str(xgb_grid.best_params_),
        'R2': xgb_result['R2'],
        'RMSE': xgb_result['RMSE'],
        'MAE': xgb_result['MAE']
    })

    cv_results_xgb = pd.DataFrame(xgb_grid.cv_results_)
    top_5_xgb = cv_results_xgb.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
    print("\nTop 5 parameter combinations:")
    print(top_5_xgb.to_string(index=False))

# ============================================
# PART 6: HYPERPARAMETER TUNING SUMMARY TABLE
# ============================================
print("\n" + "=" * 100)
print("STEP 5: HYPERPARAMETER TUNING SUMMARY (TABLE)")
print("=" * 100)

tuning_summary_df = pd.DataFrame(tuning_results)
tuning_summary_df = tuning_summary_df.sort_values('R2', ascending=False).reset_index(drop=True)

print("\nPerformance Ranking (Best to Worst):")
print("=" * 100)
print(tuning_summary_df[['Model', 'R2', 'RMSE', 'MAE', 'Best Params']].to_string(index=False))
print("=" * 100)

# ============================================
# PART 7: WEIGHTED ENSEMBLE
# ============================================
print("\n" + "=" * 100)
print("STEP 6: CREATING WEIGHTED ENSEMBLE WITH BEST MODELS")
print("=" * 100)

ensemble_models = [
    ('lr', lr),
    ('knn', best_knn),
    ('nn', best_nn),
    ('rf', best_rf)
]

ensemble_weights = [
    lr_result['R2'],
    knn_result['R2'],
    nn_result['R2'],
    rf_result['R2']
]

if XGBOOST_AVAILABLE:
    ensemble_models.append(('xgb', best_xgb))
    ensemble_weights.append(xgb_result['R2'])

total_weight = sum(ensemble_weights)
normalized_weights = [w / total_weight for w in ensemble_weights]

print("\nEnsemble Model Weights (based on R2 scores):")
print("-" * 100)
for (name, _), weight, norm_weight in zip(ensemble_models, ensemble_weights, normalized_weights):
    print(f"  {name:10s}: R2={weight:.4f} | Weight={norm_weight:.4f} ({norm_weight*100:.2f}%)")

print("\nCreating and training weighted ensemble...")
ensemble = VotingRegressor(
    estimators=ensemble_models,
    weights=ensemble_weights
)

ensemble.fit(X_train_scaled, y_train)
ensemble_result = evaluate_model(ensemble, X_test_scaled, y_test, "Weighted Ensemble")
print_results(ensemble_result)

# ============================================
# PART 8: FINAL COMPARISON
# ============================================
print("\n" + "=" * 100)
print("STEP 7: FINAL MODEL COMPARISON")
print("=" * 100)

all_results = [lr_result, knn_result, nn_result, rf_result]
if XGBOOST_AVAILABLE:
    all_results.append(xgb_result)
all_results.append(ensemble_result)

comparison_df = pd.DataFrame(all_results)[['model', 'R2', 'RMSE', 'MAE']]
comparison_df = comparison_df.sort_values('R2', ascending=False).reset_index(drop=True)

print("\nFinal Results (Ranked by R2):")
print("=" * 100)
print(comparison_df.to_string(index=False))
print("=" * 100)

best_model_idx = comparison_df['R2'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'model']
best_r2 = comparison_df.loc[best_model_idx, 'R2']

print(f"\nBest Performing Model: {best_model_name}")
print(f"  R2 Score: {best_r2:.4f}")
print(f"  Variance Explained: {best_r2*100:.2f}%")
print(f"  RMSE: {comparison_df.loc[best_model_idx, 'RMSE']:.2f} points")
print(f"  MAE: {comparison_df.loc[best_model_idx, 'MAE']:.2f} points")

# ============================================
# PART 9: FEATURE IMPORTANCE ANALYSIS
# ============================================
print("\n" + "=" * 100)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("=" * 100)

feature_importances = best_rf.feature_importances_
feature_names = X_encoded.columns.tolist()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
print("=" * 100)
for _, row in importance_df.iterrows():
    bar_length = int(row['Importance'] * 50)
    bar = '#' * bar_length
    print(f"{row['Feature']:20s} {bar:50s} {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")
print("=" * 100)

print("\nTop 5 Most Important Features:")
for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
    print(f"  {i}. {row['Feature']:20s} - {row['Importance']*100:.2f}% importance")

print("\nLeast Important Features:")
for i, (_, row) in enumerate(importance_df.tail(3).iterrows(), 1):
    print(f"  {i}. {row['Feature']:20s} - {row['Importance']*100:.2f}% importance")

# ============================================
# PART 10: VISUALIZATIONS
# ============================================
print("\n" + "=" * 100)
print("STEP 9: GENERATING VISUALIZATIONS")
print("=" * 100)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :2])
models_list = comparison_df['model'].tolist()
r2_list = comparison_df['R2'].tolist()
colors_palette = plt.cm.viridis(np.linspace(0, 1, len(models_list)))

bars = ax1.bar(range(len(models_list)), r2_list, color=colors_palette, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('R2 Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison - R2 Score', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(models_list)))
ax1.set_xticklabels(models_list, rotation=45, ha='right')
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars, r2_list):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 2])
rmse_list = comparison_df['RMSE'].tolist()
bars = ax2.barh(range(len(models_list)), rmse_list, color=colors_palette, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('RMSE', fontsize=12, fontweight='bold')
ax2.set_title('RMSE Comparison\n(Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_yticks(range(len(models_list)))
ax2.set_yticklabels(models_list, fontsize=9)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3, linestyle='--')

for bar, val in zip(bars, rmse_list):
    ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

ax3 = fig.add_subplot(gs[1, :])
colors_feat = plt.cm.plasma(np.linspace(0, 1, len(importance_df)))
bars = ax3.barh(importance_df['Feature'], importance_df['Importance'], color=colors_feat, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax3.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3, linestyle='--')

for bar, val in zip(bars, importance_df['Importance']):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)

ax4 = fig.add_subplot(gs[2, 0])
best_predictions = all_results[best_model_idx]['predictions']
ax4.scatter(y_test, best_predictions, alpha=0.6, s=20, color='steelblue', edgecolors='black', linewidth=0.5)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Exam Score', fontsize=11, fontweight='bold')
ax4.set_ylabel('Predicted Exam Score', fontsize=11, fontweight='bold')
ax4.set_title(f'Actual vs Predicted\n{best_model_name}', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(fontsize=9)

ax5 = fig.add_subplot(gs[2, 1])
errors = y_test.values - best_predictions
ax5.hist(errors, bins=40, edgecolor='black', alpha=0.7, color='mediumseagreen')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax5.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title(f'Error Distribution\n{best_model_name}', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

ax6 = fig.add_subplot(gs[2, 2])
baseline_r2 = lr_result['R2']
improvements = [(r2 - baseline_r2) * 100 for r2 in r2_list]
colors_imp = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
bars = ax6.barh(range(len(models_list)), improvements, color=colors_imp, edgecolor='black', linewidth=1.5, alpha=0.7)
ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax6.set_xlabel('R2 Improvement (%)', fontsize=11, fontweight='bold')
ax6.set_title('Improvement Over\nLinear Regression', fontsize=12, fontweight='bold')
ax6.set_yticks(range(len(models_list)))
ax6.set_yticklabels(models_list, fontsize=9)
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3, linestyle='--')

for bar, val in zip(bars, improvements):
    x_pos = val + (0.5 if val > 0 else -0.5)
    ha = 'left' if val > 0 else 'right'
    ax6.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.2f}%', va='center', ha=ha, fontsize=9, fontweight='bold')

plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
print("\nComprehensive visualization saved as 'comprehensive_results.png'")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
colors_feat = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors_feat, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance Analysis (Random Forest)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')

for bar, val in zip(bars, importance_df['Importance']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance visualization saved as 'feature_importance.png'")
plt.show()

# ============================================
# PART 11: PREDICTION FUNCTION
# ============================================
print("\n" + "=" * 100)
print("STEP 10: PREDICTION FUNCTION FOR NEW STUDENTS")
print("=" * 100)

def predict_exam_score(age, gender, course, study_hours, class_attendance,
                       internet_access, sleep_hours, sleep_quality,
                       study_method, facility_rating, exam_difficulty):

    new_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'course': [course],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_access],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality],
        'study_method': [study_method],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty]
    })

    new_encoded = new_data.copy()
    for col in categorical_cols:
        try:
            new_encoded[col] = label_encoders[col].transform(new_data[col])
        except ValueError:
            print(f"Error: Invalid value for '{col}'")
            print(f"Valid options: {list(label_encoders[col].classes_)}")
            return None

    new_scaled = scaler.transform(new_encoded)
    ensemble_pred = ensemble.predict(new_scaled)[0]

    print("\nPrediction Results:")
    print(f"  Final Prediction (Ensemble): {ensemble_pred:.2f}/100")

    if ensemble_pred >= 90:
        grade = "A+ (Outstanding)"
    elif ensemble_pred >= 80:
        grade = "A (Excellent)"
    elif ensemble_pred >= 70:
        grade = "B (Very Good)"
    elif ensemble_pred >= 60:
        grade = "C (Good)"
    elif ensemble_pred >= 50:
        grade = "D (Pass)"
    else:
        grade = "F (Fail)"

    print(f"  Expected Grade: {grade}")
    print(f"  Confidence Interval: {max(0, ensemble_pred-ensemble_result['RMSE']):.1f} - {min(100, ensemble_pred+ensemble_result['RMSE']):.1f}")

    return ensemble_pred

print("\nExample 1: High-performing student")
predict_exam_score(
    age=20, gender='female', course='b.sc',
    study_hours=7.5, class_attendance=95,
    internet_access='yes', sleep_hours=8, sleep_quality='good',
    study_method='mixed', facility_rating='high', exam_difficulty='moderate'
)

print("\n" + "-" * 100)

print("\nExample 2: Average student")
predict_exam_score(
    age=19, gender='male', course='bca',
    study_hours=3.5, class_attendance=70,
    internet_access='yes', sleep_hours=6.5, sleep_quality='average',
    study_method='online videos', facility_rating='medium', exam_difficulty='moderate'
)

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 100)
print(" " * 35 + "PROJECT SUMMARY")
print("=" * 100)

print("\nDataset Statistics:")
print(f"  Total samples: {len(df)}")
print(f"  Features: {len(X.columns)}")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

print("\nModels Evaluated:")
for i, model_name in enumerate(comparison_df['model'], 1):
    print(f"  {i}. {model_name}")

print(f"\nBest Model: {best_model_name}")
print(f"  R2 Score: {best_r2:.4f}")
print(f"  Variance Explained: {best_r2*100:.2f}%")
print(f"  Average Error: {comparison_df.loc[best_model_idx, 'RMSE']:.2f} points")

print(f"\nTop 3 Most Important Features:")
for i, (_, row) in enumerate(importance_df.head(3).iterrows(), 1):
    print(f"  {i}. {row['Feature']} ({row['Importance']*100:.2f}%)")

print("\n" + "=" * 100)
print(" " * 30 + "PROJECT COMPLETED SUCCESSFULLY")
print("=" * 100)
