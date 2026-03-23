import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.manifold import TSNE  # Importing T-SNE

# Load data
def load_data_from_folders(base_path):
    data = []
    labels = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.xlsx'):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_excel(file_path, header=None)
                    values = df.iloc[1:, 1:].values
                    features = values.T.flatten()
                    data.append(features)
                    labels.append(folder)
    return np.array(data), np.array(labels)


# Data preprocessing
def preprocess_data(data, labels):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return data_scaled, labels_encoded, label_encoder


# Objective function for Bayesian optimization (optimizing Random Forest hyperparameters)
def objective_rf(trial, X_train, y_train):
    # Hyperparameter space definition
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    # Random Forest model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    # Model training
    rf.fit(X_train, y_train)

    # Evaluate model performance using cross-validation
    score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    return score.mean()


# Train Random Forest with Bayesian optimization
def train_rf_with_optuna(X_train, y_train):
    # Create an optuna optimizer
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=50)

    # Output optimal hyperparameters
    print(f"Best trial: {study.best_trial.params}")

    # Train the final model with optimal hyperparameters
    best_params = study.best_trial.params
    rf_best = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=42
    )
    rf_best.fit(X_train, y_train)
    return rf_best


# Output the top 50 most important features
def plot_feature_importance(model, feature_names, top_n=50):
    # Get feature importances
    feature_importance = model.feature_importances_

    # Pair features with importances and sort
    feature_importance_sorted = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

    # Extract sorted feature names and importance values
    sorted_feature_names, sorted_importances = zip(*feature_importance_sorted)

    # Keep only the top_n most important features
    sorted_feature_names = sorted_feature_names[:top_n]
    sorted_importances = sorted_importances[:top_n]

    # Plot feature importance bar chart
    plt.figure(figsize=(16, 12))  # Increase figure size
    plt.barh(sorted_feature_names, sorted_importances, color='dodgerblue')
    plt.xlabel('Feature Importance', fontsize=32)
    plt.ylabel('Feature', fontsize=32) 
    plt.title(f'Top {top_n} Feature Importance (Random Forest)', fontsize=36)    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24) 

    # Reverse y-axis so that higher importance features appear at the top
    plt.gca().invert_yaxis()

    plt.tight_layout()  # Prevent label overlap
    plt.show()

    # Print sorted features and their importances
    print("\nTop 50 Feature Importance Ranking:")
    for i, (name, importance) in enumerate(zip(sorted_feature_names, sorted_importances)):
        print(f"{i + 1}. {name} - Importance: {importance:.4f}")


# T-SNE visualization
def tsne_visualization(X_train, y_train, label_encoder):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_train)

    # Define custom colors for the four classes
    unique_labels = np.unique(y_train)
    colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700']  # Red, Blue, Green, Yellow

    # Create a mapping of labels to colors
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(12, 10))  # Increase figure size

    # Scatter plot with custom colors
    for label in unique_labels:
        indices = np.where(y_train == label)
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
                    c=[label_to_color[label]] * len(indices[0]),
                    label=label_encoder.inverse_transform([label])[0], s=100, edgecolor='k')  # Increase point size

    plt.xlabel('T-SNE Component 1', fontsize=28)
    plt.ylabel('T-SNE Component 2', fontsize=28) 
    plt.title('T-SNE Visualization of the Data', fontsize=32)
    plt.legend(title='Class', fontsize=24, title_fontsize=24) 
    plt.tight_layout()  # Prevent label overlap
    plt.show()


def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n{model_name} Accuracy: {accuracy * 100:.2f}%')
    print(f"\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Use a lighter color map (such as 'Blues' or 'YlOrBr')
    plt.figure(figsize=(12, 10))  # Increase figure size
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt='.2f',
        cmap='PuRd',  # Lighter colors, such as 'Blues', 'YlOrBr', 'PuRd'
        annot_kws={"size": 40},  # Adjust number font size
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    # Adjust legend font size (via colorbar object)
    cbar = plt.gcf().axes[-1]  # Get colorbar object
    cbar.yaxis.label.set_size(40)
    cbar.tick_params(labelsize=40)

    # Adjust axis label font size
    plt.xlabel('Predicted', fontsize=40) 
    plt.ylabel('True', fontsize=40) 
    plt.xticks(fontsize=40) 
    plt.yticks(fontsize=40)  
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == '__main__':
    base_path = "sensor_features"

    # Load data
    data, labels = load_data_from_folders(base_path)
    print(f"Loaded {len(data)} samples with {data.shape[1]} features each")

    # Data preprocessing
    data_processed, labels_encoded, label_encoder = preprocess_data(data, labels)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data_processed, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    # Bayesian optimization training for Random Forest
    print("\nTraining Random Forest with Bayesian Optimization...")
    rf_model = train_rf_with_optuna(X_train, y_train)

    # T-SNE visualization
    print("\nVisualizing Data with T-SNE...")
    tsne_visualization(X_train, y_train, label_encoder)

    # Evaluate model
    print("\nEvaluating Random Forest with Optimized Hyperparameters...")
    evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest (Optimized)")

    # Feature names
    feature_names = [
        "mean", "std", "max", "min", "median", "skew", "kurtosis",
        "diff1_mean", "diff1_std", "autocorr", "moving_avg_mean",
        "moving_avg_std", "fft_peak", "psd_peak"
    ]

    # Since each sensor in the dataset has 14 features, we need to name all features
    total_sensors = X_train.shape[1] // 14  # Assuming each sensor has 14 features
    feature_names_expanded = []
    for sensor_id in range(1, total_sensors + 1):
        for feature in feature_names:
            feature_names_expanded.append(f"Sensor {sensor_id} {feature}")

    # Output the top 50 most important features
    plot_feature_importance(rf_model, feature_names_expanded, top_n=50)