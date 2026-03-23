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
from itertools import combinations
import random

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


def ensure_confusion_matrix_dir():
    cm_dir = "Different Sensor Arrangements (4-2)"
    if not os.path.exists(cm_dir):
        os.makedirs(cm_dir)
    return cm_dir


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
                    values = df.iloc[1:, 0:].values
                    features = values.T.flatten()
                    data.append(features)
                    labels.append(folder)
    return np.array(data), np.array(labels)


def preprocess_data(data, labels):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return data_scaled, labels_encoded, label_encoder


def objective_rf(trial, X_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    return score.mean()


def train_rf_with_optuna(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=50)

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
    return rf_best, best_params


def get_sensor_importance(model, total_sensors, features_per_sensor=14):
    """Calculate the importance of each sensor (averaging all feature importances)"""
    feature_importance = model.feature_importances_
    sensor_importance = []

    for sensor_idx in range(total_sensors):
        start = sensor_idx * features_per_sensor
        end = start + features_per_sensor
        sensor_importance.append(np.mean(feature_importance[start:end]))

    return np.array(sensor_importance)


def generate_top4_combinations(sensor_importance, num_combinations=100):
    """Generate combinations of top 4 sensors based on sensor importance"""
    top_sensors = np.argsort(sensor_importance)[-4:]  # Select the 4 most important sensors
    print(f"\nTop 4 important sensors: {top_sensors}")

    combinations_list = []

    # 1. Directly use the 4 most important sensors
    combinations_list.append(list(top_sensors))

    # 2. Randomly replace 1 sensor
    for _ in range(num_combinations - 1):
        new_comb = top_sensors.copy()
        replace_idx = random.randint(0, 3)  # Randomly select one of the 4 sensors to replace
        new_sensor = random.choice([x for x in range(len(sensor_importance)) if x not in top_sensors])
        new_comb[replace_idx] = new_sensor
        combinations_list.append(sorted(new_comb))

    return combinations_list[:num_combinations]


def select_features_by_sensors(data, sensor_combination, features_per_sensor=14):
    selected_indices = []
    for sensor in sensor_combination:
        start = sensor * features_per_sensor
        end = start + features_per_sensor
        selected_indices.extend(range(start, end))
    return data[:, selected_indices]


def evaluate_model(model, X_test, y_test, label_encoder, model_name, combination_id):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n{model_name} Accuracy: {accuracy * 100:.2f}%')
    print(f"\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")

    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(f'{model_name} Confusion Matrix (Percentage)', fontsize=16)

    cm_dir = ensure_confusion_matrix_dir()
    plt.savefig(f"{cm_dir}/cm_top4_comb_{combination_id}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, cm


def plot_sensor_importance(sensor_importance):
    """Plot sensor importance ranking chart"""
    sorted_idx = np.argsort(sensor_importance)
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(sensor_importance)), sensor_importance[sorted_idx], color='dodgerblue')
    plt.yticks(range(len(sensor_importance)), sorted_idx)
    plt.xlabel('Sensor Importance (Average)', fontsize=14)
    plt.ylabel('Sensor Index', fontsize=14)
    plt.title('Sensor Importance Ranking', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    base_path = "sensor_features"

    # 1. Load data
    data, labels = load_data_from_folders(base_path)
    print(f"Loaded {len(data)} samples with {data.shape[1]} features each")

    # 2. Data preprocessing
    data_processed, labels_encoded, label_encoder = preprocess_data(data, labels)

    # 3. Calculate total number of sensors
    total_features = data_processed.shape[1]
    features_per_sensor = 14
    total_sensors = total_features // features_per_sensor
    print(f"Total sensors: {total_sensors}")

    # 4. Train base model using all sensors
    print("\nTraining base model with all sensors to evaluate sensor importance...")
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        data_processed, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )
    rf_base, _ = train_rf_with_optuna(X_train_all, y_train_all)

    # 5. Calculate sensor importance
    sensor_importance = get_sensor_importance(rf_base, total_sensors)
    plot_sensor_importance(sensor_importance)

    # 6. Generate sensor combinations based on importance
    sensor_combinations = generate_top4_combinations(sensor_importance)

    # 7. Store results
    results = []
    confusion_matrices = []
    ensure_confusion_matrix_dir()

    # 8. Evaluate each combination
    for i, sensor_combination in enumerate(sensor_combinations):
        print(f"\nEvaluating combination {i + 1}/{len(sensor_combinations)}: Sensors {sensor_combination}")

        # Select features
        X_selected = select_features_by_sensors(data_processed, sensor_combination)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )

        # Train model
        rf_model, best_params = train_rf_with_optuna(X_train, y_train)

        # Evaluate model
        accuracy, cm = evaluate_model(rf_model, X_test, y_test, label_encoder,
                                      f"Top4-Sensor RF (Comb {i + 1})", i + 1)

        # Store results
        results.append({
            'combination_id': i + 1,
            'sensors': sensor_combination,
            'accuracy': accuracy,
            'best_params': best_params
        })
        confusion_matrices.append(cm)

    # 9. Output all results
    print("\n\n=== All Top4-Sensor Combinations Results ===")
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"\nCombination {result['combination_id']}:")
        print(f"Sensors: {result['sensors']}")
        print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
        print(f"Best params: {result['best_params']}")

    # 10. Plot accuracy distribution chart
    accuracies = [r['accuracy'] for r in results]
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(accuracies) + 1), accuracies, color='dodgerblue')
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                label=f'Mean Accuracy: {np.mean(accuracies):.4f}')
    plt.xlabel('Combination ID', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy of Top4-Sensor Combinations', fontsize=16)
    plt.xticks(range(1, len(accuracies) + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # 11. Find the best combination
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n=== Best Top4-Sensor Combination ===")
    print(f"Combination ID: {best_result['combination_id']}")
    print(f"Sensors: {best_result['sensors']}")
    print(f"Accuracy: {best_result['accuracy'] * 100:.2f}%")
    print(f"Best params: {best_result['best_params']}")

    # 12. Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("top4_sensor_combination_results.csv", index=False)
    print("\nResults saved to 'top4_sensor_combination_results.csv'")