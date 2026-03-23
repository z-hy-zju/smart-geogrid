import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import json
import torch.nn.functional as F
import seaborn as sns
from scipy.stats import spearmanr

# Set matplotlib Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to display Chinese labels normally
plt.rcParams['axes.unicode_minus'] = False  # Used to display negative signs normally

# Set random seed
def set_seed(seed=42):
    """Set random seed to ensure reproducible results"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Setup logging
def setup_logging(save_dir):
    """Configure logging to output to both file and console"""
    log_file = os.path.join(save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels)
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class CNNModel(nn.Module):
    def __init__(self, input_size):
        """Initialize the model
        Args:
            input_size: Total size of input features
        """
        super(CNNModel, self).__init__()

        # Calculate number of input channels
        self.num_channels = input_size // 14  # Each sensor has 14 features

        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(in_channels=self.num_channels, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Second convolutional layer
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Third convolutional layer
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Dropout layer
            nn.Dropout(0.5),

            # Flatten
            nn.Flatten(),

            # Fully connected layer
            nn.Linear(512 * 1, 256),  # 14 becomes 1 after 3 MaxPool1d(2)
            nn.ReLU(),

            # Output layer
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # Reshape input to (batch_size, channels, features)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_channels, 14)  # Reshape to (batch_size, num_channels, 14)

        # Feature extraction and prediction
        outputs = self.feature_extractor(x)

        return outputs

class SelectedFeaturesCNNModel(nn.Module):
    def __init__(self, input_size):
        """Initialize CNN model with selected features
        Args:
            input_size: Total size of input features
        """
        super(SelectedFeaturesCNNModel, self).__init__()

        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Second convolutional layer
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Third convolutional layer
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Dropout layer
            nn.Dropout(0.3),

            # Flatten
            nn.Flatten(),

            # Fully connected layer
            nn.Linear(256 * (input_size // 8), 128),  # Calculate feature dimension after 3 MaxPool1d(2)
            nn.ReLU(),

            # Output layer
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Reshape input to (batch_size, 1, features)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)  # Reshape to (batch_size, 1, num_features)

        # Feature extraction and prediction
        outputs = self.feature_extractor(x)

        return outputs

class EarlyStopping:
    """Early stopping mechanism"""

    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_r2_score(predictions, targets):
    """Calculate R2 score, handling NaN values"""
    try:
        # Convert to numpy arrays
        pred_np = predictions.cpu().detach().numpy()
        target_np = targets.cpu().detach().numpy()

        # Check and handle NaN values
        mask = ~np.isnan(pred_np).any(axis=1) & ~np.isnan(target_np).any(axis=1)
        if not mask.any():
            return 0.0  # Return 0 if all values are NaN

        # Calculate R2 with valid values
        pred_valid = pred_np[mask]
        target_valid = target_np[mask]

        r2_y1 = r2_score(target_valid[:, 0], pred_valid[:, 0])
        r2_y2 = r2_score(target_valid[:, 1], pred_valid[:, 1])

        # Handle invalid R2 values
        r2_y1 = 0.0 if np.isnan(r2_y1) else r2_y1
        r2_y2 = 0.0 if np.isnan(r2_y2) else r2_y2

        return (r2_y1 + r2_y2) / 2
    except Exception as e:
        logging.warning(f"Error calculating R2 score: {str(e)}")
        return 0.0

def custom_loss(outputs, targets, alpha=0.4, beta=0.2, gamma=0.4):
    """Custom loss function combining MSE, correlation, and individual y1/y2 losses"""
    # Separate y1 and y2 predictions and targets
    y1_pred = outputs[:, 0]
    y2_pred = outputs[:, 1]
    y1_true = targets[:, 0]
    y2_true = targets[:, 1]

    try:
        # MSE loss
        mse_loss = F.mse_loss(outputs, targets)

        # Individual y1 and y2 losses
        y1_loss = F.mse_loss(y1_pred, y1_true)
        y2_loss = F.mse_loss(y2_pred, y2_true)

        # Correlation loss - using Pearson correlation
        if len(outputs) > 1:  # Only calculate correlation when batch size > 1
            pred_corr = torch.corrcoef(outputs.t())
            target_corr = torch.corrcoef(targets.t())

            if not torch.isnan(pred_corr).any() and not torch.isnan(target_corr).any():
                corr_loss = F.mse_loss(pred_corr, target_corr)
            else:
                corr_loss = torch.tensor(0.0, device=outputs.device)
        else:
            corr_loss = torch.tensor(0.0, device=outputs.device)

        # Combined loss
        total_loss = (alpha * mse_loss +
                      beta * corr_loss +
                      gamma * (0.5 * y1_loss + 0.5 * y2_loss))

        return total_loss if not torch.isnan(total_loss) else (0.5 * y1_loss + 0.5 * y2_loss)
    except:
        # Return weighted y1 and y2 loss if calculation fails
        return 0.5 * y1_loss + 0.5 * y2_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    """Train the model"""
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []
    best_r2_y1 = float('-inf')
    best_r2_y2 = float('-inf')

    # Add learning rate warmup
    warmup_epochs = 5
    warmup_factor = 0.1

    for epoch in range(num_epochs):
        # Learning rate warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = optimizer.param_groups[0]['lr'] * (
                        1 + epoch * (1 - warmup_factor) / warmup_epochs
                )

        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Use custom loss function
            loss = criterion(outputs, targets)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_predictions.append(outputs.detach())
            train_targets.append(targets.detach())

        # Calculate R2 score for training set
        train_predictions = torch.cat(train_predictions)
        train_targets = torch.cat(train_targets)
        train_r2 = calculate_r2_score(train_predictions, train_targets)
        train_r2_scores.append(train_r2)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_predictions.append(outputs)
                val_targets.append(targets)

        # Calculate R2 score for validation set
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)
        val_r2 = calculate_r2_score(val_predictions, val_targets)
        val_r2_scores.append(val_r2)

        # Calculate individual R2 scores for y1 and y2
        val_r2_y1 = r2_score(val_targets[:, 0].cpu().numpy(), val_predictions[:, 0].cpu().numpy())
        val_r2_y2 = r2_score(val_targets[:, 1].cpu().numpy(), val_predictions[:, 1].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Update learning rate
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Log training progress
        logging.info(f'Epoch {epoch + 1}/{num_epochs} - '
                     f'Training Loss: {avg_train_loss:.4f} - '
                     f'Validation Loss: {avg_val_loss:.4f} - '
                     f'Validation R2: {val_r2:.4f} - '
                     f'Validation R2_y1: {val_r2_y1:.4f} - '
                     f'Validation R2_y2: {val_r2_y2:.4f} - '
                     f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model (based on R2 scores for y1 and y2)
        if val_r2_y1 > best_r2_y1 and val_r2_y2 > best_r2_y2:
            best_r2_y1 = val_r2_y1
            best_r2_y2 = val_r2_y2
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            logging.info(f'Saved new best model, Validation R2_y1: {val_r2_y1:.4f}, Validation R2_y2: {val_r2_y2:.4f}')

    return train_losses, val_losses, train_r2_scores, val_r2_scores

def plot_training_curves(train_losses, val_losses, train_r2_scores, val_r2_scores, save_dir):
    """Plot training curves"""
    plt.style.use('default')  # Use matplotlib default style
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Set Chinese font
    plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot loss curves
    ax1.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2, alpha=0.8)
    ax1.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10, frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Plot R2 score curves
    ax2.plot(train_r2_scores, label='Training R2', color='#2ca02c', linewidth=2, alpha=0.8)
    ax2.plot(val_r2_scores, label='Validation R2', color='#d62728', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R2 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation R2 Score Curves', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10, frameon=True, shadow=True)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, device, save_dir, y_scaler):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Inverse transform predictions and actuals
    predictions = y_scaler.inverse_transform(predictions)
    actuals = y_scaler.inverse_transform(actuals)

    # Calculate performance metrics
    mse_y1 = mean_squared_error(actuals[:, 0], predictions[:, 0])
    mse_y2 = mean_squared_error(actuals[:, 1], predictions[:, 1])
    r2_y1 = r2_score(actuals[:, 0], predictions[:, 0])
    r2_y2 = r2_score(actuals[:, 1], predictions[:, 1])

    # Calculate prediction confidence intervals
    y1_std = np.std(actuals[:, 0] - predictions[:, 0])
    y2_std = np.std(actuals[:, 1] - predictions[:, 1])

    # Plot scatter plots
    plt.style.use('default')  # Use matplotlib default style
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Set Chinese font
    plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # y1 scatter plot
    scatter1 = ax1.scatter(actuals[:, 0], predictions[:, 0],
                           alpha=0.6, s=50, c='#1f77b4', edgecolors='black', linewidth=0.5)
    min_val = min(actuals[:, 0].min(), predictions[:, 0].min())
    max_val = max(actuals[:, 0].max(), predictions[:, 0].max())
    ax1.plot([min_val, max_val], [min_val, max_val],
             'r--', lw=2, alpha=0.8, label='Ideal Prediction Line')

    # Add confidence interval
    x = np.linspace(min_val, max_val, 100)
    ax1.fill_between(x, x - 1.96 * y1_std, x + 1.96 * y1_std,
                     color='gray', alpha=0.2, label='95% Confidence Interval')

    ax1.set_xlabel('Actual Value (y1)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Value (y1)', fontsize=12, fontweight='bold')
    ax1.set_title(f'y1 Prediction Results (R2 = {r2_y1:.4f})', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, frameon=True, shadow=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # y2 scatter plot
    scatter2 = ax2.scatter(actuals[:, 1], predictions[:, 1],
                           alpha=0.6, s=50, c='#ff7f0e', edgecolors='black', linewidth=0.5)
    min_val = min(actuals[:, 1].min(), predictions[:, 1].min())
    max_val = max(actuals[:, 1].max(), predictions[:, 1].max())
    ax2.plot([min_val, max_val], [min_val, max_val],
             'r--', lw=2, alpha=0.8, label='Ideal Prediction Line')

    # Add confidence interval
    x = np.linspace(min_val, max_val, 100)
    ax2.fill_between(x, x - 1.96 * y2_std, x + 1.96 * y2_std,
                     color='gray', alpha=0.2, label='95% Confidence Interval')

    ax2.set_xlabel('Actual Value (y2)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Value (y2)', fontsize=12, fontweight='bold')
    ax2.set_title(f'y2 Prediction Results (R2 = {r2_y2:.4f})', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, frameon=True, shadow=True)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Convert numpy types to Python native types
    metrics = {
        'MSE_y1': float(mse_y1),
        'MSE_y2': float(mse_y2),
        'R2_y1': float(r2_y1),
        'R2_y2': float(r2_y2)
    }

    # Save performance metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # Log performance metrics
    logging.info('Model Evaluation Metrics:')
    for metric, value in metrics.items():
        logging.info(f'{metric}: {value:.4f}')

    return metrics

def augment_data(X, y, noise_std=0.01, scale_range=(0.95, 1.05), shift_range=(-0.05, 0.05)):
    """Data augmentation function
    Args:
        X: Feature data
        y: Target values
        noise_std: Gaussian noise standard deviation
        scale_range: Scaling range
        shift_range: Shifting range
    Returns:
        Augmented data
    """
    # Original data
    augmented_X = [X]
    augmented_y = [y]

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, X.shape)
    augmented_X.append(X + noise)
    augmented_y.append(y)

    # Random scaling
    scale = np.random.uniform(scale_range[0], scale_range[1], X.shape[1])
    augmented_X.append(X * scale)
    augmented_y.append(y)

    # Random shifting
    shift = np.random.uniform(shift_range[0], shift_range[1], X.shape[1])
    augmented_X.append(X + shift)
    augmented_y.append(y)

    # Combine all augmented data (ensure X and y are synchronized)
    augmented_X = np.vstack(augmented_X)
    augmented_y = np.vstack(augmented_y)

    # Validate dimensions
    assert augmented_X.shape[0] == augmented_y.shape[0], \
        f"X and y sample counts don't match! X:{augmented_X.shape[0]}, y:{augmented_y.shape[0]}"

    return augmented_X, augmented_y

def generate_sensor_feature_heatmap(X, feature_names, save_dir):
    """Generate correlation heatmap for sensors and features"""
    # Get sensor and feature names
    sensor_names = []
    feature_types = ['mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis',
                     'diff1_mean', 'diff1_std', 'autocorr', 'moving_avg_mean',
                     'moving_avg_std', 'fft_peak', 'psd_peak']

    # Extract sensor names from feature names
    for name in feature_names:
        if name.startswith('Sensor_'):
            sensor_name = name.split('_')[0] + '_' + name.split('_')[1]
            if sensor_name not in sensor_names:
                sensor_names.append(sensor_name)

    # Calculate correlation for each sensor and feature
    num_sensors = len(sensor_names)
    num_features = len(feature_types)
    corr_matrix = np.zeros((num_sensors, num_features))

    for i, sensor_name in enumerate(sensor_names):
        for j, feature_type in enumerate(feature_types):
            # Find all features for this sensor and feature type
            sensor_features = [idx for idx, name in enumerate(feature_names)
                               if name.startswith(sensor_name) and feature_type in name]
            if sensor_features:
                # Calculate mean of these features
                corr_matrix[i, j] = np.mean(X[:, sensor_features])

    # Create correlation DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=sensor_names, columns=feature_types)

    # Plot heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_df,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                annot=True,  # Display values
                fmt='.2f',  # Format values to two decimal places
                annot_kws={'size': 8},  # Value font size
                cbar_kws={"shrink": 0.8})
    plt.title('Sensor and Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensor_feature_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return corr_df

def select_top_features(X, feature_names, importance_df, top_n=50):
    """Select top N most important features"""
    # Define feature types
    feature_types = ['mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis',
                     'diff1_mean', 'diff1_std', 'autocorr', 'moving_avg_mean',
                     'moving_avg_std', 'fft_peak', 'psd_peak']

    # Reshape DataFrame to long format
    importance_long = importance_df.stack().reset_index()
    importance_long.columns = ['Sensor', 'Feature', 'Importance']

    # Sort by importance
    importance_long = importance_long.sort_values('Importance', ascending=False)

    # Get top N most important features
    top_features = importance_long.head(top_n)

    # Get indices of selected features
    feature_indices = []
    for _, row in top_features.iterrows():
        sensor_idx = int(row['Sensor'].split('_')[1]) - 1
        feature_idx = sensor_idx * 14 + feature_types.index(row['Feature'])
        feature_indices.append(feature_idx)

    # Select corresponding feature data
    X_selected = X[:, feature_indices]

    # Get names of selected features
    selected_features = [feature_names[i] for i in feature_indices]

    # Print debug info
    logging.info(f"Selected features shape: {X_selected.shape}")
    logging.info(f"Number of selected features: {len(selected_features)}")

    return X_selected, selected_features, top_features

def generate_feature_type_importance(X, feature_names, save_dir):
    """Generate average importance ranking of feature types across all sensors"""
    # Define feature types
    feature_types = ['mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis',
                     'diff1_mean', 'diff1_std', 'autocorr', 'moving_avg_mean',
                     'moving_avg_std', 'fft_peak', 'psd_peak']

    # Calculate average for each feature type
    feature_type_importance = []
    for feature_type in feature_types:
        # Find all features containing this feature type
        feature_indices = [idx for idx, name in enumerate(feature_names)
                           if feature_type in name]
        if feature_indices:
            # Calculate mean of these features
            avg_value = np.mean(X[:, feature_indices])
            feature_type_importance.append((feature_type, avg_value))

    # Create DataFrame and sort
    importance_df = pd.DataFrame(feature_type_importance,
                                 columns=['Feature Type', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Plot feature type importance ranking
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature Type', data=importance_df,
                palette='viridis')
    plt.title('Feature Type Importance Ranking(Average Across All Sensors)')
    plt.xlabel('Average Importance Score')
    plt.ylabel('Feature Type')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_type_importance_ranking.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return importance_df

def analyze_feature_importance(model, feature_names, save_dir):
    """Analyze feature importance of the original model"""
    # Get model weights
    weights = []
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and 'weight' in name:
            weights.append(param.detach().cpu().numpy())
            logging.info(f"Found weights with shape: {weights[-1].shape}")
            break

    if not weights:
        logging.warning("No weights found in feature extractor")
        return None

    # Calculate feature importance - using weights from first convolutional layer
    conv_weights = weights[0]  # Shape (128, 36, 5)

    # Create importance matrix for sensors and features
    importance_matrix = np.zeros((36, 14))
    for i in range(36):
        for j in range(14):
            # Calculate importance for this sensor and feature type
            feature_idx = i * 14 + j
            if feature_idx < len(feature_names):
                importance = np.abs(conv_weights[:, i, :]).mean()
                importance_matrix[i, j] = importance

    # Create correlation DataFrame
    sensor_names = [f'Sensor_{i}' for i in range(1, 37)]
    feature_types = ['mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis',
                     'diff1_mean', 'diff1_std', 'autocorr', 'moving_avg_mean',
                     'moving_avg_std', 'fft_peak', 'psd_peak']

    importance_df = pd.DataFrame(importance_matrix, index=sensor_names, columns=feature_types)

    # Plot heatmap
    plt.figure(figsize=(15, 12))

    # Calculate minimum and maximum values
    vmin = importance_df.min().min()  # DataFrame minimum value
    vmax = importance_df.max().max()  # DataFrame maximum value

    sns.heatmap(importance_df,
                cmap='viridis_r',
                vmin=vmin,  # Reverse minimum and maximum values
                vmax=vmax,  # Reverse minimum and maximum values
                square=True,
                linewidths=0.5,
                annot=True,  # Display values
                fmt='.2f',  # Format values to two decimal places
                annot_kws={'size': 8},  # Value font size
                cbar_kws={"shrink": 0.8})
    plt.title('Sensor and Feature Importance Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensor_feature_importance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot sensor importance ranking
    sensor_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sensor_importance.values, y=sensor_importance.index, palette='viridis')
    plt.title('Sensor Importance Ranking')
    plt.xlabel('Average Importance Score')
    plt.ylabel('Sensor')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensor_importance_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return importance_df

def generate_feature_correlation_heatmap(X, feature_names, save_dir):
    """Generate correlation heatmap for all features"""
    # Calculate correlation between features
    corr_matrix = np.corrcoef(X.T)

    # Create correlation DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)

    # Plot heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_df,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                annot=False,  # Don't display values as there are too many features
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return corr_df

def select_sensor_features(X, feature_names, sensor_indices):
    """Improved feature selection function with strict validation"""
    selected_feature_indices = []
    selected_feature_names = []

    # Validate input
    sensor_indices = np.unique(sensor_indices).astype(int)
    if any(i < 0 or i > 35 for i in sensor_indices):
        raise ValueError("Sensor indices must be in range 0-35")

    # Precisely select features
    for sensor_idx in sensor_indices:
        sensor_name = f"Sensor_{sensor_idx + 1}_"  # Add underscore for exact matching
        indices = [i for i, name in enumerate(feature_names) if name.startswith(sensor_name)]

        if len(indices) != 14:
            raise ValueError(f"Sensor {sensor_name} should contain 14 features, but found {len(indices)}")

        selected_feature_indices.extend(indices)
        selected_feature_names.extend([feature_names[i] for i in indices])

    # Validate output
    expected_features = len(sensor_indices) * 14
    if len(selected_feature_indices) != expected_features:
        raise ValueError(f"Feature count error! Expected {expected_features}, actual {len(selected_feature_indices)}")

    return X[:, selected_feature_indices], selected_feature_names


def analyze_selected_features_importance(model, feature_names, save_dir):
    """Analyze feature importance of the model with selected features"""
    # Get model weights
    weights = []
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and 'weight' in name:
            weights.append(param.detach().cpu().numpy())
            logging.info(f"Found weights with shape: {weights[-1].shape}")
            break

    if not weights:
        logging.warning("No weights found in feature extractor")
        return None

    # Calculate feature importance - using weights from first convolutional layer
    conv_weights = weights[0]  # Shape (64, 1, 3)

    # Calculate importance for each feature
    feature_importance = np.zeros(len(feature_names))
    for i in range(len(feature_names)):
        # Calculate importance of this feature in convolutional kernels
        importance = np.abs(conv_weights[:, 0, :]).mean()
        feature_importance[i] = importance

    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # Plot feature importance ranking
    plt.figure(figsize=(12, 8))
    importance_df = importance_df.sort_values('Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Selected Features Importance Ranking')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'selected_features_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return importance_df


def main():
    """Main function"""
    # Set random seed
    set_seed(42)

    # Create timestamp directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Setup logging
    setup_logging(save_dir)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    logging.info("Starting to load sensor combinations...")
    sensor_combinations_file = "C:/Users/Administrator/Desktop/s.xlsx"
    try:
        sensor_combinations_df = pd.read_excel(sensor_combinations_file)
        sensor_combinations = []
        for idx, row in sensor_combinations_df.iterrows():
            combo_str = str(row[0]).strip('[]')
            try:
                # Parse format like "[np.int64(2), np.int64(9), np.int64(18), np.int64(33)]"
                sensors = [int(s.split('(')[1].split(')')[0].strip())
                           for s in combo_str.split(',')]

                # Validate sensor index range (0-35)
                if any(s < 0 or s > 35 for s in sensors):
                    raise ValueError(f"Sensor index out of range (0-35): {sensors}")

                # Validate combination size (should be 4 sensors)
                if len(sensors) != 4:
                    raise ValueError(f"Each combination should contain 4 sensors, currently has {len(sensors)}")

                sensor_combinations.append(sensors)
                logging.info(f"Successfully loaded combination {idx + 1}: {sensors} -> Corresponding sensors {[f'Sensor_{s + 1}' for s in sensors]}")

            except Exception as e:
                logging.error(f"Failed to parse sensor combination: {combo_str} | Error: {str(e)}")
                continue

        if not sensor_combinations:
            raise ValueError("No valid sensor combinations found")
        logging.info(f"Loaded {len(sensor_combinations)} valid sensor combinations in total")

    except Exception as e:
        logging.error(f'Failed to load sensor combination file: {str(e)}')
        return

    # Load and preprocess data
    try:
        # Load all feature files
        features_folder = "sensor_features"
        all_features = []
        all_targets = []
        base_feature_names = []  # Store base feature names
        feature_names = []  # Store complete feature names

        # Read coordinate file to get y1 and y2 mapping
        coordinate_file = "coordinate_set.xlsx"
        coord_df = pd.read_excel(coordinate_file, header=None)
        coord_map = {}
        for _, row in coord_df.iterrows():
            filename = row[0].replace('_output.xlsx', '.xlsx')
            if not filename.endswith('.xlsx'):
                filename = f"{filename}.xlsx"
            coord_map[filename] = (float(row[1]), float(row[2]))

        # Iterate through all CSV files in features folder
        for filename in os.listdir(features_folder):
            if filename.endswith('_features.csv'):
                # Get corresponding original filename
                base_filename = filename.replace('_features.csv', '.xlsx')

                # Get y1 and y2 from coordinate mapping
                if base_filename in coord_map:
                    y1, y2 = coord_map[base_filename]
                else:
                    logging.warning(f"Coordinate information not found for file {base_filename}, skipping")
                    continue

                # Read feature file
                file_path = os.path.join(features_folder, filename)
                features_df = pd.read_csv(file_path, index_col=0)

                # If first read, save base feature names and generate complete feature names
                if not base_feature_names:
                    base_feature_names = features_df.index.tolist()
                    # Generate complete feature names (14 features * 36 sensors)
                    for sensor_idx in range(36):
                        for feature_name in base_feature_names:
                            feature_names.append(f'Sensor_{sensor_idx + 1}_{feature_name}')

                # Check if feature values are valid
                features = features_df.values.flatten()
                if np.isnan(features).any():
                    logging.warning(f"File {filename} contains invalid feature values, skipping")
                    continue

                all_features.append(features)
                all_targets.append([y1, y2])

        if not all_features:
            raise ValueError("No valid feature files found")

        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_targets)

        # Data standardization
        X_scaler = RobustScaler()
        y_scaler = RobustScaler()

        # Check and remove samples with NaN
        valid_samples = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
        X = X[valid_samples]
        y = y[valid_samples]

        X = X_scaler.fit_transform(X)
        y = y_scaler.fit_transform(y)

        # Store results for all combinations
        results = []

        # Analyze each sensor combination
        for i, sensor_indices in enumerate(sensor_combinations):
            combo_dir = os.path.join(save_dir, f'combo_{i + 1}')
            os.makedirs(combo_dir, exist_ok=True)

            # Create subdirectories to save results for different training approaches
            all_features_dir = os.path.join(combo_dir, 'all_features')
            selected_features_dir = os.path.join(combo_dir, 'selected_features')
            os.makedirs(all_features_dir, exist_ok=True)
            os.makedirs(selected_features_dir, exist_ok=True)

            logging.info(f"\nAnalyzing sensor combination {i + 1}: {sensor_indices}")

            try:
                # =============================================
                # First training: Using all features of current combination
                # =============================================
                logging.info("First training: Using all features of current combination")

                # Select features of specified sensors
                X_selected, selected_features = select_sensor_features(X, feature_names, sensor_indices)

                # Perform independent data augmentation for current combination
                logging.info("Performing data augmentation for current combination...")
                X_augmented, y_augmented = augment_data(X_selected, y)
                logging.info(f"Sample count after augmentation: {len(X_augmented)} (Original samples: {len(X_selected)})")

                # Convert data to tensors
                X_tensor = torch.tensor(X_augmented, dtype=torch.float32)
                y_tensor = torch.tensor(y_augmented, dtype=torch.float32)

                # Create dataset
                dataset = TensorDataset(X_tensor, y_tensor)

                # Split into training, validation and test sets
                total_size = len(dataset)
                train_size = int(0.8 * total_size)
                val_size = int(0.1 * total_size)
                test_size = total_size - train_size - val_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size, test_size]
                )

                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16)
                test_loader = DataLoader(test_dataset, batch_size=16)

                # Initialize model
                input_size = X_selected.shape[1]
                model = CNNModel(input_size).to(device)

                # Define loss function and optimizer
                criterion = custom_loss
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

                # Train model
                num_epochs = 240
                train_losses, val_losses, train_r2_scores, val_r2_scores = train_model(
                    model, train_loader, val_loader, criterion, optimizer, scheduler,
                    num_epochs, device, all_features_dir
                )

                # Plot training curves
                plot_training_curves(train_losses, val_losses, train_r2_scores, val_r2_scores, all_features_dir)

                # Load best model for evaluation
                model.load_state_dict(torch.load(os.path.join(all_features_dir, 'best_model.pth')))
                metrics_all = evaluate_model(model, test_loader, device, all_features_dir, y_scaler)

                # =============================================
                # Second training: Selecting important features from current combination
                # =============================================
                logging.info("Second training: Selecting important features from current combination")

                # Analyze feature importance
                model.eval()
                importance_df = analyze_feature_importance(model, selected_features, all_features_dir)

                if importance_df is not None:
                    logging.info("Feature importance analysis completed")

                    # Select top 50% features
                    top_n = 50  # Select at least 1 feature
                    X_selected_important, selected_important_features, top_features = select_top_features(
                        X_selected, selected_features, importance_df, top_n
                    )

                    # Save selected feature information
                    top_features.to_csv(os.path.join(selected_features_dir, 'selected_features_info.csv'), index=False)

                    # Perform independent data augmentation for selected features
                    logging.info("Performing data augmentation for selected features...")
                    X_selected_important_augmented, y_selected_augmented = augment_data(X_selected_important, y)

                    # Convert selected features to tensors
                    X_selected_important_tensor = torch.tensor(X_selected_important_augmented, dtype=torch.float32)
                    y_selected_tensor = torch.tensor(y_selected_augmented, dtype=torch.float32)

                    # Create new dataset
                    dataset_selected = TensorDataset(X_selected_important_tensor, y_selected_tensor)

                    # Create new data loaders using same split method
                    train_size = int(0.8 * len(dataset_selected))
                    val_size = int(0.1 * len(dataset_selected))
                    test_size = len(dataset_selected) - train_size - val_size

                    train_dataset_selected, val_dataset_selected, test_dataset_selected = torch.utils.data.random_split(
                        dataset_selected, [train_size, val_size, test_size]
                    )

                    train_loader_selected = DataLoader(train_dataset_selected, batch_size=16, shuffle=True)
                    val_loader_selected = DataLoader(val_dataset_selected, batch_size=16)
                    test_loader_selected = DataLoader(test_dataset_selected, batch_size=16)

                    # Initialize model
                    input_size = X_selected_important.shape[1]
                    model_selected = SelectedFeaturesCNNModel(input_size).to(device)

                    # Define loss function and optimizer
                    criterion = custom_loss
                    optimizer = optim.AdamW(model_selected.parameters(), lr=0.001, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

                    # Train model
                    train_losses, val_losses, train_r2_scores, val_r2_scores = train_model(
                        model_selected, train_loader_selected, val_loader_selected, criterion, optimizer, scheduler,
                        num_epochs, device, selected_features_dir
                    )

                    # Plot training curves
                    plot_training_curves(train_losses, val_losses, train_r2_scores, val_r2_scores,
                                         selected_features_dir)

                    # Load best model for evaluation
                    model_selected.load_state_dict(torch.load(os.path.join(selected_features_dir, 'best_model.pth')))
                    metrics_selected = evaluate_model(model_selected, test_loader_selected, device,
                                                      selected_features_dir,
                                                      y_scaler)

                    # Save results
                    results.append({
                        'combination': str(sensor_indices),
                        'all_features_R2_y1': metrics_all['R2_y1'],
                        'all_features_R2_y2': metrics_all['R2_y2'],
                        'all_features_avg_R2': (metrics_all['R2_y1'] + metrics_all['R2_y2']) / 2,
                        'selected_features_R2_y1': metrics_selected['R2_y1'],
                        'selected_features_R2_y2': metrics_selected['R2_y2'],
                        'selected_features_avg_R2': (metrics_selected['R2_y1'] + metrics_selected['R2_y2']) / 2,
                        'num_features': len(selected_features),
                        'num_selected_features': len(selected_important_features)
                    })

                    # Compare performance of both approaches
                    logging.info("\nModel performance comparison:")
                    logging.info("Using all features:")
                    for metric, value in metrics_all.items():
                        logging.info(f"{metric}: {value:.4f}")
                    logging.info("\nUsing selected features:")
                    for metric, value in metrics_selected.items():
                        logging.info(f"{metric}: {value:.4f}")
                else:
                    logging.warning("Unable to analyze feature importance, skipping second training")
                    results.append({
                        'combination': str(sensor_indices),
                        'all_features_R2_y1': metrics_all['R2_y1'],
                        'all_features_R2_y2': metrics_all['R2_y2'],
                        'all_features_avg_R2': (metrics_all['R2_y1'] + metrics_all['R2_y2']) / 2,
                        'selected_features_R2_y1': None,
                        'selected_features_R2_y2': None,
                        'selected_features_avg_R2': None,
                        'num_features': len(selected_features),
                        'num_selected_features': None
                    })

            except Exception as e:
                logging.error(f"Error processing sensor combination {sensor_indices}: {str(e)}")
                results.append({
                    'combination': str(sensor_indices),
                    'all_features_R2_y1': None,
                    'all_features_R2_y2': None,
                    'all_features_avg_R2': None,
                    'selected_features_R2_y1': None,
                    'selected_features_R2_y2': None,
                    'selected_features_avg_R2': None,
                    'num_features': None,
                    'num_selected_features': None
                })
                continue

        # Save results of all combinations to Excel
        results_df = pd.DataFrame(results)
        results_df.to_excel(os.path.join(save_dir, 'sensor_combinations_results.xlsx'), index=False)

        # Print results summary
        logging.info("\nSensor combination analysis results:")
        for result in results:
            logging.info(f"Combination {result['combination']}:")
            if result['all_features_R2_y1'] is not None:
                logging.info(
                    f"  Using all features: R2_y1={result['all_features_R2_y1']:.4f}, R2_y2={result['all_features_R2_y2']:.4f}, Average R2={result['all_features_avg_R2']:.4f}")
            else:
                logging.info("  Using all features: No results")
            if result['selected_features_R2_y1'] is not None:
                logging.info(
                    f"  Using selected features: R2_y1={result['selected_features_R2_y1']:.4f}, R2_y2={result['selected_features_R2_y2']:.4f}, Average R2={result['selected_features_avg_R2']:.4f}")
            else:
                logging.info("  Using selected features: No results")

        logging.info('Analysis completed!')

    except Exception as e:
        logging.error(f'Data loading failed: {str(e)}')
        return


if __name__ == '__main__':
    main()