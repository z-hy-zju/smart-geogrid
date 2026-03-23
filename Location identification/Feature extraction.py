import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.fft import fft
from scipy.signal import welch


def calculate_features(sensor_data):
    """Calculate features for a single sensor"""
    sensor_data = np.array(sensor_data, dtype=np.float64)

    # Check if there is valid data
    if np.all(np.isnan(sensor_data)):
        raise ValueError("All data is invalid")

    features = {}

    # Basic statistical features
    features['mean'] = np.nanmean(sensor_data)
    features['std'] = np.nanstd(sensor_data)
    features['max'] = np.nanmax(sensor_data)
    features['min'] = np.nanmin(sensor_data)
    features['median'] = np.nanmedian(sensor_data)

    # Remove NaN values for calculations
    valid_data = sensor_data[~np.isnan(sensor_data)]
    if len(valid_data) > 0:
        features['skew'] = stats.skew(valid_data)
        features['kurtosis'] = stats.kurtosis(valid_data)

        # Time series features
        diff1 = np.diff(valid_data)
        features['diff1_mean'] = np.mean(diff1)
        features['diff1_std'] = np.std(diff1)

        autocorr = np.correlate(valid_data, valid_data, mode='full')
        features['autocorr'] = autocorr[len(autocorr) // 2]

        window_size = min(5, len(valid_data))
        moving_avg = np.convolve(valid_data, np.ones(window_size) / window_size, mode='valid')
        features['moving_avg_mean'] = np.mean(moving_avg)
        features['moving_avg_std'] = np.std(moving_avg)

        fft_values = fft(valid_data)
        features['fft_peak'] = np.max(np.abs(fft_values))

        freqs, psd = welch(valid_data)
        features['psd_peak'] = np.max(psd)
    else:
        features['skew'] = np.nan
        features['kurtosis'] = np.nan
        features['diff1_mean'] = np.nan
        features['diff1_std'] = np.nan
        features['autocorr'] = np.nan
        features['moving_avg_mean'] = np.nan
        features['moving_avg_std'] = np.nan
        features['fft_peak'] = np.nan
        features['psd_peak'] = np.nan

    return features


def process_subfolders(data_folder, output_folder):
    """Process Excel files in each subfolder of the target folder, and output processed Excel files to corresponding subfolders"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder

    processed_files = set()  # Used to track processed files
    skipped_files = []  # Used to record skipped files
    error_files = []  # Used to record files with errors

    # Traverse subfolders in the target folder
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(root, file)

                # Skip if file has already been processed
                if file in processed_files:
                    continue

                # Get relative path of current file
                relative_path = os.path.relpath(root, data_folder)
                output_subfolder = os.path.join(output_folder, relative_path)

                # Create corresponding subfolder (if it doesn't exist)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                try:
                    # Read sensor data file (preserve original column names)
                    sensor_data = pd.read_excel(file_path)

                    # Ensure data is for 36 sensors
                    if len(sensor_data.columns) != 36:
                        print(f"Warning: {file} does not have 36 sensors, actual number: {len(sensor_data.columns)}")
                        skipped_files.append((file, "Incorrect number of sensors"))
                        continue

                    # Get data values (skip column names)
                    sensor_values = sensor_data.values.astype(np.float64)

                    # Initialize feature dictionary, pre-allocate 36 positions for each feature
                    all_features = {}
                    # Get feature names from first sensor as template
                    first_sensor_features = calculate_features(sensor_values[:, 0])
                    for feature_name in first_sensor_features.keys():
                        all_features[feature_name] = np.full(36, np.nan, dtype=np.float64)  # Pre-allocate 36 positions with float64 type

                    # Calculate features for each sensor
                    for sensor_idx in range(36):
                        try:
                            features = calculate_features(sensor_values[:, sensor_idx])
                            for feature_name, value in features.items():
                                all_features[feature_name][sensor_idx] = float(value)  # Ensure conversion to float
                        except Exception as e:
                            print(f"Warning: Error processing sensor {sensor_idx + 1} in file {file}: {str(e)}")
                            continue

                    # Create feature DataFrame, features as rows, sensors as columns
                    features_df = pd.DataFrame(all_features)
                    # Transpose DataFrame so features are rows and sensors are columns
                    features_df = features_df.T
                    # Set column names as sensor numbers
                    features_df.columns = [f'sensor_{i + 1}' for i in range(36)]

                    # Create a feature file for each file, ensure all values are saved as floats
                    feature_filename = f"{os.path.splitext(file)[0]}_features.xlsx"
                    feature_filepath = os.path.join(output_subfolder, feature_filename)
                    features_df.to_excel(feature_filepath, index=False, float_format='%.6f')  # Save float values with 6 decimal places

                    print(f"Successfully processed file: {file}, features saved to {feature_filepath}")

                    processed_files.add(file)

                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    error_files.append((file, str(e)))

    # Print detailed processing report
    print("\nData Processing Report:")
    print(f"Number of successfully processed files: {len(processed_files)}")
    print(f"Feature files saved to {output_folder} folder")

    if skipped_files:
        print("\nSkipped files:")
        for f, reason in skipped_files:
            print(f"- {f}: {reason}")

    if error_files:
        print("\nFiles with processing errors:")
        for f, error in error_files:
            print(f"- {f}: {error}")


if __name__ == "__main__":
    print("Starting to process Excel files in subfolders...")

    # Set target folder and output folder paths
    data_folder = "excel"  # Subfolders contained in the target folder
    output_folder = "sensor_features"  # New folder for saving processed Excel files

    # Process all Excel files in subfolders
    process_subfolders(data_folder, output_folder)

    print("\nProcessing completed.")