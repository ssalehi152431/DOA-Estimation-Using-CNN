import numpy as np
from itertools import combinations
from scipy.io import savemat
import random

def generate_covariance_matrix(angles, snr_db, N=16, d=0.5, wavelength=1.0, snapshots=200):
    """
    Generate a covariance matrix for given angles and SNR levels.
    
    Parameters:
    - angles: List of target angles (in degrees).
    - snr_db: Signal-to-Noise Ratio in dB.
    - N: Number of antenna elements.
    - d: Spacing between antenna elements in wavelengths.
    - wavelength: Wavelength of the signal.
    - snapshots: Number of snapshots for signal generation.
    
    Returns:
    - covariance_matrix: 16x16 complex covariance matrix.
    """
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    # Array response matrix (N x M)
    array_response = np.exp(1j * 2 * np.pi * d / wavelength * 
                            np.outer(np.arange(N), np.sin(angles_rad)))  # (N x M)
    
    # Signal power based on SNR
    snr_linear = 10 ** (snr_db / 10)
    signal_power = snr_linear / len(angles)  # Split power among targets
    noise_power = 1  # Assume unit noise power
    
    # Generate random signals (M x S)
    signals = (np.random.randn(len(angles), snapshots) + 
               1j * np.random.randn(len(angles), snapshots))
    signals *= np.sqrt(signal_power)
    
    # Generate received signals (N x S)
    received_signals = np.dot(array_response, signals)  # (N x S)
    
    # Covariance matrix (N x N)
    covariance_matrix = np.dot(received_signals, received_signals.conj().T) / snapshots
    covariance_matrix += noise_power * np.eye(N)  # Add noise power to diagonal
    
    return covariance_matrix



def create_dataset(grid_range, resolution, snr_levels, N=16):
    """
    Create a dataset for DOA estimation.
    
    Parameters:
    - grid_range: Tuple defining the range of angles (e.g., (-60, 60) or (-90, 90)).
    - resolution: Grid resolution in degrees (e.g., 1 degree).
    - snr_levels: List of SNR levels in dB (e.g., [-20, -15, -10, -5, 0]).
    - N: Number of antenna elements.
    
    Returns:
    - dataset: List of 3D matrices (real, imag, phase).
    - labels: List of sparse label vectors.
    """
    angles = np.arange(grid_range[0], grid_range[1] + 1, resolution)
    angle_combinations = list(combinations(angles, 2))
    
    dataset = []
    labels = []
    
    for snr in snr_levels:
        for combo in angle_combinations:
            cov_matrix = generate_covariance_matrix(combo, snr, N=N)
            
            # Convert to feature matrix (16x16x3)
            real_part = np.real(cov_matrix)
            imag_part = np.imag(cov_matrix)
            phase_part = np.angle(cov_matrix)
            feature_matrix = np.stack([real_part, imag_part, phase_part], axis=-1)
            
            # Create label vector
            label_vector = np.zeros(len(angles))
            for angle in combo:
                label_vector[int((angle - grid_range[0]) / resolution)] = 1
            
            dataset.append(feature_matrix)
            labels.append(label_vector)
    
    return np.array(dataset), np.array(labels)


# Define parameters
grid_range_narrow = (-60, 60)
grid_range_wide = (-90, 90)
resolution = 1
snr_levels = [-20, -15, -10, -5, 0]
N = 16  # Number of antenna elements

# Generate Narrow Grid Dataset
dataset_narrow, labels_narrow = create_dataset(grid_range_narrow, resolution, snr_levels, N)
print("Narrow Grid Dataset Shape:", dataset_narrow.shape)
print("Narrow Grid Labels Shape:", labels_narrow.shape)
# Save data as .mat
narrow_data = {
    "dataset_narrow": dataset_narrow,
    "labels_narrow": labels_narrow
}
savemat("narrow_grid_dataset.mat", narrow_data)
print("Narrow Grid Dataset saved as 'narrow_grid_dataset.mat'")

# Generate Wide Grid Dataset
dataset_wide, labels_wide = create_dataset(grid_range_wide, resolution, snr_levels, N)
print("Wide Grid Dataset Shape:", dataset_wide.shape)
print("Wide Grid Labels Shape:", labels_wide.shape)
# Save data as .mat
wide_data = {
    "dataset_wide": dataset_wide,
    "labels_wide": labels_wide
}
savemat("wide_grid_dataset.mat", wide_data)
print("Wide Grid Dataset saved as 'wide_grid_dataset.mat'")