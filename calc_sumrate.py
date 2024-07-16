import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

# Parameters
nTest = 5000  # Number of test samples
na = 64  # Number of antennas
nc = 160  # Number of subcarriers
K = 8  # Number of users
snr_values = np.arange(-30, 15, 5)  # SNR values in dB
B_values = [512, 1024, 1536, 2048]
total_power = 1  # Total power constraint

# TODO: Load data
H_reconstructed = {}
for i in B_values:
    num_bit = int(i/256)
    H_reconstructed[num_bit] = sio.loadmat(f'result/testing/recovered_DL_CSI_complex_reshaped_{num_bit}bit.mat')['recovered_DL_CSI_complex_reshaped']

HDL_test = sio.loadmat('result/testing/H_test_complex_reshaped_8bit.mat')['H_test_complex_reshaped']

# Initialize array for storing average sum rates
avg_sum_rates = np.zeros((len(B_values) + 1, len(snr_values)))  # Add one row for perfect channel

# Function for water-filling power allocation
def waterfilling(noise_power, total_power):
    sorted_noise = np.sort(noise_power)
    remaining_power = total_power
    K = len(noise_power)
    water_level = 0
    for i in range(K):
        water_level = (remaining_power + np.sum(sorted_noise[:i + 1])) / (i + 1)
        if i < K - 1 and water_level < sorted_noise[i + 1]:
            break
    p_alloc = np.zeros_like(noise_power)
    for j in range(i + 1):
        p_alloc[j] = max(water_level - sorted_noise[j], 0)
    return p_alloc

# Function to compute sum rate for a single sample
def compute_sum_rate(args):
    H_selected, snr, total_power = args
    sum_rate_temp = 0
    for subcarrier in range(nc):
        H_subcarrier = H_selected[:, subcarrier, :].reshape(na, K)
        W = np.linalg.pinv(H_subcarrier)
        noise_power = np.sum(np.abs(W) ** 2, axis=1) / (10 ** (snr / 10))
        p_alloc = waterfilling(noise_power, total_power)
        signal_power = np.abs(np.diag(W @ H_subcarrier)) ** 2 * p_alloc
        sum_rate_temp += np.sum(np.log2(1 + signal_power / noise_power))
    return sum_rate_temp / nc

# Function to compute average sum rates
def compute_avg_sum_rates(H_reconstructed_current, snr):
    sum_rates = np.zeros(nTest)
    args = [(H_reconstructed_current[:, :, np.random.choice(nTest, K, replace=False)], snr, total_power) for _ in range(nTest)]
    with Pool() as pool:
        sum_rates = pool.map(compute_sum_rate, args)
    return np.mean(sum_rates)

# TODO: Loop over each B value
for i, B in enumerate(B_values):
    num_bit = int(B / 256)
    H_reconstructed_current = H_reconstructed[num_bit]
    for j in range(len(snr_values)):
        avg_sum_rates[i, j] = compute_avg_sum_rates(H_reconstructed_current, snr_values[j])
        print(f'avg_sum_rates({i + 1}, {j + 1}) = {avg_sum_rates[i, j]}')

# TODO: Calculate average sum rates for the perfect channel
for j in range(len(snr_values)):
    sum_rates_perfect = np.zeros(nTest)
    args = [(HDL_test[:, :, np.random.choice(nTest, K, replace=False)], snr_values[j], total_power) for _ in range(nTest)]
    with Pool() as pool:
        sum_rates_perfect = pool.map(compute_sum_rate, args)
    avg_sum_rates[-1, j] = np.mean(sum_rates_perfect)
    print(f'avg_sum_rates(perfect, {j + 1}) = {avg_sum_rates[-1, j]}')

# Save avg_sum_rates to a .mat file
sio.savemat('avg_sum_rates.mat', {'avg_sum_rates': avg_sum_rates})

# Ensure the directory for saving figures exists
output_dir = 'result/testing'
os.makedirs(output_dir, exist_ok=True)

# Plot the average sum rates
plt.figure()
for i in range(len(B_values)):
    plt.plot(snr_values, avg_sum_rates[i, :], label=f'B = {B_values[i]}')
plt.plot(snr_values, avg_sum_rates[-1, :], '--k', label='Perfect Channel')
plt.xlabel('SNR (dB)')
plt.ylabel('Average Sum Rate (bits/channel use)')
plt.legend()
plt.title('Average Sum Rate vs SNR for AE')
plt.grid(True)
output_file = os.path.join(output_dir, 'loss_curve_python.png')
plt.savefig(output_file)
plt.show()

print(f"Loss curve saved at {output_file}")
