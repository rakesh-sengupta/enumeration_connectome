import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.signal import butter, filtfilt, correlate
from scipy.stats import spearmanr
import warnings

# Suppress warnings
warnings.simplefilter('ignore')

# =============================================================================
# 1. FINAL CONFIGURATION
# =============================================================================
matfile_connectome = "/content/drive/MyDrive/Connectome_enumeration/GroupAverage_rsfMRI_matrix.mat"
matfile_eeg = "/content/drive/MyDrive/Connectome_enumeration/numerosity_EEG.mat"

# TUNED PARAMETERS (Based on your successful run)
dt = 0.001          
tau = 0.04          
alpha = 1.2         
coupling_global = 0.002 
input_scale = 0.3   
input_width = 0.100 
optimal_beta = 0.050  # Hardcoded from your result

# Region Definitions
enumeration_regions = [
    {"label": "Superior frontal lobe", "coordinates": [0, 20, 46]},
    {"label": "R superior medial frontal lobe", "coordinates": [3, 29, 40]},
    {"label": "Supplementary motor area", "coordinates": [0, 5, 67]},
    {"label": "R inferior frontal cortex/insula", "coordinates": [-30, 23, 4]},
    {"label": "L superior parietal lobe", "coordinates": [-24, -58, 43]},
    {"label": "L precuneus", "coordinates": [-12, -64, 55]},
    {"label": "L inferior parietal lobe", "coordinates": [-36, -49, 40]},
    {"label": "R insula", "coordinates": [33, 23, 4]},
    {"label": "L precentral gyrus", "coordinates": [-45, 8, 34]},
    {"label": "L precentral gyrus", "coordinates": [-48, -4, 43]},
    {"label": "L inferior frontal lobe", "coordinates": [-42, 20, 28]},
    {"label": "R inferior parietal lobe", "coordinates": [36, -49, 49]},
    {"label": "R superior parietal lobe", "coordinates": [27, -58, 49]},
    {"label": "R superior parietal lobe", "coordinates": [15, -67, 55]},
    {"label": "R precentral gyrus", "coordinates": [45, 8, 31]},
    {"label": "L middle frontal lobe", "coordinates": [-30, 2, 55]},
    {"label": "L superior frontal lobe", "coordinates": [-15, 5, 55]},
    {"label": "L precentral gyrus", "coordinates": [-27, 2, 43]},
]

# =============================================================================
# 2. CORE FUNCTIONS
# =============================================================================
def load_data():
    try: m = sio.loadmat(matfile_connectome)
    except: return None, None, None, None, None
    if 'GroupAverage_rsfMRI' in m: W_raw = m['GroupAverage_rsfMRI'].astype(float)
    else: return None, None, None, None, None
    if 'Coord' in m: coords = m['Coord'].astype(float)
    else: return None, None, None, None, None
    if coords.shape[0] == 3 and coords.shape[1] > 3: coords = coords.T
    np.fill_diagonal(W_raw, 0)
    row_sums = W_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_raw = W_raw / row_sums * coupling_global
    try: m_eeg = sio.loadmat(matfile_eeg)
    except: return None, None, None, None, None
    n_timepoints = 461
    def get_erp(keys):
        for k in keys:
            if k in m_eeg:
                d = np.asarray(m_eeg[k], float)
                if d.ndim==2: return d.mean(axis=0) if d.shape[1]==n_timepoints else d.mean(axis=1)
                if d.ndim==1: return d[:n_timepoints]
        return None
    P2_S = get_erp(["P2_S", "P2_S_all", "P2S", "subitizing"])
    P2_L = get_erp(["P2_L", "P2_L_all", "P2L", "estimation"])
    eeg_time = np.linspace(-0.3, 1.5, n_timepoints)
    return W_raw, coords, P2_S, P2_L, eeg_time

def map_regions_to_nodes(coords_connectome, region_list):
    region_coords = np.array([r['coordinates'] for r in region_list])
    dists = cdist(region_coords, coords_connectome)
    node_indices = np.argmin(dists, axis=1)
    parietal_indices = []
    for i, r in enumerate(region_list):
        if 'parietal' in r['label'].lower() or 'precuneus' in r['label'].lower():
            parietal_indices.append(node_indices[i])
    return np.unique(node_indices), np.unique(parietal_indices)

def perform_graph_surgery(W_rest, enum_indices, beta_val):
    W_task = W_rest.copy()
    for i in enum_indices:
        for j in enum_indices:
            if i != j: W_task[i, j] = -beta_val
    return W_task

def run_simulation(W_matrix, enum_indices, input_amp, t_steps):
    N = W_matrix.shape[0]
    x = np.zeros(N)
    history = np.zeros((len(t_steps), N))
    noise = np.random.normal(0, 0.02, (len(t_steps), N))
    def phi(u): return np.maximum(0, u / (1 + u))
    
    center_time = input_width / 2
    sigma = input_width / 4
    gaussian_profile = np.exp(-0.5 * ((t_steps - center_time) / sigma)**2)
    
    for i, t in enumerate(t_steps):
        inp_vec = np.zeros(N)
        if gaussian_profile[i] > 0.001:
            inp_vec[enum_indices] = input_amp * gaussian_profile[i]
        network_input = (alpha * x) + (W_matrix @ x) + inp_vec + noise[i]
        dxdt = -x + phi(network_input)
        x = x + dxdt * (dt/tau)
        history[i, :] = x
    return history

def compute_virtual_eeg(activity_history, all_coords, parietal_indices):
    virtual_loc = np.array([[0, -60, 60]]) 
    relevant_activity = activity_history[:, parietal_indices]
    relevant_coords = all_coords[parietal_indices]
    dists = cdist(virtual_loc, relevant_coords)[0]
    weights = 1 / (dists**2 + 50)
    weights = weights / weights.sum()
    eeg_raw = np.dot(relevant_activity, weights)
    return (eeg_raw - np.mean(eeg_raw)) / (np.std(eeg_raw) + 1e-6)

# =============================================================================
# 3. ANALYSIS & PLOTTING
# =============================================================================
def find_constrained_lag(real_sig, sim_sig, dt_real, max_lag_ms=400):
    corr = correlate(real_sig, sim_sig, mode='full')
    lags = np.arange(-(len(real_sig) - 1), len(real_sig))
    max_lag_idx = int((max_lag_ms / 1000) / dt_real)
    valid_mask = (lags >= -max_lag_idx) & (lags <= max_lag_idx)
    valid_lags = lags[valid_mask]
    valid_corr = corr[valid_mask]
    best_idx = np.argmax(valid_corr)
    return valid_lags[best_idx], valid_lags[best_idx] * dt_real

def main():
    W_rest, coords, real_S, real_L, real_time = load_data()
    if W_rest is None: return

    enum_idx, parietal_idx = map_regions_to_nodes(coords, enumeration_regions)
    W_task = perform_graph_surgery(W_rest, enum_idx, beta_val=optimal_beta)
    
    sim_time = np.arange(-0.3, 1.2, dt)
    
    # Run Models
    print(f"Running Simulation (Beta={optimal_beta})...")
    act_S = run_simulation(W_task, enum_idx, input_amp=2.0 * input_scale, t_steps=sim_time)
    eeg_S = filtfilt(*butter(4, [2.0/(0.5/dt), 25.0/(0.5/dt)], btype='band'), 
                     compute_virtual_eeg(act_S, coords, parietal_idx))
    
    act_L = run_simulation(W_task, enum_idx, input_amp=6.0 * input_scale, t_steps=sim_time)
    eeg_L = filtfilt(*butter(4, [2.0/(0.5/dt), 25.0/(0.5/dt)], btype='band'), 
                     compute_virtual_eeg(act_L, coords, parietal_idx))

    # Process Data
    window_smooth = 20
    real_S_sm = np.convolve(real_S, np.ones(window_smooth)/window_smooth, mode='valid')
    real_L_sm = np.convolve(real_L, np.ones(window_smooth)/window_smooth, mode='valid')
    real_time_valid = real_time[window_smooth//2 : -window_smooth//2 + 1]
    
    # Crop to match
    min_len = min(len(real_time_valid), len(real_S_sm))
    real_time_valid = real_time_valid[:min_len]
    real_S_sm = real_S_sm[:min_len]
    real_L_sm = real_L_sm[:min_len]

    # Interpolate Sim
    sim_S_interp = np.interp(real_time_valid, sim_time, eeg_S)
    sim_L_interp = np.interp(real_time_valid, sim_time, eeg_L)

    # Find Lags
    _, lag_S_time = find_constrained_lag(real_S_sm, sim_S_interp, real_time[1]-real_time[0])
    _, lag_L_time = find_constrained_lag(real_L_sm, sim_L_interp, real_time[1]-real_time[0])
    
    lag_idx_S = int(lag_S_time / (real_time[1]-real_time[0]))
    sim_S_shifted = np.roll(sim_S_interp, lag_idx_S)
    if lag_idx_S > 0: sim_S_shifted[:lag_idx_S]=0
    else: sim_S_shifted[lag_idx_S:]=0

    lag_idx_L = int(lag_L_time / (real_time[1]-real_time[0]))
    sim_L_shifted = np.roll(sim_L_interp, lag_idx_L)
    if lag_idx_L > 0: sim_L_shifted[:lag_idx_L]=0
    else: sim_L_shifted[lag_idx_L:]=0

    # PLOT
    plt.figure(figsize=(12, 6))

    # --- SUBITIZING ---
    plt.subplot(1, 2, 1)
    # Highlight P2 and P3 Windows
    plt.axvspan(0.150, 0.275, color='yellow', alpha=0.2, label='P2 Window')
    plt.axvspan(0.300, 0.500, color='green', alpha=0.1, label='P3 Window')
    
    plt.plot(real_time_valid, (real_S_sm-np.mean(real_S_sm))/np.std(real_S_sm), 'k', alpha=0.6, label='Real EEG')
    plt.plot(real_time_valid, (sim_S_shifted-np.mean(sim_S_shifted))/np.std(sim_S_shifted), 'r', linewidth=2, label=f'Model (Lag {lag_S_time*1000:.0f}ms)')
    
    plt.title("Subitizing (Late Alignment)")
    plt.xlabel("Time (s)")
    plt.ylabel("Norm. Amplitude")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(alpha=0.3)
    plt.xlim(-0.1, 0.8)

    # --- ESTIMATION ---
    plt.subplot(1, 2, 2)
    # Highlight P2 and P3 Windows
    plt.axvspan(0.150, 0.275, color='yellow', alpha=0.2, label='P2 Window')
    plt.axvspan(0.300, 0.500, color='green', alpha=0.1, label='P3 Window')

    plt.plot(real_time_valid, (real_L_sm-np.mean(real_L_sm))/np.std(real_L_sm), 'k', alpha=0.6, label='Real EEG')
    plt.plot(real_time_valid, (sim_L_shifted-np.mean(sim_L_shifted))/np.std(sim_L_shifted), 'b', linewidth=2, label=f'Model (Lag {lag_L_time*1000:.0f}ms)')
    
    plt.title("Estimation (Early Alignment)")
    plt.xlabel("Time (s)")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(alpha=0.3)
    plt.xlim(-0.1, 0.8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
