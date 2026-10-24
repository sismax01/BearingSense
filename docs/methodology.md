# Technical Methodology — BearingSense (Engineered by SISMAX / sismax01)

## 1. Engineering Problem Statement

Rolling-element bearings are among the most critical and failure-prone components in rotating industrial machinery (induction motors, pumps, turbines, and actuators studied in Automatic Control). When a localized surface defect (spall, pit, or crack) develops on the **outer raceway**, **inner raceway**, or a **rolling element (ball)**, each passage of a ball over the defect generates a short-duration mechanical impulse.

In raw accelerometer measurements, these periodic fault impulses are frequently masked by:
1. Low-frequency rotor unbalance and shaft misalignment harmonics ($1\times, 2\times, 3\times f_r$).
2. Broadband mechanical and electromagnetic background noise.
3. Structural transmission path distortion between the bearing raceway and the housing-mounted accelerometer.

Consequently, black-box machine learning models trained directly on raw time-domain windows or naive FFT spectra often fail to generalize across operating loads—or worse, achieve artificially high accuracy only because of **window-level data leakage** during evaluation.

**BearingSense** addresses this problem by combining **analytical bearing kinematics**, **high-frequency resonance demodulation (Hilbert envelope spectrum analysis)**, **leak-free run-grouped evaluation**, and **counterfactual feature attribution**.

---

## 2. Analytical Rolling-Element Bearing Kinematics

Consider a deep-groove ball bearing with:
- $N_b$: Number of rolling elements (balls)
- $d$: Ball diameter
- $D$: Bearing pitch diameter
- $\alpha$: Contact angle
- $f_r = \text{RPM} / 60$: Shaft rotational frequency in Hz

Assuming pure rolling without macroscopic slip, the four fundamental characteristic defect frequencies are:

1. **Fundamental Train Frequency (FTF — Cage Speed)**:
   $$\text{FTF} = \frac{f_r}{2} \left( 1 - \frac{d}{D} \cos\alpha \right)$$

2. **Ball Pass Frequency Outer Race (BPFO)**:
   $$\text{BPFO} = \frac{N_b \, f_r}{2} \left( 1 - \frac{d}{D} \cos\alpha \right)$$

3. **Ball Pass Frequency Inner Race (BPFI)**:
   $$\text{BPFI} = \frac{N_b \, f_r}{2} \left( 1 + \frac{d}{D} \cos\alpha \right)$$

4. **Ball Spin Frequency (BSF) & Ball Defect Frequency ($\text{BDF} = 2 \times \text{BSF}$)**:
   $$\text{BSF} = \frac{D \, f_r}{2 d} \left( 1 - \left(\frac{d}{D} \cos\alpha\right)^2 \right), \quad \text{BDF} = 2 \times \text{BSF}$$

### Pre-Configured CWRU Test Stand Bearings (`src/bearingsense/bearing_kinematics.py`)

For the Case Western Reserve University (CWRU) Drive End bearing (**SKF 6205-2RS JEM**, $N_b = 9$, $d = 0.3126\text{ in}$, $D = 1.5370\text{ in}$, $\alpha = 0^\circ$):
- $\text{BPFO} = 3.5848 \times f_r$ (e.g., $107.36\text{ Hz}$ at $1797\text{ RPM}$)
- $\text{BPFI} = 5.4152 \times f_r$ (e.g., $162.19\text{ Hz}$ at $1797\text{ RPM}$)
- $\text{BSF} = 2.3568 \times f_r$, $\text{BDF} = 4.7135 \times f_r$ (e.g., $141.17\text{ Hz}$ at $1797\text{ RPM}$)
- $\text{FTF} = 0.3983 \times f_r$ (e.g., $11.93\text{ Hz}$ at $1797\text{ RPM}$)

---

## 3. Signal Processing & Hilbert Envelope Demodulation

While the repetition rate of bearing impacts lies in the low-to-mid frequency range ($70\text{–}350\text{ Hz}$), each individual strike is a sharp impulse that excites high-frequency structural resonances of the bearing housing and accelerometer mounting ($\approx 2,000\text{–}5,000\text{ Hz}$).

BearingSense implements the classic three-stage **Hilbert Envelope Spectrum** pipeline (`src/bearingsense/signal_processing.py`):

1. **Zero-Phase Resonance Bandpass Filtering**:
   Applies a 4th-order Butterworth bandpass filter (`scipy.signal.butter` + `sosfiltfilt`) across $[2000\text{ Hz}, 5000\text{ Hz}]$ to reject low-frequency shaft harmonics ($1\times, 2\times f_r$) and isolate the structural carrier resonance $x_{\text{bp}}(t)$.
2. **Analytic Signal & Instantaneous Amplitude Extraction**:
   Computes the analytic signal via the Hilbert transform:
   $$z(t) = x_{\text{bp}}(t) + j \, \mathcal{H}\{x_{\text{bp}}(t)\}, \quad e(t) = |z(t)|$$
3. **Envelope Spectrum & Kinematic Harmonic Integration**:
   Removes the DC component from $e(t)$, applies a Hann window, and computes the one-sided amplitude spectrum $E(f)$. BearingSense then integrates the normalized spectral energy within $\pm 4.5\text{ Hz}$ bands (accounting for $1\text{–}2\%$ induction motor slip) around the $1\times, 2\times, 3\times$ harmonics of $\text{BPFO}$, $\text{BPFI}$, $\text{BSF}$, and modulation sidebands ($\text{BPFI} \pm f_r$, $\text{FTF}$).

---

## 4. Feature Engineering Summary (`src/bearingsense/features.py`)

Each fixed-length vibration window ($N = 2048$ samples, $\approx 170.7\text{ ms}$ at $12\text{ kHz}$) is mapped to an 18-dimensional interpretable feature vector:

| Category | Feature Name | Physical Meaning |
| :--- | :--- | :--- |
| **Time Domain** | `td_rms` | Root Mean Square vibration amplitude (overall energy) |
| **Time Domain** | `td_peak` | Maximum instantaneous shock amplitude |
| **Time Domain** | `td_kurtosis` | Excess 4th standardized moment (impulsiveness / spalling shock) |
| **Time Domain** | `td_skewness` | Waveform distribution asymmetry |
| **Time Domain** | `td_crest_factor` | Peak-to-RMS ratio ($\text{Peak} / \text{RMS}$) |
| **Time Domain** | `td_impulse_factor` | Peak-to-Mean-Absolute ratio ($\text{Peak} / \overline{|x|}$) |
| **Time Domain** | `td_clearance_factor` | Peak-to-Root-Amplitude ratio (sensitive to early surface pitting) |
| **Time Domain** | `td_shape_factor` | RMS-to-Mean-Absolute ratio |
| **Frequency Domain** | `fd_spectral_centroid_hz` | Center of gravity of the broadband FFT power spectrum |
| **Frequency Domain** | `fd_spectral_spread_hz` | Bandwidth dispersion around the spectral centroid |
| **Frequency Domain** | `fd_spectral_entropy` | Normalized Shannon entropy of the power spectrum |
| **Frequency Domain** | `fd_high_freq_energy_ratio` | Fraction of total vibration energy residing in the $2\text{–}5\text{ kHz}$ resonance band |
| **Envelope Domain** | `env_bpfo_energy_ratio` | Fraction of envelope spectrum energy at $1\times, 2\times, 3\times \text{BPFO}$ |
| **Envelope Domain** | `env_bpfi_energy_ratio` | Fraction of envelope spectrum energy at $1\times, 2\times, 3\times \text{BPFI}$ |
| **Envelope Domain** | `env_bsf_energy_ratio` | Fraction of envelope spectrum energy at $1\times\text{–}4\times \text{BSF}$ (including $\text{BDF}$) |
| **Envelope Domain** | `env_ftf_sideband_ratio` | Fraction of envelope energy at $\text{FTF}$ and $\text{BPFI} \pm f_r$ modulation sidebands |
| **Envelope Domain** | `env_kurtosis` | Kurtosis of the demodulated envelope signal $e(t)$ |
| **Envelope Domain** | `env_dominant_freq_hz` | Frequency of the largest peak in the demodulated envelope spectrum |

---

## 5. Preventing Window-Level Data Leakage (`src/bearingsense/evaluation.py`)

A widespread methodological flaw in published CWRU bearing studies is slicing a single continuous 10-second `.mat` file into hundreds of overlapping windows and applying a standard random `train_test_split` across all windows.

Why is that invalid?
- Adjacent or overlapping windows extracted from the same physical recording share the exact same motor load, shaft speed, ambient background noise, sensor mounting stiffness, and defect severity.
- Under a random window split, a classifier can achieve $>80\text{–}95\%$ accuracy using only generic spectral shape features by memorizing run-specific background signatures rather than learning true fault kinematics.

**How BearingSense Solves This**:
- Every window retains its physical acquisition identifier (`run_id`).
- `create_stratified_group_folds()` enforces **Stratified Group K-Fold Cross-Validation**: all windows from a given physical run are assigned strictly to either the training folds or the held-out test fold (`set(train_runs) & set(test_runs) == set()`).
- In our built-in leakage experiment (`evaluate_models_cross_validation`), we explicitly compare models trained on time/frequency features under a **Naive Random Window Split (Leaky)** ($81.9\%$ accuracy / $0.820$ Macro-F1) versus a **Leak-Free Run-Grouped Split** ($44.9\%$ accuracy / $0.451$ Macro-F1), and demonstrate how adding physics-informed **Hilbert Envelope** features restores leak-free cross-run generalization ($97.5\%$ accuracy / $0.975$ Macro-F1).

---

## 6. Explainability & Physical Evidence Attribution (`src/bearingsense/explainability.py`)

BearingSense provides three complementary layers of model explainability:
1. **Out-of-Fold Global Permutation Importance**: Combines model-agnostic permutation probability sensitivity (increase in multiclass Brier error when a feature is shuffled across held-out out-of-fold validation windows) with tree-based Gini impurity reduction.
2. **Local Counterfactual Attribution**: For any individual vibration window diagnosed by the model, BearingSense evaluates how much the predicted class probability drops ($\Delta P$) when each feature is counterfactually replaced by its healthy (`Normal`) baseline mean, alongside its standardized deviation ($z$-score) relative to healthy operation.
3. **Full-Recording Multi-Window Aggregation (`analyze_full_recording`)**: Slices a full recording into sliding windows, computes recording-level vote agreement and mean class confidence, and explains the most representative fault window.

---

## 7. Known Limitations & Scope

- **Educational & Research Scope**: BearingSense is a university portfolio and research software project. It is **not** a safety-certified industrial monitoring device (such as ISO 10816 / ISO 20816 certified hardware) and must not be used as the sole basis for safety-critical maintenance decisions.
- **Quasi-Steady Speed Assumption**: Standard FFT envelope analysis assumes quasi-constant shaft speed within each $2048$-sample window ($\approx 0.17\text{ s}$). For machines undergoing rapid speed transients, order tracking (angular resampling synchronized to a tachometer pulse) should precede envelope demodulation.
- **Single-Defect Classification**: The default taxonomy models single-mode conditions (`Normal`, `Outer_Race_Fault`, `Inner_Race_Fault`, `Ball_Fault`). Compound simultaneous faults require multi-label architectures.
