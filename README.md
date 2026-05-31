Bearing RUL Prediction using Multi-Branch Deep Learning
CRISP-DM Methodology | PRONOSTIA Dataset | PyTorch
Table of Contents
Project Overview
Dataset
Project Structure
CRISP-DM Methodology
Architecture
Installation & Requirements
How to Run
Code Blocks Reference
Lambda Functions
Results & Evaluation
Degradation Classes
Troubleshooting
References
Project Overview
This project predicts the Remaining Useful Life (RUL) of rolling element bearings using a multi-branch deep learning architecture trained on the PRONOSTIA (IEEE 2012 PHM Challenge) dataset. The pipeline follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology end-to-end.

Key Features
Three-branch neural network: 1D-CNN (tabular features) + ResNet-style 2D-CNN (CWT spectrograms) + 1D-CBB (raw denoised signal)
LSTM Autoencoder for unsupervised noise filtering before feature extraction
Attention-Bidirectional LSTM (AB-LSTM) with 16-head self-attention in each branch
Dual output: RUL regression (MSLE loss) + Operating Condition classification (cross-entropy)
Joint loss: λ·L_OC + (1−λ)·L_RUL with λ=0.6
Two RUL label strategies: HI-FPT (3σ method) and HI-PCA
5-class degradation classification for maintenance decision support
Noise robustness testing with Gaussian injection (σ = 0.01 to 0.5)
Gradient-based feature importance analysis
Dataset
PRONOSTIA (IEEE 2012 PHM Challenge)
Property	Value
Total bearings	17 trials
Operating conditions	3
Learning set	6 bearings (Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2)
Test set	11 bearings (Bearing1_3 through Bearing3_3)
Sampling frequency	25,600 Hz
Window size	2,560 samples (0.1 s recording every 10 s)
Sensors	2 accelerometers (horizontal + vertical, 90° apart)
Failure criterion	Acceleration exceeds 20g
Operating Conditions
Condition	Load	Speed
OC1	4000 N	1800 rpm
OC2	4200 N	1650 rpm
OC3	5000 N	1500 rpm
Ground Truth RUL (Test Set)
Bearing	Actual RUL (s)
Bearing1_3	5730
Bearing1_4	339
Bearing1_5	1610
Bearing1_6	1460
Bearing1_7	7570
Bearing2_3	7530
Bearing2_4	1390
Bearing2_5	3090
Bearing2_6	1290
Bearing2_7	580
Bearing3_3	820
Expected Folder Structure
your_data_folder/               ← set this as DATA_ROOT in Block 2
├── Learning_set/
│   ├── Bearing1_1/
│   │   ├── acc_00001.csv
│   │   ├── acc_00002.csv
│   │   └── ...
│   ├── Bearing1_2/
│   ├── Bearing2_1/
│   ├── Bearing2_2/
│   ├── Bearing3_1/
│   └── Bearing3_2/
└── Test_set/
    ├── Bearing1_3/
    ├── Bearing1_4/
    ├── Bearing1_5/
    ├── Bearing1_6/
    ├── Bearing1_7/
    ├── Bearing2_3/
    ├── Bearing2_4/
    ├── Bearing2_5/
    ├── Bearing2_6/
    ├── Bearing2_7/
    └── Bearing3_3/
Each bearing folder contains CSV files named acc_00001.csv, acc_00002.csv, etc., with 6 columns:

Hour | Minute | Second | Microsecond | Horiz_Accel | Vert_Accel
Project Structure
project/
├── notebook.ipynb              ← main Jupyter notebook (11 blocks)
├── README.md                   ← this file
│
├── Saved Models
│   ├── ae_model.pt             ← trained LSTM Autoencoder
│   ├── best_model_fold1.pt     ← best model checkpoint (fold 1)
│   ├── best_model_fold2.pt     ← best model checkpoint (fold 2)
│   ├── best_model_fold3.pt     ← best model checkpoint (fold 3)
│   └── final_model.pt          ← final deployed model
│
├── Outputs / Plots
│   ├── eda_rms_curves.png      ← RMS degradation per bearing
│   ├── eda_lifetimes.png       ← bearing lifetime distribution
│   ├── ae_loss.png             ← autoencoder training loss
│   ├── rul_labels.png          ← HI-FPT vs HI-PCA label comparison
│   ├── feature_importance.png  ← gradient-based feature importance
│   ├── noise_robustness.png    ← RMSE vs noise level
│   ├── per_bearing_rul.png     ← predicted vs true RUL all bearings
│   ├── rul_comparison.png      ← test set bar chart + classification
│   └── rul_curves_test.png     ← per-bearing RUL prediction curves
│
└── Results
    ├── final_metrics.csv       ← per-bearing RMSE and MAE
    ├── noise_robustness.csv    ← noise level vs RMSE
    └── test_rul_results.csv    ← test bearing predictions vs actual
CRISP-DM Methodology
┌─────────────────────────────────────────────────────────────┐
│                        CRISP-DM                             │
│                                                             │
│  1. Business         Define RUL prediction goal,           │
│     Understanding    scoring metric, OC classification      │
│         ↓                                                   │
│  2. Data             Load PRONOSTIA, EDA, RMS curves,      │
│     Understanding    lifetime analysis, window stats        │
│         ↓                                                   │
│  3. Data             LSTM-AE denoising, 14 1D features,    │
│     Preparation      CWT spectrograms, raw denoised,        │
│                      HI-FPT & HI-PCA label construction     │
│         ↓                                                   │
│  4. Modelling        3-branch network + AB-LSTM,           │
│                      joint loss training, K-fold CV         │
│         ↓                                                   │
│  5. Evaluation       Per-bearing RMSE/MAE, SHAP/gradient   │
│                      importance, noise robustness           │
│         ↓                                                   │
│  6. Deployment       Inference pipeline, final report,     │
│                      model card, CSV export                 │
└─────────────────────────────────────────────────────────────┘
Architecture
Full Pipeline
Raw Vibration Signal (25.6 kHz)
         │
         ▼
┌─────────────────────┐
│  LSTM Autoencoder   │  Encoder: LSTM(512) → LSTM(64)
│  (Noise Filtering)  │  Decoder: LSTM(64) → LSTM(512) → Linear
└─────────────────────┘  Loss: MSE | Epochs: 150 | Adam
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐       ┌──────────┐
   │ Stream 1 │      │ Stream 2 │       │ Stream 3 │
   │ 14 × 1D  │      │ CWT 2D   │       │ Raw 1D   │
   │ Features │      │128×128px │       │ Denoised │
   └──────────┘      └──────────┘       └──────────┘
         │                  │                  │
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐       ┌──────────┐
   │ 1D-CNN   │      │ResNet-34 │       │ 1D-CBB   │
   │ 2 blocks │      │ 2D-CBBs  │       │ 16 CBBs  │
   │(28,56 f) │      │(13 CBBs) │       │(3+4+6+3) │
   └──────────┘      └──────────┘       └──────────┘
         │                  │                  │
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐       ┌──────────┐
   │ AB-LSTM  │      │ AB-LSTM  │       │ AB-LSTM  │
   │ BiLSTM + │      │ BiLSTM + │       │ BiLSTM + │
   │ 16-head  │      │ 16-head  │       │ 16-head  │
   │attention │      │attention │       │attention │
   └──────────┘      └──────────┘       └──────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
        ┌────────────┐           ┌────────────┐
        │  RUL Head  │           │   OC Head  │
        │ Linear →   │           │ Linear →   │
        │  Sigmoid   │           │  Softmax   │
        └────────────┘           └────────────┘
               │                         │
               ▼                         ▼
        RUL ∈ [0, 1]            OC ∈ {0, 1, 2}
Loss Function
Total Loss = λ · L_OC + (1 − λ) · L_RUL

where:
  λ         = 0.6  (tuned experimentally)
  L_OC      = Categorical Cross-Entropy
  L_RUL     = Mean Squared Logarithmic Error (MSLE)
  Optimiser = RMSProp, lr = 1e-4
  Epochs    = 1000 (300 minimum recommended)
  Batch     = 16
  K-Fold    = 3
14 Extracted Features (Stream 1)
#	Feature	Domain
1	RMS	Time
2	Variance	Time
3	Kurtosis	Time
4	Skewness	Time
5	Crest Factor	Time
6	Peak Value	Time
7	Peak-to-Peak	Time
8	Shape Factor	Time
9	Impulse Factor	Time
10	Clearance Factor	Time
11	Margin Factor	Time
12	FFT Peak-to-Peak	Frequency
13	FFT Energy	Frequency
14	Power Spectral Density	Frequency
Installation & Requirements
Python Version
Python >= 3.8
Install Dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pywavelets shap tqdm
Hardware Recommendations
Component	Minimum	Recommended
RAM	8 GB	16+ GB
GPU	None (CPU only, slow)	NVIDIA GPU with 6+ GB VRAM
Storage	5 GB	10 GB
Training time (CPU)	~8 h @ 300 epochs	—
Training time (GPU)	—	~45 min @ 300 epochs
How to Run
Step 1 — Set your data path
Open the notebook and in Block 2, set:

DATA_ROOT = Path("path/to/your/data/folder")
# Example Windows : Path("C:/Users/name/data/PRONOSTIA")
# Example Mac/Linux: Path("/home/name/datasets/PRONOSTIA")
Step 2 — Sanity check (optional but recommended)
Set low epochs for a quick end-to-end test:

# Block 4
ae_losses = train_autoencoder(ae_model, ae_train_data, epochs=10, ...)

# Block 8
EPOCHS = 10
all_names = list(bearing_index.keys())[:6]   # use 6 bearings only
Confirm all 11 blocks run without errors before full training.

Step 3 — Full training
# Block 4
ae_losses = train_autoencoder(ae_model, ae_train_data, epochs=150, ...)

# Block 6
LABEL_MODE = "pca"    # smoother gradients, better convergence

# Block 8
EPOCHS = 300          # minimum; use 1000 for paper-quality results
all_names = list(bearing_index.keys())   # all bearings
Step 4 — Run all blocks in order
Block 1  → Block 2 → Block 3 → Block 4 → Block 5
→ Block 6 → Block 7 → Block 8 → Block 9 → Block 10 → Block 11
Step 5 — Check results
test_rul_results.csv — predicted vs actual RUL per test bearing
rul_comparison.png — visual comparison chart
rul_curves_test.png — degradation curves with class labels
Code Blocks Reference
Block	CRISP-DM Phase	Description	Key Output
1	Setup	Imports, seeds, lambda utilities	Lambda function registry
2	Data Understanding	Load PRONOSTIA, discover bearings	bearing_index dict
3	Data Understanding	EDA — RMS curves, lifetime plots	eda_rms_curves.png, eda_lifetimes.png
4	Data Preparation	LSTM Autoencoder training	ae_model.pt
5	Data Preparation	3-stream feature extraction (1D, CWT, raw)	features_1d, features_cwt, features_raw
6	Data Preparation	RUL label construction (HI-FPT + HI-PCA)	rul_labels dict
7	Modelling	Multi-branch network definition	MultiBranchRUL class
8	Modelling	K-fold training with early stopping	best_model_foldN.pt
9	Evaluation	Metrics, gradient importance, noise test	feature_importance.png, noise_robustness.png
10	Deployment	Inference pipeline, final report	final_model.pt, final_metrics.csv
11	Evaluation	Test set comparison vs ground truth	test_rul_results.csv, rul_comparison.png
Lambda Functions
Lambda functions are used throughout for clean, inline transforms. All core lambdas are defined in Block 1 and reused across the pipeline.

Signal Feature Lambdas
rms          = lambda x: np.sqrt(np.mean(x**2))
variance_fn  = lambda x: np.var(x)
kurtosis_fn  = lambda x: scipy.stats.kurtosis(x)
skewness_fn  = lambda x: scipy.stats.skew(x)
crest_fn     = lambda x: np.max(np.abs(x)) / (rms(x) + 1e-10)
peak_fn      = lambda x: np.max(np.abs(x))
ppv_fn       = lambda x: np.max(x) - np.min(x)
shape_fn     = lambda x: rms(x) / (np.mean(np.abs(x)) + 1e-10)
impulse_fn   = lambda x: peak_fn(x) / (np.mean(np.abs(x)) + 1e-10)
clearance_fn = lambda x: peak_fn(x) / (np.mean(np.sqrt(np.abs(x))) + 1e-10)**2
margin_fn    = lambda x: peak_fn(x) / (np.mean(np.abs(x))**2 + 1e-10)
Frequency Domain Lambdas
fft_peak2peak = lambda x: np.max(np.abs(np.fft.rfft(x))) - np.min(np.abs(np.fft.rfft(x)))
fft_energy    = lambda x: np.sum(np.abs(np.fft.rfft(x))**2)
psd_fn        = lambda x, fs=25600: np.sum(signal.welch(x, fs=fs, nperseg=256)[1])
Loss & Training Lambdas
msle_loss  = lambda y, yhat: torch.mean((torch.log1p(y.clamp(0,1)) - torch.log1p(yhat.clamp(0,1)))**2)
joint_loss = lambda l_oc, l_rul, lam=0.6: lam * l_oc + (1 - lam) * l_rul
pct_error  = lambda actual, pred: 100 * (actual - pred) / (actual + 1e-10)
add_noise  = lambda x, sigma: (x + np.random.normal(0, sigma, x.shape)).astype(np.float32)
Label Construction Lambdas
is_anomaly   = lambda val, mu, sigma: val < mu - 3*sigma or val > mu + 3*sigma
classify_rul = lambda rul_norm: next(c["label"] for c in CLASSES if c["range"][0] <= rul_norm <= c["range"][1])
Scheduler Lambda
scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: 0.95**(ep//100))
Results & Evaluation
Metrics Used
Metric	Task	Formula
RMSE	RUL regression	√(mean((actual − pred)²))
MAE	RUL regression	mean(
% Error	Per-bearing	100 × (actual − pred) / actual
Accuracy	OC classification	correct / total
Class Accuracy	Degradation class	class_match / total_bearings
Scoring Function (IEEE 2012 PHM)
The challenge uses an asymmetric scoring function that penalises late predictions more severely than early ones:

%Err_i = 100 × (ActRUL_i − PredRUL_i) / ActRUL_i

       ⎧ exp(−ln(0.5) × (%Err/5))   if %Err ≤ 0   (late prediction)
A_i =  ⎨
       ⎩ exp(−ln(0.5) × (%Err/20))  if %Err > 0   (early prediction)

Score = (1/11) × Σ A_i
Expected Performance by Training Duration
Epochs	Class Accuracy	RMSE (approx)	Notes
20	~9%	Very high	Model undertrained, all outputs near 1.0
100	~30–45%	High	Trends visible, still noisy
300	~55–70%	Moderate	Good working results
1000	~75–85%	Low	Paper-quality, matches original study
Degradation Classes
Bearings are classified into 5 health states based on normalised RUL:

Class	RUL Range	Emoji	Recommended Action
Healthy	0.80 – 1.00	🟢	Normal operation, routine monitoring
Wear Detectable	0.55 – 0.80	🟡	Increase monitoring frequency
Non-Critical	0.30 – 0.55	🟠	Schedule maintenance within next cycle
Imminent Failure	0.10 – 0.30	🔴	Plan replacement immediately
Critical / EOL	0.00 – 0.10	⛔	Replace before next operation
Troubleshooting
RuntimeError: cudnn RNN backward can only be called in training mode
Cause: SHAP GradientExplainer conflicts with cuDNN LSTM in eval mode.
Fix: Replaced with manual gradient attribution. Block 9 uses model.train() before gradient computation and model.eval() immediately after.

RuntimeError: Input type (double) and bias type (float) should be the same
Cause: np.random.normal returns float64; model weights are float32.
Fix: Cast noise output explicitly: add_noise = lambda x, sigma: (x + np.random.normal(0, sigma, x.shape)).astype(np.float32). Also add .float() on all tensors before passing to model.

All RUL predictions are near 1.0 (Healthy)
Cause: Model undertrained — sigmoid output hasn't been pushed toward 0.
Fix: Increase EPOCHS to at least 300. Add early stopping with patience=30.

CWT extraction is very slow
Cause: PyWavelets CWT is O(n²) per window.
Fix: Run on GPU if possible, or reduce CWT_SIZE from 128 to 64 for development. Use tqdm progress bars to monitor — expect ~1–2 min per bearing on CPU.

Out of Memory (GPU)
Cause: CWT spectrograms (128×128) loaded all at once.
Fix: Reduce batch_size from 16 to 8, or reduce CWT_SIZE to 64.

FileNotFoundError: No acc_*.csv in path
Cause: DATA_ROOT pointing to wrong folder, or folder name mismatch.
Fix: Verify the path contains Learning_set/ and Test_set/ subdirectories directly. Check capitalisation on Linux/Mac systems.

References
PRONOSTIA Dataset — IEEE PHM 2012 Data Challenge, FEMTO-ST Institute.
Nectoux P. et al., "PRONOSTIA: An experimental platform for bearings accelerated degradation tests." IEEE International Conference on Prognostics and Health Management, 2012.

Original paper methodology — Multi-branch deep learning with AB-LSTM for RUL estimation using 1D features, CWT spectrograms, and raw denoised signals with joint OC + RUL loss.

CRISP-DM — Chapman P. et al., "CRISP-DM 1.0: Step-by-step data mining guide." SPSS Inc., 2000.

PyWavelets — Lee G. et al., "PyWavelets: A Python package for wavelet analysis." Journal of Open Source Software, 2019.

SHAP — Lundberg S. & Lee S., "A unified approach to interpreting model predictions." NeurIPS, 2017.

Notes
The horizontal accelerometer axis (horiz_acc) is used as the primary signal throughout. The vertical axis (vert_acc) is available for extension.
Temperature signals (temp_*.csv) are not used in this implementation but are present in the dataset.
All random seeds are fixed to 42 for reproducibility. Re-running with a different seed may produce slightly different K-fold splits and checkpoint scores.
The LABEL_MODE = "pca" setting is recommended over "fpt" for training stability, as PCA labels produce smoother gradients for the MSLE loss.
Last updated: 2026 | CRISP-DM Bearing RUL Prediction Pipeline
