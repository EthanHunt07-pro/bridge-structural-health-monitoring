# Bridge Structural Health Monitoring Using Neural Networks

A machine-learning prototype for classifying labelled structural-condition scenarios from vibration data in the processed Z24 Bridge benchmark dataset.

## Overview

This project implements a complete workflow for vibration-based structural health monitoring (SHM), from statistical feature extraction and neural-network training to interactive predictions through a Streamlit application.

### Features

- Data exploration and preprocessing
- Statistical feature extraction from 27 sensor channels
- Feed-forward neural network with 17 output classes
- Model evaluation using accuracy, loss, and a confusion matrix
- Interactive Demo Mode using held-out test samples
- Upload Mode for user-provided vibration data
- Top-five prediction visualisation
- Automated PDF prediction reports

## Model and Input Format

The application expects a NumPy array with the following shape:

`(27, 6000)`

This represents 27 sensor channels, each containing 6,000 time steps.

Seven statistical features are extracted from each sensor:

1. Mean
2. Standard deviation
3. Root mean square (RMS)
4. Maximum
5. Minimum
6. Peak-to-peak range
7. Energy

The resulting feature vector contains 189 features.

The trained feed-forward neural network contains two hidden layers with 128 and 64 ReLU units, followed by a 17-unit Softmax output layer.

## Dataset and Attribution

The project uses the processed Z24 Bridge benchmark dataset hosted on Hugging Face:

[Z24-dataset-processed](https://huggingface.co/datasets/thanglexuan/Z24-dataset-processed)

The processed dataset contains 1,530 samples, 27 sensor channels, and 6,000 time steps per channel. The project uses the supplied labels as 17-class classification targets.

The dataset README attributes the underlying benchmark to KU Leuven and requests acknowledgement of KU Leuven and citation of the relevant original research papers. Please consult the dataset README for provenance, references, and licensing details.

The full raw dataset is not included in this repository. The two processed test arrays used by Demo Mode are included under `data/processed/`.

## Recorded Model Results

The saved training notebook reports:

| Metric | Result |
|---|---:|
| Training accuracy | 100% |
| Validation accuracy | 99.59% |
| Test accuracy | 99.35% |
| Test loss | 0.0377 |
| Held-out test samples | 306 |

These values correspond to the reported experiment and its stratified random train/test split.

**Validation limitation:** the dataset README describes each scenario/setup recording being segmented into multiple samples. Related segments may therefore appear in both training and test partitions. The reported accuracy should not be interpreted as proof of generalisation to independent recordings, other bridges, or real-world operating conditions. Grouped or condition-aware validation is an important next step.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/EthanHunt07-pro/bridge-structural-health-monitoring.git
cd bridge-structural-health-monitoring
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## Using the Application

### Demo Mode

Choose a sample from the held-out test set and run a prediction. The application displays the predicted class, confidence score, ground-truth comparison, and top-five prediction probabilities.

### Upload Mode

Upload a `.npy` file containing a numeric array with shape `(27, 6000)`. The application extracts features, scales them using the saved training scaler, and runs inference with the saved model.

The displayed Softmax score is a model confidence score and is not necessarily a calibrated probability.

### PDF Reports

The application can generate a PDF summary of a prediction, including the predicted class, confidence score, signal statistics, and prediction visualisation.

## Project Structure

```text
app/
    app.py
    class_labels.py
    config.py
    predictor.py
    preprocessing.py
    report_generator.py
models/
    bridge_damage_classifier.keras
    scaler.pkl
data/
    processed/
        X_test.npy
        y_test.npy
notebooks/
    01_Data_Exploration.ipynb
    02_Data_Preprocessing.ipynb
    03_Neural_Network_model.ipynb
requirements.txt
create_upload_samples.py
README.md
```

## Limitations and Future Work

- Evaluate generalisation using grouped or condition-aware data splits.
- Investigate robustness under environmental and loading variations.
- Explore frequency-domain and time-frequency features.
- Assess probability calibration and uncertainty on unfamiliar inputs.
- Validate against independent measurements before considering practical deployment.

This application is a research and educational prototype. Its predictions are not a substitute for professional bridge inspection, engineering judgement, or a certified structural safety assessment.

## Author

Jitesh Rajwani
B.Tech, Civil Engineering
Vellore Institute of Technology, India