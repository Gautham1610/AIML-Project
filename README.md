# AIML-Project
# Malware Detection Pipeline

## Setup
1. **Clone the repository** to your local machine.
2. **Create a Python virtual environment** (Python 3.8 or newer recommended).
3. **Install dependencies**:
pip install -r requirements.txt
markdown
Copy
Edit
4. **Configure Kaggle API credentials**:
- Go to your Kaggle account settings and create a new API token.  
- Place the downloaded `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or in the current working directory.  
- Alternatively set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Usage
- Run the full training pipeline with:
python main.py --train
markdown
Copy
Edit
- This will download the datasets from Kaggle, preprocess them, train all models, and save outputs (models and plots) in the project directory.
- If you wish to run parts of the pipeline separately, you can modify the script accordingly (e.g. comment out download or training sections).

# Next-Gen Malware Analysis: Harnessing AI for Advanced Threat Intelligence

This is the final package directory containing the analysis tool.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Download and analyze with your Kaggle credentials:

```bash
python -m malware_tool.cli \
  --kaggle_user YOUR_USERNAME \
  --kaggle_key YOUR_KEY \
  [--sample_static sample_static.csv] \
  [--no_download]
```

- `--sample_static`: specify a CSV (in `data/static/`) to use instead of default.
- `--no_download`: skip downloading datasets if already present.

Generated images will appear in `images/`, and trained models in `models/`.


## Datasets
The pipeline uses the following Kaggle datasets:
- **Drebin** – Android malware dataset (malware vs benign)  
- **Microsoft Malware Classification (BIG 2015)** – malware family classification (requires feature extraction not included here)  
- **EMBER** – static analysis features for malware detection  
- **CICMalDroid 2020** – Android malware classification data  

*Note:* The Microsoft dataset (`trainLabels.csv`) provides sample IDs and classes but no raw features; in this code it is only used to show label distribution (further feature extraction would be needed to train models).  

## Models and Evaluation
The following models are trained for each dataset (except Microsoft, as noted):

- **Random Forest** (scikit-learn)
- **SVM** (scikit-learn with probability outputs)
- **Gradient Boosting** (scikit-learn)
- **1D CNN** (TensorFlow/Keras)
- **LSTM RNN** (TensorFlow/Keras)

After training, the script outputs:
- Confusion matrix plots (`confusion_matrix_<model>_<dataset>.png`) for every model.  
- ROC curve plots (`roc_curve_<model>_<dataset>.png`) for binary classification tasks (e.g. malware vs benign).  
- Trained model files (`<model>_model_<dataset>.pkl` or `.h5`) saved in the working directory.  

## Repository Structure
. ├── main.py # Main pipeline script ├── requirements.txt # Python dependencies ├── README.md # Project documentation ├── data/ # Downloaded Kaggle datasets (created at runtime) │ ├── drebin/ │ ├── microsoft/ │ ├── ember/ │ └── cicmaldroid/ ├── *.pkl # Saved model files (after training) ├── *.h5 # Saved Keras model files └── *.png # Saved evaluation plots
csharp
Copy
Edit

## References
- Kaggle API for downloading datasets (requires authentication)&#8203;:contentReference[oaicite:0]{index=0}  
- Typical malware detection datasets and features (Drebin, EMBER, etc.)&#8203;:contentReference[oaicite:1]{index=1}&#8203;:contentReference[oaicite:2]{index=2}  
This project setup ensures a reproducible, command-line workflow for training and evaluating malware detection models. Follow the Setup and Usage instructions in the README to get started.
