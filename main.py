# main.py
import argparse, os
import pandas as pd, numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt, seaborn as sns
import joblib

# Deep learning imports (TensorFlow/Keras)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM
from tensorflow.keras.utils import to_categorical

def download_datasets():
    """Downloads the four Kaggle datasets into data/* directories."""
    api = KaggleApi()
    api.authenticate()
    os.makedirs('data/drebin', exist_ok=True)
    os.makedirs('data/microsoft', exist_ok=True)
    os.makedirs('data/ember', exist_ok=True)
    os.makedirs('data/cicmaldroid', exist_ok=True)
    print("Downloading Drebin dataset...")
    api.dataset_download_files('likhithadurusoju/drebindataset', path='data/drebin', unzip=True)
    print("Downloading Microsoft dataset...")
    api.competition_download_files('malware-classification', path='data/microsoft', unzip=True)
    print("Downloading EMBER dataset...")
    api.dataset_download_files('trinhvanquynh/ember-for-static-malware-analysis', path='data/ember', unzip=True)
    print("Downloading CICMalDroid dataset...")
    api.dataset_download_files('hasanccr92/cicmaldroid-2020', path='data/cicmaldroid', unzip=True)

def preprocess_df(df, dataset_name):
    """Detects label column, cleans data, encodes features and labels."""
    label_col = None
    for col in df.columns:
        if col.lower() in ['label','class','family','type','target']:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"No label column found in {dataset_name}")
    # Separate features and target
    X = df.drop(columns=[label_col])
    y = df[label_col]
    # Drop missing data
    X = X.dropna(axis=0, how='any')
    y = y.loc[X.index]
    # One-hot encode categorical features
    X = pd.get_dummies(X)
    # Encode target labels as integers
    if y.dtype == object or str(y.dtype).startswith('category'):
        y = LabelEncoder().fit_transform(y)
    else:
        y = LabelEncoder().fit_transform(y)
    return X, y

def train_evaluate_models(X_train, X_test, y_train, y_test, name):
    """Trains and evaluates Random Forest, SVM, Gradient Boosting models."""
    models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    for label, model in models.items():
        print(f"Training {label.upper()} on {name} data...")
        model.fit(X_train, y_train)
        joblib.dump(model, f'{label}_model_{name}.pkl')  # Save model
        # Predictions and confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'{label.upper()} Confusion Matrix ({name})')
        plt.savefig(f'confusion_matrix_{label}_{name}.png'); plt.clf()
        # ROC curve for binary classification
        if len(np.unique(y_train)) == 2:
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=label.upper())
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve ({name})'); plt.legend()
            plt.savefig(f'roc_curve_{label}_{name}.png'); plt.clf()

def train_cnn(X_train, X_test, y_train, y_test, name):
    """Trains and evaluates a simple 1D CNN."""
    num_classes = len(np.unique(y_train))
    # Reshape for Conv1D: (samples, features, 1)
    X_train_c = X_train.values.reshape(-1, X_train.shape[1], 1)
    X_test_c  = X_test.values.reshape(-1, X_test.shape[1], 1)
    # Prepare labels (one-hot if multi-class)
    if num_classes > 2:
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat  = to_categorical(y_test, num_classes)
        activation = 'softmax'; loss='categorical_crossentropy'
    else:
        y_train_cat = y_train; y_test_cat = y_test
        activation = 'sigmoid'; loss='binary_crossentropy'
    # Build CNN model
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1],1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation=activation)
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    print(f"Training CNN on {name} data...")
    model.fit(X_train_c, y_train_cat, epochs=5, batch_size=32, validation_split=0.1)
    # Evaluate
    y_prob = model.predict(X_test_c)
    if num_classes > 2:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = (y_prob > 0.5).astype('int32').ravel()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'CNN Confusion Matrix ({name})')
    plt.savefig(f'confusion_matrix_cnn_{name}.png'); plt.clf()
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label='CNN')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (CNN - {name})'); plt.legend()
        plt.savefig(f'roc_curve_cnn_{name}.png'); plt.clf()
    model.save(f'cnn_model_{name}.h5')

def train_rnn(X_train, X_test, y_train, y_test, name):
    """Trains and evaluates an LSTM-based RNN."""
    num_classes = len(np.unique(y_train))
    X_train_r = X_train.values.reshape(-1, X_train.shape[1], 1)
    X_test_r  = X_test.values.reshape(-1, X_test.shape[1], 1)
    if num_classes > 2:
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat  = to_categorical(y_test, num_classes)
        activation = 'softmax'; loss='categorical_crossentropy'
    else:
        y_train_cat = y_train; y_test_cat = y_test
        activation = 'sigmoid'; loss='binary_crossentropy'
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1],1)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation=activation)
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    print(f"Training RNN on {name} data...")
    model.fit(X_train_r, y_train_cat, epochs=5, batch_size=32, validation_split=0.1)
    # Evaluate
    y_prob = model.predict(X_test_r)
    if num_classes > 2:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = (y_prob > 0.5).astype('int32').ravel()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'RNN Confusion Matrix ({name})')
    plt.savefig(f'confusion_matrix_rnn_{name}.png'); plt.clf()
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label='RNN')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (RNN - {name})'); plt.legend()
        plt.savefig(f'roc_curve_rnn_{name}.png'); plt.clf()
    model.save(f'rnn_model_{name}.h5')

def process_and_train():
    """Loads each dataset, preprocesses, splits, and trains all models."""
    # Drebin dataset (Android malware)
    drebin_df = pd.read_csv('data/drebin/drebin.csv')
    X, y = preprocess_df(drebin_df, 'drebin')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    train_evaluate_models(X_tr, X_te, y_tr, y_te, 'drebin')
    train_cnn(X_tr, X_te, y_tr, y_te, 'drebin')
    train_rnn(X_tr, X_te, y_tr, y_te, 'drebin')
    # EMBER dataset
    ember_df = pd.read_csv('data/ember/ember.csv')
    X, y = preprocess_df(ember_df, 'ember')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    train_evaluate_models(X_tr, X_te, y_tr, y_te, 'ember')
    train_cnn(X_tr, X_te, y_tr, y_te, 'ember')
    train_rnn(X_tr, X_te, y_tr, y_te, 'ember')
    # CICMalDroid dataset
    cic_df = pd.read_csv('data/cicmaldroid/CICMalDroid2020.csv')
    X, y = preprocess_df(cic_df, 'cicmaldroid')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    train_evaluate_models(X_tr, X_te, y_tr, y_te, 'cicmaldroid')
    train_cnn(X_tr, X_te, y_tr, y_te, 'cicmaldroid')
    train_rnn(X_tr, X_te, y_tr, y_te, 'cicmaldroid')
    # Microsoft Malware Classification (BIG 2015) dataset
    try:
        msft_df = pd.read_csv('data/microsoft/trainLabels.csv')
        if 'Class' in msft_df.columns:
            print("Microsoft dataset (BIG 2015) label distribution:")
            print(msft_df['Class'].value_counts())
            print("Feature extraction for Microsoft dataset not implemented; skipping training.")
    except Exception as e:
        print("Microsoft dataset not found; skipping.")
        
def main():
    parser = argparse.ArgumentParser(description='Malware Detection Pipeline')
    parser.add_argument('--train', action='store_true', help='Run full training pipeline')
    args = parser.parse_args()
    if args.train:
        download_datasets()
        process_and_train()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
