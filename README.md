# matching-no-matching

**Match Outcome Prediction**
This project demonstrates a complete deep learning workflow, from synthetic data collection to model training, optimization, and deployment through an interactive Streamlit app for real-time match outcome prediction.

📌 Highlights
📊 Synthetic Data Collection via Google Forms

🧠 Unsupervised Labeling using KMeans

🔐 ANN with L2 Regularization, Dropout & BatchNorm

🧪 Inference via Command Line with Argparse

🌐 Interactive Streamlit App

🔍 Hyperparameter Tuning using Optuna (Grid, Random, Bayesian)

**🎯 Overview**
This project involves the development of an Artificial Neural Network (ANN) to predict match outcomes based on synthetic user-generated input features. The model can be utilized through:

⚙ Command-Line Interface (CLI) – Enables predictions via terminal using an inference script

🌐 Interactive Web App – Provides a user-friendly Streamlit interface, deployable locally or on Hugging Face Spaces

**📊 Data Collection**
Source: Synthetic (collected via Google Form)
Entries: 140+ responses
Cleaning: Dropped columns like Timestamp, Email Address

**🏷 Data Labeling**

Technique: KMeans Clustering (n_clusters=2)
Why: Automatically generates labels by uncovering hidden patterns in the data—no manual annotation or predefined rules required

**🔧 Data Preprocessing**

Label Encoding: Converted categorical labels to integers using LabelEncoder
Why: Efficient for small datasets with clear label classes
🧠 Model Architecture
Layer Type	Details
Input Layer	Shape: (10,)
Dense Layer #1	8 units, PReLU, Glorot Initialization
Dense Layer #2	3 units, PReLU, L2 Regularization
BatchNorm + Dropout	20% Dropout
Output Layer	1 unit, Sigmoid
Optimizer: Adam
Loss: Binary Crossentropy
Metrics: Accuracy, Precision, Recall

🌐 Streamlit App
🚀 Deployable on Hugging Face or run locally!:

📥 Accepts user input via dropdowns
🔮 Predicts match outcome with a confidence score
🧠 Supports real-time model training
💻 Hugging Face Demo:https://huggingface.co/spaces/Swathikiran/matching

streamlit run app.py

🧪 Inference Script (CLI)

Use the run.py script to make predictions from the terminal.


Built an inference script using argparse to allow CLI usage:


--weights_path: Path to trained model
--data_path: Path to prediction data (default: data.csv)
--num_preds: Number of samples to predict
Used Python's time module to measure prediction time.

Example:

python run.py --weights_path weights.h5 --data_path data.csv --num_preds 5

🧠 Output:

total time taken for 5 prediction is 0.0892 seconds

💘 It's a Match!
💔 No Match.

📈 Hyperparameter Optimization

Used Optuna for tuning with:

✅ Grid Search

🎲 Random Search

🔁 Bayesian Optimization

🔥 Best Result (Bayesian Optimization)
Accuracy: 93.33%

Validation Precision: 1.0000

Validation Recall: 0.8000

⚙ Technologies Used
Python 🐍

TensorFlow/Keras 🧠

Scikit-learn (LabelEncoder, KMeans)

Streamlit (UI)

Optuna (Tuning)

Argparse (CLI interface)

🚀 Getting Started
1️⃣ Clone the Repository:
git clone :https://github.com/swathikiran123/matching-nomatch_partner-
2️⃣ Install Dependencies:

3️⃣ Run Inference Script:

4️⃣ Launch Streamlit App:

streamlit run app.py

📂 Project Structure:

match-prediction-ann/
├── data/
│   ├── raw/                 # Raw synthetic responses (Google Form)
│   ├── cleaned/             # Cleaned & preprocessed datasets
│   └── data.csv            # Final dataset for training & inference
│
├── notebooks/
│   └── optuna DL.ipynb           # Exploratory Data Analysis notebook
│
├── models/
│   └── ann_model.h5        # Saved ANN weights
│
├── app/
│   ├── app.py             # Streamlit app script
│   └── utils.py           # Utility functions for app
│
├── scripts/
│   ├── train.py          # Script for model training
│   ├── run.py            # Inference script using argparse (CLI)
│   └── preprocess.py     # Data preprocessing & labeling script
│
├── config/
│   └── config.yaml       # Hyperparameters & global configs
│
├── requirements.txt     
├── README.md           
├── LICENSE
└── .gitignore

