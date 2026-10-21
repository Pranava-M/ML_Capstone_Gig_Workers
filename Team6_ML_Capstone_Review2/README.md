# ML Capstone Project — Gig Workers

An interactive machine-learning application developed as part of the ML Capstone Review 2 project. The application explores how machine-learning models can be used to analyze gig-worker data and generate predictions through a Streamlit web interface.

## 🚀 Live Application

**[Click here to open the deployed application](https://mlcapstonegigworkers-aerefhkb7voxzvkfxssk7i.streamlit.app/)**

## 📌 Project Overview

Gig workers often work across digital platforms and have different activity patterns, performance indicators, and work histories. Analyzing these factors with machine learning can help demonstrate how data-driven classification and prediction systems work.

This project provides a web interface for exploring trained machine-learning models and their predictions using Python and Streamlit.

## 🎯 Project Objectives

- Build an interactive machine-learning application.
- Load trained models and use them to generate predictions.
- Demonstrate classification algorithms and their prediction workflows.
- Provide an accessible interface for interacting with model outputs.
- Deploy the application online using Streamlit Community Cloud.

## 🧠 Machine-Learning Algorithms

The project review covers the following classification algorithms:

### 1. Gradient Boosting Classifier
Builds an ensemble of decision trees sequentially, with each new tree attempting to correct errors made by the existing ensemble.

### 2. Bagging Classifier
Trains multiple models on different bootstrap samples of the training data and combines their predictions to improve stability.

### 3. MLP Classifier (Neural Network)
Uses interconnected layers of artificial neurons to learn patterns in the input features and predict a target class.

The application uses saved model artifacts where configured. Refer to the source code and model files for the exact implementation and prediction targets.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine-learning algorithms |
| Joblib | Loading saved machine-learning models |
| Matplotlib & Seaborn | Data visualization |
| Plotly | Interactive visualizations |

## 📂 Project Structure

```text
ML_Capstone_Gig_Workers/
└── Team6_ML_Capstone_Review2/
    ├── requirements.txt
    └── app/
        ├── app.py
        ├── requirements.txt
        └── models/
            └── Saved model files
```

*Note: This is an illustrative structure. Actual file and model locations may differ in the repository.*

## 💻 Run Locally

### Prerequisites

- Python installed on your system
- Git
- A code editor such as VS Code

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Pranava-M/ML_Capstone_Gig_Workers.git
   ```

2. Navigate to the application directory:

   ```bash
   cd ML_Capstone_Gig_Workers/Team6_ML_Capstone_Review2/app
   ```

3. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

4. Install the application dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL displayed in the terminal, usually `http://localhost:8501`.

Ensure the saved model files and any required datasets are present at the locations expected by the application.

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud and connected to the project's GitHub repository.

**Live URL:**  
https://mlcapstonegigworkers-aerefhkb7voxzvkfxssk7i.streamlit.app/

## 👥 Project Information

- **Project:** ML Capstone — Gig Workers
- **Review:** Review 2
- **Repository:** [ML_Capstone_Gig_Workers on GitHub](https://github.com/Pranava-M/ML_Capstone_Gig_Workers)
- **Application:** [Open Live Demo](https://mlcapstonegigworkers-aerefhkb7voxzvkfxssk7i.streamlit.app/)

---

*Developed for academic learning and demonstration of machine-learning concepts and application deployment.*
