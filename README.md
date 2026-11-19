# 🤖 Machine Learning Projects

A collection of end-to-end Machine Learning and Deep Learning projects, spanning real-world data cleaning, exploratory data analysis (EDA), predictive modeling, pipeline engineering, and interactive web application deployments.

---

## 📂 Featured Projects

### 🚗 [Used Car Price Predictor](./Car%20price%20prediction)
> **Stack:** Python, Scikit-Learn, Pandas, NumPy, Flask, Bootstrap 4, AJAX  
> **Status:** Completed & Deployed Locally  
> **Documentation:** [Full Project README](./Car%20price%20prediction/README.md)

An end-to-end regression application that predicts the resale price of pre-owned cars in the Indian automotive market based on brand, model, purchase year, kilometers driven, and fuel type.

#### 🌟 Highlights:
- **Cleaned Real-World Web Data:** Transformed 892 raw Quikr car records down to 815 clean instances, fixing corrupted text fields, stripping currency commas, pruning multi-word vehicle names to 3 tokens, and removing multi-crore outliers.
- **Scikit-Learn Pipeline:** Combined `ColumnTransformer`, `OneHotEncoder`, and `LinearRegression` into a single reproducible pipeline attaining an $R^2$ score of **~0.846 (84.6%)**.
- **Interactive Web App:** Flask application with asynchronous AJAX predictions and dynamically cascading model dropdowns filtered by car manufacturer.

---

### 📸 Application Interface Preview

| Input Dashboard | Live Price Estimation |
| :---: | :---: |
| ![Car Price Predictor UI](./Car%20price%20prediction/assets/ui_home.png) | ![Car Price Predictor Result](./Car%20price%20prediction/assets/ui_prediction.png) |

---

## 🛠️ Repository Structure

```text
Machine-learning-Projects/
│
├── Car price prediction/               # Used Car Price Prediction ML Project
│   ├── assets/                         # UI screenshots and application visuals
│   │   ├── ui_home.png                 # Web form interface
│   │   └── ui_prediction.png           # Dynamic valuation result
│   ├── templates/                      # Jinja2 frontend templates
│   │   └── index.html                  # Responsive UI with dynamic JS filters
│   ├── static/                         # CSS styling files
│   ├── app.py                          # Flask backend service & REST API
│   ├── car price prediction.ipynb      # EDA, data cleaning, modeling & pipeline
│   ├── cleaned car data.csv            # Preprocessed dataset
│   ├── quikr_car.csv                   # Raw Quikr dataset
│   ├── LinearModelPrediction.pkl        # Serialized Scikit-Learn model pipeline
│   └── README.md                       # In-depth project documentation
│
└── README.md                           # Portfolio overview (this file)
```

---

## ⚡ Quickstart

To run any project, navigate into its directory and follow its project-specific instructions:

```bash
# Clone the repository
git clone https://github.com/dhruv/Machine-learning-Projects.git
cd Machine-learning-Projects

# Navigate to Car Price Prediction
cd "Car price prediction"

# Install requirements and run
pip install flask flask-cors pandas numpy scikit-learn
python app.py
```

---

## 📜 License
Distributed under the [MIT License](https://opensource.org/licenses/MIT).
