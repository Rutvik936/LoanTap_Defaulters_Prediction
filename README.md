# 💰 LoanTap Default Prediction

## 🧩 Problem Statement
LoanTap aims to predict whether a borrower will **fully repay** their loan or **default (Charged Off)** using financial and behavioral data.  
This helps the company **identify high-risk applicants early**, minimize losses, and strengthen lending strategies.

---

## 🎯 Objective
- Build machine learning models to classify loans as:
  - **0 → Fully Paid**
  - **1 → Charged Off (Default)**
- Handle **class imbalance** using **SMOTE**.
- Compare performance of **XGBoost** and **Random Forest** models.
- Focus on **recall for defaults**, since missing a defaulter is costlier than a false alarm.

---

## 📊 XGBoost Model Performance (LoanTap: Fully Paid vs Charged Off)

### ⚖️ Class Distribution
- **Before SMOTE:** `{Fully Paid: 254,686 , Charged Off: 62,138}`
- **After SMOTE:** `{Fully Paid: 254,686 , Charged Off: 254,686}` (Balanced dataset)

---

### ✅ Model Results (Threshold = 0.4)
**Accuracy:** 66.9%

|                | Predicted Fully Paid | Predicted Charged Off |
|----------------|----------------------|------------------------|
| **Actual Fully Paid** | 42,533 ✅ | 21,138 ❌ |
| **Actual Charged Off** | 5,077 ❌ | 10,458 ✅ |

---

### 📌 Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|------------|---------|-----------|----------|
| 0 (Fully Paid) | 0.89 | 0.67 | 0.76 | 63,671 |
| 1 (Charged Off) | 0.33 | 0.67 | 0.44 | 15,535 |
| **Macro Avg** | - | - | **0.60** | - |
| **Weighted Avg** | - | - | **0.70** | - |

---

### 🔎 Key Insights
- High **recall (0.67)** for *Charged Off* → effectively flags risky borrowers.  
- **Low precision (0.33)** → some safe loans incorrectly flagged as risky.  
- SMOTE balancing improved minority detection performance.  
- Model focuses more on catching defaulters than minimizing false positives.  
- Good for **risk-based screening** of new loan applicants.

---

## 🌲 Random Forest Model Performance

### 📊 Confusion Matrix

|                | Predicted 0 | Predicted 1 |
|----------------|-------------|-------------|
| **Actual 0** | 58,246 ✅ | 5,425 ❌ |
| **Actual 1** | 11,116 ❌ | 4,419 ✅ |

**Accuracy:** 79.12%

---

### 📈 Metrics
- **Class 0 (Fully Paid):** Precision **0.84** | Recall **0.91** | F1 **0.88**  
- **Class 1 (Charged Off):** Precision **0.45** | Recall **0.28** | F1 **0.35**

---

### 🔍 Insights
- Performs strongly for **Fully Paid** loans with high recall (0.91).  
- **Low recall (0.28)** for *Charged Off* → some defaulters missed.  
- **Moderate precision (0.45)** → false positives exist but are controlled.  
- Better accuracy overall (79%), slightly weaker at catching risky cases.  
- Suitable when you prefer **stability and fewer false alerts**.

---

## 🧩 Top 5 Key Takeaways

1. **SMOTE** successfully balanced the dataset, improving default detection.  
2. **XGBoost** achieved **higher recall (0.67)** — better at catching defaulters.  
3. **Random Forest** achieved **higher accuracy (79%)** and better stability.  
4. Both models show a **trade-off** between recall and precision.  
5. For risk prediction, **XGBoost is preferable** when identifying risky borrowers is the goal.

---

## ⚙️ Setup & Usage Instructions

### 1️⃣ Clone this repository
```bash
git clone https://github.com/your-username/LoanTap-Default-Prediction.git
cd LoanTap-Default-Prediction
```
### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Download the trained models
Open the provided .ipynb notebook in Google Colab.
Run all cells to generate and download the model files:
- rf_insurence.pkl
- xgb_insurence.pkl
Store both files in the same folder as your LoanTap_model.py.

📂 Project Folder Structure
```bash
 ┣ 📜 LoanTap_model.py
 ┣ 📜 rf_insurence.pkl
 ┣ 📜 xgb_insurence.pkl
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

### 4️⃣ Run the Streamlit app
```bash
streamlit run LoanTap_model.py
```
#### 🧠 Tech Stack
- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit
- 

💡 Note:
- To use this Streamlit app, download the trained models from the .ipynb notebook and place them in the same folder as LoanTap_model.py.
- The app won’t run without these model files.

```bash
👨‍💻 Author
Rutvik Mahadik
Data Science Project – LoanTap Default Prediction (Random Forest & XGBoost)
