#  Customer Satisfaction Prediction Project

> This project is part of **our ongoing Data Analysis Series**, where we explore real-world datasets, apply data science techniques, and develop deployable machine learning solutions.

---

## 📊 Project Overview

In this project, we built an **end-to-end machine learning pipeline** to predict **Customer Satisfaction Ratings (1 to 5)** based on support ticket data, including both structured and unstructured inputs. After processing and modeling the dataset, we deployed a working web application using **Flask** that allows users to interactively input data and get real-time satisfaction predictions.

A custom HTML/CSS interface was developed to provide a clean and intuitive UI.

---

## 📂 Dataset Information

* **File**: `customer_support_tickets.csv`
* **Features**:

  * Ticket metadata (e.g., type, priority, channel)
  * Customer demographics (e.g., age, gender)
  * Textual descriptions of issues
  * Satisfaction rating (1-5) \[Target Variable]
  
---

## 🛠️ Tools & Technologies Used

| Type              | Tools / Libraries                                                                  |
|-------------------|-------------------------------------------------------------------------------------|
| **Language**       | Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP)           |
| **NLP & Text**     | VADER Sentiment Analysis, TF-IDF Vectorization                                     |
| **ML Techniques**  | Random Forest, XGBoost, SMOTE (for class balancing), Cross-Validation              |
| **Deployment**     | Flask (API), HTML/CSS (Frontend), Jinja Templates                                  |
| **IDE/Notebook**   | Google Colab, Visual Studio Code                                                   |
| **Dataset Format** | CSV (Customer Support Tickets)                                                     |
| **Model Storage**  | joblib (`.pkl` files), JSON (`.json` for XGBoost model)                            |
| **Frontend Assets**| `/static` for images, `/templates` for HTML forms                                 |

---

## 📈 Workflow in the Notebook (`Customer_Satisfaction_Prediction.ipynb`)

### ✅ Step 1: Data Collection & Exploration

* Loaded raw dataset and inspected structure, types, and null values.

### ✅ Step 2: Data Cleaning and Preprocessing

* Removed invalid entries and focused on 'Closed' tickets with satisfaction ratings.
* Handled missing values, standardized column names.

### ✅ Step 3: Exploratory Data Analysis (EDA)

* Visualized distribution of ratings, customer demographics, and ticket types.
* Analyzed key patterns in satisfaction levels across different ticket types and genders.

### ✅ Step 4: Feature Engineering

* Created features like `Response Time Delta`, `Tickets per Customer`, `Months Since Start`
* Added sentiment scores using VADER
* Applied **TF-IDF** to ticket descriptions
* Performed **Label Encoding** and **One-Hot Encoding** on categorical variables

### ✅ Step 5: Model Building

* Scaled numerical features using `StandardScaler`
* Handled class imbalance using **SMOTE**
* Trained **Random Forest** and **XGBoost** classifiers
* Saved the final XGBoost model and preprocessing tools using `json`

### ✅ Step 6: Model Evaluation

* Evaluated using **accuracy**, **precision**, **recall**, **f1-score**, and confusion matrix
* Analyzed accuracy by ticket type category
* Final accuracy: \~22%

### ✅ Step 7: Model Explainability

* Applied **SHAP** to understand global feature importance
* Identified top predictors: `Months Since Start`, `Product Purchased`, and `Sentiment`

---

## 🌐 Web App Deployment (`Customer_Satisfaction_Prediction_Project-FLASK_APP`)

### 🔧 Flask Backend (`app.py`)

* Loaded the trained XGBoost model, scaler, and training columns
* Created two routes:

  * `/` renders an HTML form (UI)
  * `/predict` accepts POST request, processes form data, scales inputs, and returns prediction
* Handles missing columns dynamically to match model training features
* Renders predicted rating back to the user in the UI

### 🏢 Frontend Interface (`templates/index.html`)

* HTML form collects input fields like:

  * Customer Age
  * Product Purchased
  * Sentiment
  * Response Time
  * Gender
  * Ticket Type
  * Tickets per Customer
  * Months Since Start
  * Age Group
* Clean and user-friendly layout with dropdowns, number inputs, and placeholders

### 🌈 Styling and Image Support (`static/img.jpg`)

* Custom **CSS styling** applied via `<style>` tags in HTML
* Background image set using `img.jpg` placed in `static/` directory

---

## 📁 Repository Structure

```
Customer_Satisfaction_Prediction_Project-FLASK_APP/
├── static/
│   └── img.jpg                      # Background image used in UI
├── templates/
│   └── index.html                  # HTML form for input and output display
├── app.py                          # Main Flask app
├── customer_satisfaction_model_xgb.json  # Trained XGBoost model
├── scaler.pkl                      # StandardScaler used in model pipeline
├── training_columns.pkl            # List of training columns used during model training

Customer_Satisfaction_Prediction.ipynb    # Complete ML notebook pipeline
customer_support_tickets.csv              # Original dataset
README.md                                 # Project documentation (this file)
```

---

## 📊 Final Results

* **Model**: XGBoost Classifier
* **Final Accuracy**: \~22%
* **Balanced class-wise performance**
* **Improved accuracy** on specific categories like Cancellation Requests (\~28%)
* **SHAP Explainability** validated important features like Sentiment & Time-based metrics

---

## 💡 Future Enhancements

* Use advanced NLP (BERT, topic modeling) for better text understanding
* Integrate live feedback loop to improve model with time
* Deploy on cloud platforms (e.g., Render, Heroku)
* Add user authentication and logging for production use

---

## 🌐 HTML Page Preview

> The interactive web UI can be previewed by running the Flask app and visiting: [http://127.0.0.1:5000](http://127.0.0.1:5000)

> **Screenshot:** 

![Customer Satisfaction UI Preview](page preview.png)


---

## 🧠 Recommendations

| Area                      | Recommendation                                                                  |
| ------------------------- | ------------------------------------------------------------------------------- |
| **Model Improvement**     | Experiment with **advanced NLP models** like BERT for text features             |
| **Data Enrichment**       | Incorporate **customer history** or external feedback scores for better context |
| **Feature Expansion**     | Add more **time-based features** or ticket resolution stats                     |
| **UI/UX Enhancements**    | Add **dropdowns and sliders** in the web form for better user experience        |
| **Model Monitoring**      | Set up logging to track **prediction quality** over time                        |
| **Feedback Loop**         | Let users submit actual satisfaction ratings to **retrain the model**           |
| **Cloud Deployment**      | Host the app using **Render, Heroku, or AWS** for public access                 |
| **Security & Validation** | Add input validation and **error handling** for production robustness           |

---


## 🚀 Part of the "Data Analysis Series"

This repository is the latest addition to our Data Analysis Series which also includes:

* YouTube Performance Analytics (YT360 Project)
* Olympics Medal Analysis
* Coffee Sales Forecasting
* Iris Classification
* and many more...

---

## 🙋‍♂️ Author

Made by Hashir khan   
Feel free to ⭐ the repo if you found it helpful!
