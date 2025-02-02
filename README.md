
## end to end mlrepo
# Student Exam Performance Predictor - Flask API

## 📌 Project Overview
This is a Flask-based web application that predicts student exam performance based on input parameters such as gender, parental education, lunch type, test preparation course, and scores in reading and writing. The model uses machine learning techniques to provide accurate predictions.

## 📂 Project Structure
```
|-- app.py                  # Main Flask application
|-- templates/
|   |-- index.html          # Landing page
|   |-- home.html           # Prediction input page
|-- static/
|-- artifacts/
|   |-- model.pkl           # Trained machine learning model
|-- src/
|   |-- exception.py        # Custom exception handling
|   |-- logger.py           # Logging configuration
|   |-- utils.py            # Utility functions
|   |-- components/
|       |-- data_ingestion.py  # Data loading and preprocessing
|       |-- data_transformation.py  # Feature transformation pipeline
|       |-- model_trainer.py  # Model training and evaluation
|-- requirements.txt         # Required dependencies
|-- README.md                # Project documentation
```

## 🚀 Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed.

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Flask Application
```sh
python app.py
```

### 5️⃣ Access the Application
Open a browser and navigate to:
```
http://127.0.0.1:5000/
```

## 🎯 Usage
1. Navigate to the **Home** page.
2. Enter student details in the form.
3. Click **Predict** to see the predicted Math score.

## ⚙️ Model Training & Evaluation
- The machine learning model is trained using **RandomForestRegressor** and other regression models.
- Hyperparameter tuning is performed using **GridSearchCV**.
- The best model is saved in `artifacts/model.pkl`.

## 📜 API Endpoints
| Method | Endpoint     | Description  |
|--------|-------------|--------------|
| GET    | `/`         | Home Page |
| GET    | `/predict`  | Prediction Form |
| POST   | `/predict`  | Returns predicted math score |

## 📌 Contributing
Feel free to fork this repository, make changes, and submit a pull request!

## 🛠 Technologies Used
- **Python 3.8+**
- **Flask** (Backend)
- **Scikit-learn** (Machine Learning)
- **Pandas & NumPy** (Data Processing)
- **HTML, CSS** (Frontend)

## 🔗 License
This project is licensed under the **MIT License**.

