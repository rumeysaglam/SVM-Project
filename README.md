# 🧠 AI-Based Hiring Decision System using SVM

This project is an intelligent hiring decision support system designed for HR departments. It predicts whether a candidate should be hired for a software developer position based on **years of experience** and **technical test score**, using a Support Vector Machine (SVM) model.

---

## 🚀 Project Summary

- **ML Algorithm**: Support Vector Machine (SVM)
- **ML Task**: Binary classification
- **Data**: Synthetic data with 200 samples
- **Features**:
  - `experience_years`: 0 to 10
  - `technical_score`: 0 to 100
- **Labeling Rule**:
  - Hired (`1`): If experience ≥ 2 **and** technical score ≥ 60
  - Not Hired (`0`): Otherwise

---

## 📂 Project Structure

SVM_Project/
│
├── SVM_Project.py # Data generation, model training, evaluation & visualization
├── FastAPI.py # FastAPI app for model serving
├── model.joblib # Saved SVM model
├── scaler.joblib # Saved StandardScaler
├── README.md # Project documentation

## 🛠 Dependencies

- Python 3.8+
- FastAPI
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib
- uvicorn (for running the API)

## 📊 Model Training (SVM_Project.py)
The following steps are performed:

1. Synthetic data generation using NumPy.

2. Labeling based on rule-based criteria.

3. Train-test split using train_test_split.

4. Feature scaling using StandardScaler.

5. Model training with SVC(kernel='linear').

6. Model evaluation using:

    Accuracy

    Confusion matrix

    Classification report

7. Decision boundary visualization using matplotlib.

8. Model saving with joblib.

## 🔗 REST API (FastAPI.py)
# 📍 Endpoints:
Method	URL	        Description
 GET	   /	      Root information & usage guide
 GET	 /test	    Sample predictions
 POST	 /predict	  Make prediction for an applicant

# Example Request:

POST /predict
{
  "experience_years": 3,
  "technical_score": 75
}

# Example Response:

{
  "prediction": "✅ Hired",
  "prediction_value": 1,
  "experience": 3,
  "technical_score": 75
}

# Run the API:

uvicorn FastAPI:app --reload

Access the documentation: http://127.0.0.1:8000/docs

## Future Improvements
Try other SVM kernels (e.g., rbf, poly) to handle non-linear relationships.

Add hyperparameter tuning using GridSearchCV.

Deploy as a Docker container or on a cloud platform.

Add user interface using Streamlit or React.

## 👩‍💻 Author
Name: Rumeysa Sağlam

Field: Computer Engineering

Focus: Web development & Artificial Intelligence

## 📬 Contact
For questions or collaborations, feel free to reach out via GitHub or LinkedIn.
