import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# -------- Load model and encoders --------
model           = pickle.load(open("model.pkl", "rb"))
edu_enc         = pickle.load(open("edu_encoder.pkl", "rb"))
skill_enc       = pickle.load(open("skills_encoder.pkl", "rb"))
int_enc         = pickle.load(open("interests_encoder.pkl", "rb"))
career_enc      = pickle.load(open("career_encoder.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template(
        "index.html",
        skill_options=sorted(skill_enc.classes_),
        interest_options=sorted(int_enc.classes_),
        edu_options=list(edu_enc.classes_)
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    name       = data.get("name")
    age        = int(data.get("age", 0))
    education  = data.get("education")
    skills     = data.get("skills", [])
    interests  = data.get("interests", [])

    # Validate inputs
    if not name or not education or not skills or not interests:
        return jsonify({"error": "Please fill all fields correctly."})

    if education not in edu_enc.classes_:
        return jsonify({"error": "Invalid education selected."})

    # Encode inputs
    edu_code = edu_enc.transform([education])[0]

    try:
        skills_arr = skill_enc.transform([skills])
    except ValueError:
        return jsonify({"error": "One or more selected skills are invalid."})
    
    try:
        ints_arr = int_enc.transform([interests])
    except ValueError:
        return jsonify({"error": "One or more selected interests are invalid."})

    # Create DataFrame for prediction
    skills_df = pd.DataFrame(skills_arr, columns=skill_enc.classes_).add_prefix("Skills_enc_")
    ints_df = pd.DataFrame(ints_arr, columns=int_enc.classes_).add_prefix("Interests_enc_")

    X_input = pd.DataFrame([[age, edu_code]], columns=["Age", "Education_enc"])
    X_input = pd.concat([X_input, skills_df, ints_df], axis=1)

    # Add any missing columns with zero and align with training data
    for col in feature_columns:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[feature_columns]

    # Make prediction
    predicted_code = model.predict(X_input)[0]
    predicted_career = career_enc.inverse_transform([predicted_code])[0]

    return jsonify({"career": predicted_career})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
