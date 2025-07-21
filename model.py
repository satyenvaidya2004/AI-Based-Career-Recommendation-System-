import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------- Load Dataset --------
data = pd.read_csv("AI-based Career Recommendation System.csv")

# -------- Encode Education --------
edu_encoder = LabelEncoder()
data['Education_enc'] = edu_encoder.fit_transform(data['Education'])

# -------- Split Skills and Interests --------
data['Skills_split'] = data['Skills'].apply(lambda x: x.split(';'))
data['Interests_split'] = data['Interests'].apply(lambda x: x.split(';'))

# -------- One-hot Encode Skills --------
skills_encoder = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(skills_encoder.fit_transform(data["Skills_split"]),columns=skills_encoder.classes_).add_prefix("Skills_enc_")

# -------- One-hot Encode Interests --------
interests_encoder = MultiLabelBinarizer()
interests_encoded = pd.DataFrame(interests_encoder.fit_transform(data["Interests_split"]),columns=interests_encoder.classes_).add_prefix("Interests_enc_")

# -------- Encode Target Variable --------
career_encoder = LabelEncoder()
data["Recommended_Career_enc"] = career_encoder.fit_transform(data["Recommended_Career"])

# -------- Combine All Features --------
X = pd.concat([data[["Age", "Education_enc"]], skills_encoded, interests_encoded],axis=1)
y = data["Recommended_Career_enc"]

# -------- Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- Train RandomForestClassifier --------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------- Evaluate Model --------
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------- Save Model and Encoders --------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(edu_encoder, open("edu_encoder.pkl", "wb"))
pickle.dump(skills_encoder, open("skills_encoder.pkl", "wb"))
pickle.dump(interests_encoder, open("interests_encoder.pkl", "wb"))
pickle.dump(career_encoder, open("career_encoder.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("feature_columns.pkl", "wb"))

print("\n✅ Model and encoders saved successfully.")