# 🎓 AI-Based Career Recommendation System

An **AI-powered web application** designed to help students and professionals discover the most **suitable career paths** based on their **skills**, **interests**, and **educational background**.  

This project combines a **Machine Learning model** for career prediction with an interactive **Streamlit frontend** and a **Flask backend API** for seamless user experience.

---

## 💡 Project Overview

- 📊 **Machine Learning Algorithm**: `Random Forest Classifier`  
- 🌐 **Frontend**: `Streamlit`  
- 🗄️ **Backend API**: `Flask (served via Render)`  
- 🗂️ **Dataset**: Cleaned dataset with user skills, interests, and education levels  
- 🎯 **Goal**: Recommend the most suitable career path for users based on their profile

---

## 🚀 Live Demo

- 🌐 **Frontend (Streamlit):** [Open Streamlit App](https://ai-based-career-recommendation-system.streamlit.app/)  
- ⚙️ **Backend API (Flask on Render):** [Open Flask API](https://ai-based-career-recommendation-system.onrender.com)  

---

## 🧠 Tech Stack

- 🐍 **Python 3.9**  
- 📊 **Pandas, NumPy** for data processing  
- 🤖 **Scikit-learn** for machine learning models  
- 🌐 **Flask** for REST API backend  
- 🎨 **Streamlit** for frontend user interface  
- 🚀 **Render** for backend deployment  
- 📦 **Pickle** for model serialization  

---

## ✨ Features

- 📝 User-friendly form to input education, skills, and interests  
- 🤖 Predicts the most suitable career options using AI  
- 📊 Visualizes prediction confidence with interactive charts  
- 🌐 Deployed **frontend** and **backend** for live usage  

---

## 📤 Deployment

- **Backend (Flask):** Hosted on Render  
- **Frontend (Streamlit):** Deployed on Streamlit Cloud  

---

## 📚 Dataset

The dataset includes user information on:  
- Skills  
- Interests  
- Education levels  

*This dataset is custom-built for career recommendations (you can also link to a Kaggle dataset here if available).*  

---

## 📌 How It Works

1. User enters their details (name, age, education, skills, interests) into the **Streamlit form**.  
2. The data is sent via a **POST** request to the **Flask API**.  
3. Flask loads the trained machine learning model and predicts the most suitable career.  
4. Streamlit displays the recommendation along with visual insights like charts and graphs.  

---
