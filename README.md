# Uber Trip Analysis & Demand Forecasting

This repository contains the code, analysis, and findings from a machine learning project focused on forecasting Uber trip demand using 2024 ride data. By combining data engineering, time-series feature creation, and ensemble modeling, this project establishes a strong baseline for operational forecasting.

---

## 📌 Project Overview
The proliferation of ride-hailing services like Uber generates large amounts of trip data. Accurate demand forecasting is critical for optimizing driver allocation, improving customer experience, and supporting dynamic pricing.  

This project applies supervised learning techniques to predict hourly Uber trip counts based on past demand patterns.

---

## 🗂 Dataset
- **Source:** Kaggle — *Uber Ride Analytics Dashboard*  
- **File used:** `ncr_ride_bookings.csv`  
- **Size:** 150,000 ride records with 21 features from the year 2024  
- **Feature Categories:**
  - **Temporal & Transactional Identifiers:** Date, Time, Booking ID, Customer ID  
  - **Ride Characteristics & Logistics:** Booking Status, Vehicle Type, Pickup & Drop Locations, Payment Method  
  - **Performance & Financial Metrics:** Ride Distance, Booking Value, Avg VTAT/CTAT  
  - **Feedback & Cancellation Data:** Driver Ratings, Customer Ratings, Cancellation Reasons  

---

## 🎯 Project Objectives
1. **Data Exploration & Preprocessing** – Understand and clean the 2024 Uber trip data.  
2. **Feature Engineering** – Aggregate trips hourly and engineer temporal features (hour, weekday, month).  
3. **Model Training** – Train XGBoost, Gradient Boosted Tree Regressor (GBRT), and Random Forest models.  
4. **Model Evaluation** – Evaluate each model with SMAPE (Symmetric Mean Absolute Percentage Error).  
5. **Ensemble Techniques** – Combine models using weighted averaging for enhanced accuracy.  
6. **Comparative Analysis** – Compare individual and ensemble performance visually and numerically.  

---

## ⚙️ Methods & Tools
- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib  
- **Approach:**  
  - Aggregated hourly trip counts  
  - Created lagged features (past 24 hours) to predict the next hour  
  - Chronological train/test split  
  - Trained tree-based models (XGBoost, Random Forest, GBRT)  
  - Evaluated using SMAPE for stability with zero-demand hours  
  - Built an ensemble model weighted by inverse error  

---

## 📊 Model Performance (SMAPE %)

| Model         | SMAPE (%) |
|---------------|-----------|
| Random Forest | 25.64%    |
| Ensemble      | 25.73%    |
| XGBoost       | 26.19%    |
| GBRT          | 26.25%    |

- **Key Finding:** Random Forest was the best standalone model by SMAPE, but the Ensemble offered slightly improved stability and reduced extreme errors, making it the best choice for operational deployment.

---

## 🔑 Insights & Recommendations
- All models successfully captured cyclical demand patterns, with peaks during weekday rush hours and weekend evenings.  
- Ensemble predictions smoothed out extreme deviations and improved robustness over individual models.  
- **Next Steps:**  
  - Incorporate external signals (weather, holidays, events) to further improve forecast precision.  
  - Experiment with advanced time-series architectures (LSTMs, Temporal Fusion Transformers).  
  - Integrate the ensemble into a real-time forecasting pipeline for dynamic pricing and driver dispatch.  
  - Continuously retrain with new data to adapt to evolving patterns.  

---

## 📂 Repository Structure
```text
├── data/
│   └── ncr_ride_bookings.csv
├── notebooks/
│   └── Uber_Trip_Analysis.ipynb
├── README.md
└── reports/
    └── Project Report_Uber Trip Analysis & Demand Forecasting.pdf
```

---

## 🔗 References
- [Google Colab Notebook](https://colab.research.google.com/drive/1N9LwROMpxrHFhEXuF1r5YGnD4aEn2HyA?usp=sharing)  
- [GitHub Repository](https://github.com/yashsrivastava05/Uber-Trip-Analysis)  

---

## 🙌 Acknowledgements
Dataset provided by the NYC Taxi & Limousine Commission (TLC) and obtained by FiveThirtyEight through a Freedom of Information Law request.

---

## 📝 Author
**Yash Srivastava**  
Project report & analysis authored as part of an advanced data analytics exercise.

