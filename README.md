# Smart Inventory Forecast - Enhanced Streamlit Version

This version brings the Streamlit app closer to the original `SmartInvPredict` features while remaining fully deployable on Streamlit Community Cloud.

## Features

- **Home** — Introduction and workflow overview
- **Forecast** — Full input form (product, stock, unit cost, lead time, season, promotion)
- **CSV import** for historical sales
- **Chart** comparing historical sales vs. forecast
- **Metrics** — change %, total demand, average, peak period, confidence
- **Smart alerts** — restock / promotion / caution recommendations
- **History** — session-based list of previous forecasts (view / delete)
- Uses the original **XGBoost** model (`models/xgb_model.joblib`)

## File Structure

```
SmartInvPredict_Streamlit/
├── app.py
├── requirements.txt
└── models/
    └── xgb_model.joblib
```

## Local Testing

```bash
cd SmartInvPredict_Streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Upload the folder contents to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Create a new app, select the repository, and set Main file path to `app.py`.
4. Deploy to obtain a public link.

## Notes

- History is stored in the Streamlit session (cleared on restart or new session).
- Prediction logic matches the original `python/ml_server.py`.
- Results are for demonstration purposes only.
