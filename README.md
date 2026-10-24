# Smart Inventory Forecast - Streamlit Deployment Version

This folder contains an application ready for deployment to **Streamlit Community Cloud** to obtain a public web link.

## File Structure

```
SmartInvPredict_Streamlit/
├── app.py                  # Streamlit main application
├── requirements.txt        # Python dependencies
└── models/
    └── xgb_model.joblib    # XGBoost model (required)
```

## Local Testing

```bash
cd SmartInvPredict_Streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (Obtain Public Link)

1. Create a **public** GitHub repository and upload all files from this folder.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **New app** and select your repository.
4. Set Main file path to `app.py`.
5. Click **Deploy**.
6. After deployment completes, you will receive a public URL such as `https://your-app-name.streamlit.app`.

## Notes

- Prediction logic remains consistent with the original project's `python/ml_server.py`.
- The interface uses native Streamlit components, suitable for quick sharing and demos.
- The model file is approximately 2.2 MB and loads normally on Streamlit Cloud.
