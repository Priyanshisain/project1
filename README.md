# California House Price Prediction

Simple regression project I did while learning ML.

## What it does
Predicts median house value using California Housing dataset.

## Models
- Linear Regression (baseline)
- Random Forest
- Bonus: tried log-transform on target

## Results (typical run)
- Linear: RMSE ~0.84, R² ~0.58
- Random Forest: RMSE ~0.47, R² ~0.81

## How to run
```bash
pip install pandas matplotlib seaborn scikit-learn numpy
python house_price_prediction.py


If you get SSL certificate error on Windows when loading data:
pip install certifi


Then uncomment the two certifi lines at the top.
Made as practice for MSc AI/ML coursework.
text
if you get the SSL certificate error on Windows:
 # import certifi
# import os
# os.environ['SSL_CERT_FILE'] = certifi.where()
