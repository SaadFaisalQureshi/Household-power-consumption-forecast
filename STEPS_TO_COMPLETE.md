# Assignment 3 – Steps to Complete (Household Energy Demand Forecasting)

Deadline: **Sunday 25 Oct 2026, 23:59**. Allow about 1–2 hours in total.

## What's in this package

| File | Purpose |
|---|---|
| `household_power_kfold_full_dataset_10fold.ipynb` | Task 2 notebook in the same layout as the Heart Disease notebook (Steps 1–11, Table of Content, deployment section). Already run once, outputs included |
| `household_power_model.pkl` | Trained XGBoost model (whole pipeline) |
| `app.py` | Streamlit web app for deployment |
| `requirements.txt` | Library versions the web app needs (they match the `.pkl`) |
| `household_power_consumption.csv` | Full dataset in CSV form (UCI, 2,075,259 rows, 112 MB) |
| `household_power_consumption_csv.zip` | The same CSV zipped (19 MB), quicker to upload to Colab and Classroom |

## Results at a glance (10-fold CV, RMSE in kW, lower is better)

| Rank | Algorithm | Mean RMSE | Notes |
|---|---|---|---|
| 1 | XGBoost | 0.506 | Wins all 10 folds; 25% better than the naive baseline |
| 2 | Extra Trees | 0.523 | |
| 3 | MLP Neural Network | 0.540 | |
| 4 | Ridge Regression | 0.578 | Linear model, so it can't capture the non-linear patterns |
| 5 | AdaBoost | 0.901 | Worse than the baseline (0.679) because it over-predicts |

None of these 5 algorithms is used in the Car Price notebook (that one uses Linear Regression, KNN, Decision Tree, Random Forest and Gradient Boosting).

---

## Step 1: Task 1 (run the instructor's Car Price notebooks)

1. Open https://colab.research.google.com, then **File → Upload notebook** → `car_price_kfold_100_instances_3fold.ipynb`.
2. **Runtime → Run all**, and upload `car_price.csv` when the button appears.
3. Repeat for `car_price_kfold_full_dataset_10fold.ipynb`.
4. Read the outputs so you understand K-fold, RMSE and the comparison table. Your Task 2 notebook follows the same structure, which makes viva questions easier to answer.

## Step 2: Personalise and run the Task 2 notebook on Colab

1. In Colab: **File → Upload notebook** → `household_power_kfold_full_dataset_10fold.ipynb`.
2. The **Developer & System Information** table already shows Saad Faisal (FA24-BSE-104).
3. **Runtime → Run all**. When Step 2.1 shows **Choose Files**, select `household_power_consumption_csv.zip` (faster) or `household_power_consumption.csv`. The run takes about 3–6 minutes.
4. Check that your numbers are close to the table above. Small differences are normal if Colab uses different library versions. If the **ranking** changes, edit the numbers in **Step 7.5 Observations** and the **Conclusion**.
5. **File → Download → Download .ipynb** to save the notebook with its outputs.

> Make sure you can explain these points: why the folds are **not shuffled** (it's time-series data, so shuffling would leak neighbouring hours into training), why there's a **baseline**, and why **AdaBoost** is worse than the baseline.

## Step 3: Deploy the web app (free, about 15 minutes)

We use **Streamlit Community Cloud**.

1. Create a free **GitHub** account if you don't have one, then click **New repository**. Name it e.g. `household-energy-forecast`, set it to **Public**, and create it.
2. Click **Add file → Upload files** and upload these three files **together**:
   - `app.py`
   - `requirements.txt`
   - `household_power_model.pkl`

   Then click **Commit changes**.
   > Upload the `.pkl` and `requirements.txt` from **this package** together. If you would rather use the model from your own Colab run, download *both* `household_power_model.pkl` **and** `requirements.txt` from that run (set `DOWNLOAD_FILES = True` in Step 8.3). The model only loads with the same library versions it was saved with.
3. Go to https://share.streamlit.io, sign in **with GitHub**, then click **Create app → Deploy a public app from GitHub**.
4. Fill in: Repository = your repo, Branch = `main`, Main file path = `app.py`.
5. Open **Advanced settings** and choose **Python 3.12** (or newer).
6. Give the app a custom URL, e.g. `household-energy-yourname`, then click **Deploy**. The first build takes 2–5 minutes.
7. When the app opens, click **Predict next-hour consumption** to check that it works.
8. Copy the link, e.g. `https://household-energy-yourname.streamlit.app`.

*Alternative:* Hugging Face Spaces (choose the "Streamlit" SDK and upload the same 3 files).

## Step 4: Put the deployment link at the top of the notebook

1. In Colab, open **Edit → Find and replace**. Replace `PASTE-YOUR-LINK-HERE.streamlit.app` with your real address (e.g. `household-energy-yourname.streamlit.app`) and click **Replace all**. The link appears at the top, in the Developer table and in Step 11.
2. Download the `.ipynb` again.

## Step 5: Submit on Google Classroom

Upload:
- [ ] `household_power_kfold_full_dataset_10fold.ipynb` (with outputs and the live link at the top)
- [ ] `household_power_consumption.csv` (dataset; if it's too large to upload, use `household_power_consumption_csv.zip`)
- [ ] `household_power_model.pkl` (trained model)
- [ ] `app.py` and `requirements.txt` (deployment code)
- [ ] Optional: the two Car Price notebooks you ran for Task 1

## Troubleshooting

| Problem | Fix |
|---|---|
| Colab: `ModuleNotFoundError: xgboost` or `prettytable` | Add a cell at the top with `!pip install xgboost prettytable` and run it |
| Streamlit build fails on a version | Make sure `requirements.txt` and the `.pkl` come from the **same** run, and the Python version is 3.12+ |
| App error "No such file household_power_model.pkl" | The `.pkl` must be in the repo root, next to `app.py` |
| App sleeps after a few days | Free Streamlit apps sleep when unused. Open the link and click "wake up" before your demo |
