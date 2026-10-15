# CrewSignal

CrewSignal is a Flask-based prototype for personal health trend monitoring using Random Forest and SVM models. The app trains on startup using the included `Health data.csv` and provides a responsive dashboard to enter astronaut vitals and receive model predictions, metrics, and rule-based recommendations.

Quick start (local)

1. Create a virtual environment and activate it.

2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   python app.py

4. Open http://localhost:5000 in your browser. On first load the page will display a training message until models finish training.

Render deployment

- The repository includes a `Procfile` and `runtime.txt` and is ready to be connected to Render as a Web Service using the standard Python build.

Notes

- The application uses only the provided `Health data.csv` (synthetic) and does not require external datasets.
- No images are used in the UI.
