from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import os
import re
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import threading

app = Flask(__name__)

rf_model = None
svm_model = None
rf_metrics = {}
svm_metrics = {}
rf_cv = None
svm_cv = None
labels_map = {}
training_complete = False
training_lock = threading.Lock()


def load_and_train():
    global rf_model, svm_model, rf_metrics, svm_metrics, rf_cv, svm_cv, labels_map, training_complete
    path = os.path.join(os.path.dirname(__file__), 'Health data.csv')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cleaned = []
    for line in lines:
        cleaned.append(re.sub(r"^\s*\d+\|\s*", "", line))
    data = StringIO(''.join(cleaned))
    df = pd.read_csv(data, skipinitialspace=True)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    col_variants = {c.replace('_', ''): c for c in df.columns}
    pulse_col = col_variants.get('pulse')
    temp_col = col_variants.get('bodytemperature') or col_variants.get('bodytemp') or col_variants.get('body_temperature')
    spo2_col = col_variants.get('spo2') or col_variants.get('spo_2')
    status_col = col_variants.get('status')
    if not all([pulse_col, temp_col, spo2_col, status_col]):
        raise RuntimeError('dataset missing required columns')

    df = df[[pulse_col, temp_col, spo2_col, status_col]].copy()
    df.columns = ['pulse', 'body_temperature', 'spo2', 'status']
    df = df.dropna()
    df['pulse'] = pd.to_numeric(df['pulse'], errors='coerce')
    df['body_temperature'] = pd.to_numeric(df['body_temperature'], errors='coerce')
    df['spo2'] = pd.to_numeric(df['spo2'], errors='coerce')
    df['status'] = pd.to_numeric(df['status'], errors='coerce').astype(int)
    df = df.dropna()

    unique_status = sorted(df['status'].unique())
    names = ['Normal', 'Moderate', 'Critical']
    labels_map = {status: names[idx] if idx < len(names) else f'Status_{idx}' for idx, status in enumerate(unique_status)}

    X = df[['pulse', 'body_temperature', 'spo2']].values
    y = df['status'].values

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42)
    svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42))

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    rf_model = rf
    svm_model = svm

    y_rf = rf.predict(X_test)
    y_svm = svm.predict(X_test)

    rf_metrics.update({
        'accuracy': float(accuracy_score(y_test, y_rf)),
        'precision': float(precision_score(y_test, y_rf, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_rf, average='macro', zero_division=0)),
        'f1': float(f1_score(y_test, y_rf, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_rf).tolist(),
    })

    svm_metrics.update({
        'accuracy': float(accuracy_score(y_test, y_svm)),
        'precision': float(precision_score(y_test, y_svm, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_svm, average='macro', zero_division=0)),
        'f1': float(f1_score(y_test, y_svm, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_svm).tolist(),
    })

    try:
        rf_cv = float(np.mean(cross_val_score(rf, X, y, cv=5, scoring='accuracy')))
    except Exception:
        rf_cv = float(np.mean(cross_val_score(rf, X, y, cv=3, scoring='accuracy')))

    try:
        svm_cv = float(np.mean(cross_val_score(svm, X, y, cv=5, scoring='accuracy')))
    except Exception:
        svm_cv = float(np.mean(cross_val_score(svm, X, y, cv=3, scoring='accuracy')))

    globals()['rf_cv'] = rf_cv
    globals()['svm_cv'] = svm_cv
    globals()['training_complete'] = True


def start_training():
    global training_complete
    with training_lock:
        if training_complete:
            return
    thread = threading.Thread(target=load_and_train, daemon=True)
    thread.start()


start_training()


@app.route('/')
def index():
    if not training_complete:
        return render_template('loading.html')
    return render_template('index.html')


@app.route('/training_status')
def training_status():
    return jsonify({'status': 'done' if training_complete else 'training'})


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if not training_complete:
        return redirect(url_for('index'))

    aid = request.form.get('astronaut_id', '').strip()
    name = request.form.get('astronaut_name', '').strip()
    age = request.form.get('age', '').strip()
    sex = request.form.get('sex', '').strip()
    pulse = request.form.get('pulse', '').strip()
    body_temp = request.form.get('body_temperature', '').strip()
    spo2 = request.form.get('spo2', '').strip()

    if not aid.isalnum():
        return redirect(url_for('index'))
    if not name.replace(' ', '').isalnum():
        return redirect(url_for('index'))
    if not age.isdigit() or int(age) < 0:
        return redirect(url_for('index'))

    try:
        pulse_v = float(pulse)
        temp_v = float(body_temp)
        spo2_v = float(spo2)
        if pulse_v < 0 or temp_v < 0 or spo2_v < 0:
            return redirect(url_for('index'))
    except Exception:
        return redirect(url_for('index'))

    features = np.array([[pulse_v, temp_v, spo2_v]])
    rf_pred = int(rf_model.predict(features)[0])
    svm_pred = int(svm_model.predict(features)[0])
    rf_proba = rf_model.predict_proba(features)[0]
    svm_proba = svm_model.predict_proba(features)[0]
    rf_conf = float(np.max(rf_proba))
    svm_conf = float(np.max(svm_proba))

    svm_decision = 0.0
    try:
        dec = svm_model.named_steps['svc'].decision_function(features)
        if hasattr(dec[0], '__len__'):
            svm_decision = float(np.max(np.abs(dec[0])))
        else:
            svm_decision = float(dec[0])
    except Exception:
        svm_decision = 0.0

    agreement = rf_pred == svm_pred

    if spo2_v < 90 or pulse_v > 120 or temp_v > 38:
        risk = 'High'
        next_step = 'Escalate to medical protocol'
    elif spo2_v < 95 or pulse_v > 100 or temp_v > 37.5:
        risk = 'Moderate'
        next_step = 'Repeat measurement and monitor closely'
    else:
        risk = 'Low'
        next_step = 'Routine monitoring'

    return render_template(
        'result.html',
        astronaut_id=aid,
        astronaut_name=name,
        age=age,
        sex=sex,
        pulse=pulse_v,
        body_temperature=temp_v,
        spo2=spo2_v,
        rf_pred=labels_map.get(rf_pred, str(rf_pred)),
        svm_pred=labels_map.get(svm_pred, str(svm_pred)),
        rf_conf=round(rf_conf, 3),
        svm_conf=round(svm_conf, 3),
        svm_decision=round(svm_decision, 3),
        rf_metrics=rf_metrics,
        svm_metrics=svm_metrics,
        rf_cv=round(rf_cv, 3) if rf_cv is not None else None,
        svm_cv=round(svm_cv, 3) if svm_cv is not None else None,
        agreement=agreement,
        risk=risk,
        next_step=next_step,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
