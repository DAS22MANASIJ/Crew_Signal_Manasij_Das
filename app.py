from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import re
import csv
import threading
import math
import random
from collections import defaultdict, Counter

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


def _softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _mean(vs):
    return sum(vs) / len(vs) if vs else 0.0


def _euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class SimpleCentroidClassifier:
    def __init__(self):
        self.centroids = {}  # label -> [feat means]
        self.labels = []

    def fit(self, X, y):
        groups = defaultdict(list)
        for xi, yi in zip(X, y):
            groups[yi].append(xi)
        self.centroids = {lbl: [ _mean(col) for col in zip(*rows) ] for lbl, rows in groups.items()}
        self.labels = sorted(self.centroids.keys())

    def predict(self, X):
        preds = []
        for xi in X:
            # pick nearest centroid
            dists = [(lbl, _euclidean(xi, self.centroids[lbl])) for lbl in self.labels]
            dists.sort(key=lambda t: t[1])
            preds.append(dists[0][0])
        return preds

    def predict_proba(self, X):
        probs = []
        for xi in X:
            dists = [ -_euclidean(xi, self.centroids[lbl]) for lbl in self.labels ]
            probs.append(_softmax(dists))
        return probs


class SimpleLinearScoreClassifier(SimpleCentroidClassifier):
    # reuse centroids but compute a score vector as negative distance (like a linear-ish decision)
    def decision_function(self, X):
        out = []
        for xi in X:
            scores = [ -_euclidean(xi, self.centroids[lbl]) for lbl in self.labels ]
            out.append(scores)
        return out


def _confusion_matrix(y_true, y_pred, labels):
    n = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    mat = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        mat[idx[t]][idx[p]] += 1
    return mat


def _precision_recall_f1(y_true, y_pred, labels):
    n = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    tp = [0]*n
    fp = [0]*n
    fn = [0]*n
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[idx[t]] += 1
        else:
            fp[idx[p]] += 1
            fn[idx[t]] += 1
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for i in range(n):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i]+fp[i])>0 else 0.0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i]+fn[i])>0 else 0.0
        precision += p
        recall += r
        f1 += (2*p*r/(p+r) if (p+r)>0 else 0.0)
    # macro average
    return precision/n if n>0 else 0.0, recall/n if n>0 else 0.0, f1/n if n>0 else 0.0



def load_and_train():
    global rf_model, svm_model, rf_metrics, svm_metrics, rf_cv, svm_cv, labels_map, training_complete
    path = os.path.join(os.path.dirname(__file__), 'Health data.csv')
    if not os.path.exists(path):
        # nothing to train on
        training_complete = True
        return

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cleaned = []
    for line in lines:
        cleaned.append(re.sub(r"^\s*\d+\|\s*", "", line))
    reader = csv.reader(cleaned)
    rows = list(reader)
    if not rows:
        training_complete = True
        return
    header = [c.strip().lower().replace(' ', '_') for c in rows[0]]
    data_rows = rows[1:]

    # find columns
    col_map = {c.replace('_',''): i for i,c in enumerate(header)}
    pulse_col = col_map.get('pulse')
    temp_col = col_map.get('bodytemperature') or col_map.get('bodytemp') or col_map.get('body_temperature')
    spo2_col = col_map.get('spo2') or col_map.get('spo_2')
    status_col = col_map.get('status')
    if pulse_col is None or temp_col is None or spo2_col is None or status_col is None:
        training_complete = True
        return

    X = []
    y = []
    for r in data_rows:
        try:
            p = float(r[pulse_col])
            t = float(r[temp_col])
            s = float(r[spo2_col])
            st = int(float(r[status_col]))
            X.append([p, t, s])
            y.append(st)
        except Exception:
            continue
    if not X:
        training_complete = True
        return

    # map labels and compute centroids
    unique_status = sorted(set(y))
    names = ['Normal', 'Moderate', 'Critical']
    labels_map = {status: names[idx] if idx < len(names) else f'Status_{idx}' for idx, status in enumerate(unique_status)}

    # split
    combined = list(zip(X, y))
    random.seed(42)
    random.shuffle(combined)
    split = max(1, int(0.8 * len(combined)))
    train = combined[:split]
    test = combined[split:]
    X_train = [a for a,b in train]
    y_train = [b for a,b in train]
    X_test = [a for a,b in test]
    y_test = [b for a,b in test]

    rf = SimpleCentroidClassifier()
    svm = SimpleLinearScoreClassifier()
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    rf_model = rf
    svm_model = svm

    # predictions
    y_rf = rf.predict(X_test) if X_test else []
    y_svm = svm.predict(X_test) if X_test else []

    # metrics
    acc_rf = sum(1 for a,b in zip(y_test, y_rf) if a==b)/len(y_test) if y_test else 0.0
    acc_svm = sum(1 for a,b in zip(y_test, y_svm) if a==b)/len(y_test) if y_test else 0.0
    prec_rf, rec_rf, f1_rf = _precision_recall_f1(y_test, y_rf, unique_status) if y_test else (0,0,0)
    prec_svm, rec_svm, f1_svm = _precision_recall_f1(y_test, y_svm, unique_status) if y_test else (0,0,0)
    cm_rf = _confusion_matrix(y_test, y_rf, unique_status) if y_test else []
    cm_svm = _confusion_matrix(y_test, y_svm, unique_status) if y_test else []

    rf_metrics.update({
        'accuracy': float(acc_rf),
        'precision': float(prec_rf),
        'recall': float(rec_rf),
        'f1': float(f1_rf),
        'confusion_matrix': cm_rf,
    })
    svm_metrics.update({
        'accuracy': float(acc_svm),
        'precision': float(prec_svm),
        'recall': float(rec_svm),
        'f1': float(f1_svm),
        'confusion_matrix': cm_svm,
    })

    # very simple CV approximation: accuracy on random splits
    def cv_score(model, Xall, yall, folds=3):
        n = len(Xall)
        if n < 2:
            return 0.0
        scores = []
        indices = list(range(n))
        for i in range(folds):
            random.shuffle(indices)
            k = max(1, n//folds)
            test_idx = set(indices[:k])
            train_idx = [j for j in indices if j not in test_idx]
            Xtr = [Xall[j] for j in train_idx]
            ytr = [yall[j] for j in train_idx]
            Xte = [Xall[j] for j in test_idx]
            yte = [yall[j] for j in test_idx]
            m = model.__class__()
            m.fit(Xtr, ytr)
            preds = m.predict(Xte) if Xte else []
            score = sum(1 for a,b in zip(yte, preds) if a==b)/len(yte) if yte else 0.0
            scores.append(score)
        return float(sum(scores)/len(scores)) if scores else 0.0

    rf_cv = cv_score(rf, X, y, folds=3)
    svm_cv = cv_score(svm, X, y, folds=3)

    globals()['rf_cv'] = rf_cv
    globals()['svm_cv'] = svm_cv
    globals()['labels_map'] = labels_map
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

    features = [[pulse_v, temp_v, spo2_v]]

    rf_pred = None
    svm_pred = None
    rf_proba = [[1.0]]
    svm_proba = [[1.0]]
    rf_conf = 0.0
    svm_conf = 0.0
    svm_decision = 0.0

    try:
        rf_pred_list = rf_model.predict(features)
        rf_pred = int(rf_pred_list[0])
        rf_proba = rf_model.predict_proba(features)
        rf_conf = float(max(rf_proba[0]))
    except Exception:
        rf_pred = None

    try:
        svm_pred_list = svm_model.predict(features)
        svm_pred = int(svm_pred_list[0])
        dec = svm_model.decision_function(features)
        # decision: difference between top two scores
        if dec and len(dec[0])>=2:
            sorted_scores = sorted(dec[0], reverse=True)
            svm_decision = float(sorted_scores[0] - sorted_scores[1])
        else:
            svm_decision = float(dec[0][0]) if dec else 0.0
        svm_conf = float(max(_softmax(dec[0]))) if dec else 0.0
    except Exception:
        svm_pred = None

    agreement = (rf_pred is not None and svm_pred is not None and rf_pred == svm_pred)

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
        rf_pred=labels_map.get(rf_pred, str(rf_pred)) if rf_pred is not None else 'Unknown',
        svm_pred=labels_map.get(svm_pred, str(svm_pred)) if svm_pred is not None else 'Unknown',
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
