# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import io
import json
import torch
import torch.nn as nn
from scipy.stats import entropy


print("Loading libraries...")

app = Flask(__name__)
# Add CORS configuration
CORS(app) 
print("Flask app initialized")

# --- Required base features ---
base_features = [
    'insol', 'insol_err1', 'insol_err2',
    'period', 'period_err1', 'period_err2',
    'prad', 'prad_err1', 'prad_err2',
    'steff', 'steff_err1', 'steff_err2',
    'srad', 'srad_err1', 'srad_err2'
]
print("Base features defined")

# --- Engineered features to keep ---
engineered_keep = [
    'prad_err1_div_insol', 'prad_x_insol', 'prad_div_period_err2',
    'prad_err1_div_period_err2', 'prad_div_steff_err1', 'prad_div_insol',
    'prad_div_period', 'period_div_prad_err1', 'prad_err1_x_insol',
    'prad_div_prad_err1', 'prad_x_steff_err1', 'prad_err1_div_steff_err1',
    'prad', 'steff_err1_div_period_err2', 'steff_err1_x_period_err2',
    'prad_err1_x_steff_err1', 'steff_err1_div_insol', 'steff_err1_x_insol',
    'prad_x_prad_err1', 'period', 'prad_err1', 'prad_x_period_err2',
    'period_err2_div_insol'
]
print("Engineered features defined")

# --- Load models ---
try:
    print("Loading XGBoost model...")
    xgb_model = joblib.load("backend/models/xgb_base_model_gridsearch.joblib")
    print("XGBoost model loaded successfully")

    print("Loading RandomForest model...")
    rfr_model = joblib.load("backend/models/rfr_base_model_full.joblib")
    print("RandomForest model loaded successfully")

    print("Loading CatBoost model...")
    catb_model = joblib.load("backend/models/catboost_base_model_full.joblib")
    print("CatBoost model loaded successfully")
except Exception as e:
    print("Error loading models:", e)

def preprocess(df: pd.DataFrame):
    """Preprocess base features, scale, and generate engineered features."""
    try:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        print("Standardization complete")

        X_aug = df_scaled.copy()
        df_list = [X_aug]
        for f1, f2 in combinations(df_scaled.columns, 2):
            new_cols = pd.DataFrame({
                f"{f1}_x_{f2}": X_aug[f1] * X_aug[f2],
                f"{f1}_div_{f2}": X_aug[f1] / X_aug[f2].replace(0, np.nan)
            })
            df_list.append(new_cols)

        X_aug = pd.concat(df_list, axis=1)
        X_aug.fillna(0, inplace=True)

        # Ensure all engineered_keep features exist (fill missing with 0)
        for col in engineered_keep:
            if col not in X_aug.columns:
                X_aug[col] = 0

        X_final = X_aug[engineered_keep]
        return X_final
    except Exception as e:
        print("Error during preprocessing:", e)
        raise

class MLP(nn.Module):
    def __init__(self, n_features, n_targets=3, hidden_units=[256, 128, 64], dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        input_dim = n_features
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, n_targets))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def add_meta_features(df, xgb_cols, other_cols, prefix, eps=1e-9):
    df_new = df.copy()
    # Multiplication
    for xgb_col in [c for c in xgb_cols if '_xgb' in c]:
        for other_col in [c for c in other_cols if '_rfr' in c or '_cat' in c]:
            df_new[f"{xgb_col}_mul_{other_col}"] = df_new[xgb_col] * df_new[other_col]
    # Division
    for xgb_col in [c for c in xgb_cols if '_xgb' in c]:
        for other_col in [c for c in other_cols if '_rfr' in c or '_cat' in c]:
            df_new[f"{xgb_col}_div_{other_col}"] = df_new[xgb_col] / (df_new[other_col] + eps)
    # Entropy
    base_cols = xgb_cols + [c for c in other_cols if '_rfr' in c or '_cat' in c]
    df_new[f"{prefix}_entropy"] = df_new[base_cols].apply(lambda row: entropy(row.values), axis=1)
    # Difference
    for cls in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
        xgb_c = f"class_{cls}_xgb"
        # Safely get the other class column name
        other_c_list = [c for c in other_cols if f"class_{cls}" in c]
        if other_c_list:
            other_c = other_c_list[0]
            df_new[f"{xgb_c}_diff_{other_c}"] = df_new[xgb_c] - df_new[other_c]
        # NOTE: If a class proba column is missing, the diff feature will be missing,
        # but the MLP input selection handles missing features by indexing only existing ones.
    return df_new

import zipfile
import io
import json
import pandas as pd
import numpy as np
import torch
from flask import request, jsonify, send_file
from sklearn.preprocessing import StandardScaler

@app.route("/predict", methods=["POST"])
def predict():
    print("=== /predict called ===")
    
    stack = request.form.get("stack")
    mapping_json = request.form.get("mapping")
    file = request.files.get("csv")
    print(f"Stack: {stack}, File: {file.filename if file else None}")

    if not file:
        return jsonify({"error": "No CSV file uploaded"}), 400
    if not mapping_json:
        return jsonify({"error": "No column mapping received"}), 400

    try:
        df = pd.read_csv(file)
        print(f"CSV loaded successfully with columns: {list(df.columns)}")
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {e}"}), 400

    try:
        mapping = json.loads(mapping_json)
        print("Mapping parsed successfully:", mapping)
    except Exception as e:
        return jsonify({"error": f"Invalid mapping JSON: {e}"}), 400

    missing_in_csv = [col for col in mapping.keys() if col not in df.columns]
    if missing_in_csv:
        return jsonify({"error": f"Mapped columns not found in CSV: {missing_in_csv}"}), 400

    df_renamed = df.rename(columns=mapping)
    missing_base = [col for col in base_features if col not in df_renamed.columns]
    if missing_base:
        return jsonify({"error": f"Missing required columns after mapping: {missing_base}"}), 400

    df_final = df_renamed[base_features]

    try:
        X = preprocess(df_final)
        print(f"DEBUG: Feature shape after preprocessing: {X.shape}")
    except Exception as e:
        return jsonify({"error": f"Error during preprocessing: {e}"}), 500

    model_files = {}
    base_models = {"xgb": xgb_model, "rfr": rfr_model, "catb": catb_model}
    meta_dfs = {}
    
    # --- Start of Robust ID/Key Handling ---
    
    # Determine the columns not used as model features (these are potential ID columns)
    feature_cols_mapped = set(mapping.keys())
    all_cols = set(df.columns)
    id_cols = sorted(list(all_cols - feature_cols_mapped))
    
    # Fallback if no non-feature columns exist
    if not id_cols:
        df['row_id'] = pd.Series(range(len(df)))
        id_cols = ['row_id']
    
    # Create the key DataFrame containing only the ID columns
    df_keys = df[id_cols].copy()
    
    print(f"DEBUG: Using ID columns for merging/output: {id_cols}")
    
    # --- End of Robust ID/Key Handling ---

    try:
        # --- Base model predictions ---
        for name, model in base_models.items():
            # Start with a copy of the keys for safe merging
            df_copy = df_keys.copy() 
            
            # Perform predictions
            if name == "xgb":
                df_copy["y_oof_pred"] = model.predict(X).astype(int)
            else:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    for i, cls in enumerate(["CANDIDATE","CONFIRMED","FALSE POSITIVE"]):
                        df_copy[f"class_{cls}"] = proba[:,i]
                else:
                    df_copy["y_oof_pred"] = model.predict(X).astype(int)

            df_copy["model_used"] = name
            
            # Merge with original data for saving (using the ID columns)
            # This ensures the output CSV contains all original columns plus predictions
            df_output = pd.merge(df, df_copy, on=id_cols, how='left')

            buffer = io.StringIO()
            df_output.to_csv(buffer, index=False)
            buffer.seek(0)
            model_files[f"{name}_predictions.csv"] = io.BytesIO(buffer.getvalue().encode())

            # For meta-model input, we only need ID columns + prediction probabilities/classes
            meta_dfs[name] = df_copy

        # --- Robust meta merge function (simplified since ID columns are guaranteed) ---
        def safe_merge_meta(base_df, other_df, on_cols):
            # Only merge prediction columns
            existing_pred_cols = [c for c in other_df.columns if c not in on_cols]
            return pd.merge(base_df, other_df[on_cols + existing_pred_cols], on=on_cols, how='left')
        
        # XGB one-hot
        # --- XGB one-hot (robust for single-row or missing y_oof_pred) ---
        xgb_meta = meta_dfs['xgb']
        y_oof = xgb_meta.get('y_oof_pred', pd.Series())

        if len(y_oof) == 0:
            return jsonify({"error": "No data returned after base model predictions."}), 500

        if isinstance(y_oof.iloc[0], (int, float, np.integer, np.float64)):
            y_oof = pd.Series(y_oof.values, index=xgb_meta.index)

        class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
        xgb_probs = pd.get_dummies(y_oof).astype(float)
        
        xgb_final_cols = [f"class_{cls}_xgb" for cls in class_names]
        
        for i, col in enumerate(class_names):
            if i in xgb_probs.columns:
                xgb_probs.rename(columns={i: xgb_final_cols[i]}, inplace=True)
            elif xgb_final_cols[i] not in xgb_probs.columns:
                xgb_probs[xgb_final_cols[i]] = 0.0

        xgb_probs = xgb_probs[xgb_final_cols]
        xgb_probs[id_cols] = xgb_meta[id_cols]
        
        # Clean RFR
        rfr_meta = meta_dfs['rfr']
        # The columns we want from RFR for the meta-model
        rfr_pred_cols = ['class_CANDIDATE','class_CONFIRMED','class_FALSE POSITIVE']
        rfr_cols_to_select = [c for c in id_cols if c in rfr_meta.columns] + [c for c in rfr_pred_cols if c in rfr_meta.columns]
        rfr_clean = rfr_meta[rfr_cols_to_select].copy()
        rfr_clean.rename(columns={
            'class_CANDIDATE':'class_CANDIDATE_rfr',
            'class_CONFIRMED':'class_CONFIRMED_rfr',
            'class_FALSE POSITIVE':'class_FALSE POSITIVE_rfr'
        }, inplace=True)

        # Clean CatB
        cat_meta = meta_dfs['catb']
        # The columns we want from CatB for the meta-model
        cat_pred_cols = ['class_CANDIDATE','class_CONFIRMED','class_FALSE POSITIVE']
        cat_cols_to_select = [c for c in id_cols if c in cat_meta.columns] + [c for c in cat_pred_cols if c in cat_meta.columns]
        cat_clean = cat_meta[cat_cols_to_select].copy()
        cat_clean.rename(columns={
            'class_CANDIDATE':'class_CANDIDATE_cat',
            'class_CONFIRMED':'class_CONFIRMED_cat',
            'class_FALSE POSITIVE':'class_FALSE POSITIVE_cat'
        }, inplace=True)

        # Merge robustly on ID columns
        meta_df = safe_merge_meta(xgb_probs, rfr_clean, id_cols)
        meta_df = safe_merge_meta(meta_df, cat_clean, id_cols)
        meta_df.fillna(0.0, inplace=True)
        
        # --- Meta features ---
        meta1_df = add_meta_features(meta_df, xgb_cols=[c for c in meta_df if '_xgb' in c],
                                     other_cols=[c for c in meta_df if '_rfr' in c], prefix="meta1")
        meta2_df = add_meta_features(meta_df, xgb_cols=[c for c in meta_df if '_xgb' in c],
                                     other_cols=[c for c in meta_df if '_cat' in c], prefix="meta2")

        # --- Stack selection and prediction ---
        
        if stack == "stack1":
            selected_features = [
                'class_CONFIRMED_xgb_div_class_CONFIRMED_rfr',
                'class_CONFIRMED_xgb_div_class_FALSE POSITIVE_rfr',
                'class_CONFIRMED_xgb_div_class_CANDIDATE_rfr',
                'class_CONFIRMED_xgb',
                'class_CONFIRMED_xgb_diff_class_CONFIRMED_rfr',
                'class_FALSE POSITIVE_xgb_div_class_CONFIRMED_rfr',
                'class_FALSE POSITIVE_xgb_diff_class_FALSE POSITIVE_rfr',
                'class_FALSE POSITIVE_xgb_div_class_FALSE POSITIVE_rfr',
                'class_FALSE POSITIVE_xgb',
                'class_FALSE POSITIVE_xgb_div_class_CANDIDATE_rfr',
                'class_CONFIRMED_xgb_mul_class_CANDIDATE_rfr',
                'class_CONFIRMED_xgb_mul_class_CONFIRMED_rfr',
                'class_FALSE POSITIVE_xgb_mul_class_FALSE POSITIVE_rfr',
                'class_CONFIRMED_xgb_mul_class_FALSE POSITIVE_rfr',
                'class_FALSE POSITIVE_xgb_mul_class_CANDIDATE_rfr'
            ]
            mlp_input_df = meta1_df
            model_path = "backend/models/meta1_mlp_best_model.pth"
        else:
            selected_features = [
                'class_FALSE POSITIVE_xgb_diff_class_FALSE POSITIVE_cat',
                'class_CONFIRMED_xgb_mul_class_CONFIRMED_cat',
                'class_CONFIRMED_xgb_diff_class_CONFIRMED_cat',
                'class_CONFIRMED_xgb_div_class_CONFIRMED_cat',
                'class_CONFIRMED_xgb_mul_class_CANDIDATE_cat',
                'class_CONFIRMED_xgb_mul_class_FALSE POSITIVE_cat',
                'class_CONFIRMED_xgb_div_class_FALSE POSITIVE_cat',
                'class_CONFIRMED_xgb_div_class_CANDIDATE_cat',
                'class_CONFIRMED_xgb',
                'class_FALSE POSITIVE_xgb_div_class_CONFIRMED_cat',
                'class_FALSE POSITIVE_xgb_mul_class_CANDIDATE_cat',
                'class_FALSE POSITIVE_xgb_mul_class_CONFIRMED_cat',
                'class_FALSE POSITIVE_xgb_div_class_FALSE POSITIVE_cat',
                'class_FALSE POSITIVE_xgb',
                'class_FALSE POSITIVE_xgb_div_class_CANDIDATE_cat',
                'class_FALSE POSITIVE_xgb_mul_class_FALSE POSITIVE_cat'
            ]
            mlp_input_df = meta2_df
            model_path = "backend/models/meta2_mlp_best_model.pth"
            
        final_mlp_cols = [col for col in selected_features if col in mlp_input_df.columns]
        if not final_mlp_cols:
            return jsonify({"error": "MLP input features could not be generated from base model predictions."}), 500
        
        X_mlp_input = mlp_input_df[final_mlp_cols]


        # --- Standardize and predict ---
        scaler_mlp = StandardScaler()
        X_mlp_scaled = scaler_mlp.fit_transform(X_mlp_input.astype(np.float32))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = MLP(n_features=X_mlp_scaled.shape[1])

        state_or_model = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(state_or_model, nn.Module):
            mlp_model = state_or_model.to(device)
        else:
            mlp_model.load_state_dict(state_or_model)
            mlp_model.to(device)

        mlp_model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X_mlp_scaled, dtype=torch.float32).to(device)
            logits = mlp_model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(probs, axis=1)
            certainty = np.max(probs, axis=1)

        # --- Prepare meta CSV ---
        result_df = meta_df[id_cols].copy() 
        result_df['y_pred'] = y_pred
        result_df['certainty'] = certainty
        result_df['prob_class_0'] = probs[:,0]
        result_df['prob_class_1'] = probs[:,1]
        result_df['prob_class_2'] = probs[:,2]

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zipf:
            for fname, fbuf in model_files.items():
                zipf.writestr(fname, fbuf.getvalue())
            meta_csv_buf = io.StringIO()
            result_df.to_csv(meta_csv_buf, index=False)
            zipf.writestr(f"meta_{stack}_predictions.csv", meta_csv_buf.getvalue())
        buffer.seek(0)

        return send_file(buffer,
                         mimetype='application/zip',
                         as_attachment=True,
                         download_name=f"all_predictions_{stack}.zip")

    except Exception as e:
        print(f"ERROR during meta prediction: {e}")
        return jsonify({"error": f"Error during meta prediction: {e}"}), 500


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=5000)