from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load artifacts
with open("model/fire_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
ord_enc = artifacts["ordinal_encoder"]
label_enc = artifacts["label_encoder"]
freq_maps = artifacts["freq_maps"]
freq_cols = artifacts["freq_cols"]          # ["state", "stat_cause_descr", "wstation_usaf"]
ord_cols = artifacts["ord_cols"]           # ["disc_pre_month", "Vegetation"]
num_cols_all = artifacts["num_cols_all"]   # numeric columns

severity_map = {
    "B": "Low",
    "C": "Moderate",
    "D": "High",
    "E": "Very High",
    "F": "Severe",
    "G": "Extreme"
}

risk_profiles = {
    "Low": {
        "classes": ["B"],   # Low
        "band": "Low",
        "color": "green",
        "message": "Low Risk: Monitor conditions."
    },
    "Medium": {
        # Moderate, High, Very High
        "classes": ["C", "D"],
        "band": "Medium",
        "color": "yellow",
        "message": "Medium Risk: Stay alert and prepare for changes."
    },
    "High": {
        # Severe, Extreme
        "classes": ["E","F", "G"],
        "band": "High",
        "color": "red",
        "message": "High Risk: Immediate evacuation recommended."
    }
}

def map_class_to_risk(fire_class):
    for profile in risk_profiles.values():
        if fire_class in profile["classes"]:
            return profile["band"], profile["color"], profile["message"]
    return "Unknown", "grey", "Risk level unknown: consult authorities."

@app.route("/", methods=["GET"])
def index():
    # Render only input form
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # ---- categorical (frequency-encoded) ----
    stat_cause_descr = request.form.get("stat_cause_descr")
    state = request.form.get("state")
    wstation_usaf = request.form.get("wstation_usaf")

    # ---- categorical (ordinal-encoded) ----
    disc_pre_month = request.form.get("disc_pre_month")
    Vegetation = request.form.get("Vegetation")

    # ---- numeric ----
    latitude = float(request.form.get("latitude"))
    longitude = float(request.form.get("longitude"))
    disc_pre_year = float(request.form.get("disc_pre_year"))
    dstation_m = float(request.form.get("dstation_m"))
    fire_mag = float(request.form.get("fire_mag"))
    Temp_pre_7 = float(request.form.get("Temp_pre_7"))
    Wind_pre_7 = float(request.form.get("Wind_pre_7"))
    Hum_pre_7 = float(request.form.get("Hum_pre_7"))
    Prec_pre_7 = float(request.form.get("Prec_pre_7"))
    remoteness = float(request.form.get("remoteness"))

    # ---- frequency encoding using saved maps (fallback 0 if unseen) ----
    state_freq = freq_maps["state"].get(state, 0)
    cause_freq = freq_maps["stat_cause_descr"].get(stat_cause_descr, 0)
    wstation_freq = freq_maps["wstation_usaf"].get(wstation_usaf, 0)
    freq_features = [state_freq, cause_freq, wstation_freq]

    # ---- ordinal encoding ----
    ord_input = [[disc_pre_month, Vegetation]]  
    ord_encoded = ord_enc.transform(ord_input)[0].tolist()

    # ---- numeric scaling ----
    num_features = [
        latitude, longitude, disc_pre_year, dstation_m,
        fire_mag, Temp_pre_7, Wind_pre_7, Hum_pre_7,
        Prec_pre_7, remoteness
    ]
    num_scaled = scaler.transform([num_features])[0].tolist()

    # ---- final feature vector ----
    final_features = np.array([freq_features + ord_encoded + num_scaled])

    # ---- prediction ----
    y_pred_encoded = model.predict(final_features)[0]
    fire_size_class = label_enc.inverse_transform([y_pred_encoded])[0]
    severity = severity_map.get(fire_size_class, "Unknown")
    risk_band, risk_color, risk_message = map_class_to_risk(fire_size_class)

    
    return render_template(
        "result.html",
        fire_class=fire_size_class,
        severity=severity,
        risk_band=risk_band,
        risk_color=risk_color,
        risk_message=risk_message,
        input_data={
            "stat_cause_descr": stat_cause_descr,
            "state": state,
            "wstation_usaf": wstation_usaf,
            "disc_pre_month": disc_pre_month,
            "Vegetation": Vegetation,
            "latitude": latitude,
            "longitude": longitude,
            "disc_pre_year": disc_pre_year,
            "dstation_m": dstation_m,
            "fire_mag": fire_mag,
            "Temp_pre_7": Temp_pre_7,
            "Wind_pre_7": Wind_pre_7,
            "Hum_pre_7": Hum_pre_7,
            "Prec_pre_7": Prec_pre_7,
            "remoteness": remoteness
        }
    )

if __name__ == "__main__":
    app.run(debug=True)