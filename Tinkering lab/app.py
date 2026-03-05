# app.py — Dynamic Risk Dashboard (final stable version)
from flask import Flask, request, render_template
import pandas as pd, numpy as np, joblib
import plotly.graph_objects as go

app = Flask(__name__)

# -----------------------------------------------------
# Load model and preprocessing artifacts
# -----------------------------------------------------
model  = joblib.load("leakagemodel_calibrated.pkl")
scaler = joblib.load("scaler.pkl")
features = [str(f) for f in joblib.load("selected_features.pkl")]
calib_df = pd.read_csv("calibration_data.csv")

# -----------------------------------------------------
# Feature preparation (works for both dict & DataFrame)
# -----------------------------------------------------
def prepare_features(d):
    # Accept both dict (single input) and DataFrame (bulk grid)
    if isinstance(d, dict):
        df = pd.DataFrame([d])
    else:
        df = d.copy()

    # Derived features
    df["volume_diff"] = df["initialvolume"] - df["transferredvolume"]
    df["percent_loss"] = (df["volume_diff"] / (df["initialvolume"] + 1e-9)) * 100
    df["temp_diff"] = df["ambienttemp"] - df["chemicalevappoint"]
    df["evaporation_ratio"] = df["ambienttemp"] / (df["chemicalevappoint"] + 1e-9)
    df["safety_margin"] = (df["chemicalevappoint"] - df["ambienttemp"]).clip(lower=0)

    # Vapor pressure proxy
    T = df["ambienttemp"] + 273.15
    bp = df["chemicalevappoint"] + 273.15
    mw = df.get("chemical_mw", pd.Series([100] * len(df)))
    df["vapor_pressure_proxy"] = np.exp(-mw / 500.0) * np.exp((T - bp) / (50.0 + 0.01 * np.abs(bp)))

    if "chemical_mw" not in df.columns:
        df["chemical_mw"] = 100.0

    X = df[features]
    X_scaled = scaler.transform(X)
    return X_scaled, df

# -----------------------------------------------------
# Safety recommendations
# -----------------------------------------------------
def safety_recommendations(prob, df):
    recs = []
    if prob >= 0.85:
        recs.append("🚨 Immediate action: stop transfer and contain leak.")
    elif prob >= 0.6:
        recs.append("⚠ High risk: inspect seals and pressure control.")
    elif prob >= 0.4:
        recs.append("🟡 Moderate risk: monitor and cool system.")
    else:
        recs.append("🟢 Low risk: normal monitoring.")

    if df["ambienttemp"].iloc[0] > df["chemicalevappoint"].iloc[0]:
        recs.append("Ambient temperature above evaporation point — cool system.")
    if df["pressure"].iloc[0] < 0.9:
        recs.append("Low pressure increases evaporation risk.")
    if df["percent_loss"].iloc[0] > 5:
        recs.append("Percent loss >5% — inspect transfer lines.")
    return recs

# -----------------------------------------------------
# Dynamic visualization functions
# -----------------------------------------------------
def plot_heatmap(center_temp, center_humidity, df_params):
    temps = np.linspace(center_temp - 40, center_temp + 40, 60)
    hums  = np.linspace(max(5, center_humidity - 40), min(95, center_humidity + 40), 60)
    grid = []
    for T in temps:
        for H in hums:
            g = df_params.copy()
            g["ambienttemp"] = T
            g["humidity"] = H
            grid.append(g)
    grid_df = pd.DataFrame(grid)
    Xs, _ = prepare_features(grid_df)
    p = model.predict_proba(Xs)[:, 1]
    Z = p.reshape(len(temps), len(hums))
    fig = go.Figure(go.Heatmap(z=Z, x=hums, y=temps, colorscale="RdYlGn_r", colorbar=dict(title="Leak prob")))
    fig.update_layout(title="Dynamic Risk Heatmap", xaxis_title="Humidity (%)", yaxis_title="Temperature (°C)")
    return fig.to_html(full_html=False)

def plot_3d(center_temp, center_pressure, df_params):
    temps = np.linspace(center_temp - 40, center_temp + 40, 40)
    press = np.linspace(max(0.7, center_pressure - 0.4), min(1.5, center_pressure + 0.4), 40)
    grid = []
    for T in temps:
        for P in press:
            g = df_params.copy()
            g["ambienttemp"] = T
            g["pressure"] = P
            grid.append(g)
    grid_df = pd.DataFrame(grid)
    Xs, _ = prepare_features(grid_df)
    probs = model.predict_proba(Xs)[:, 1].reshape(len(temps), len(press))
    fig = go.Figure(go.Surface(x=press, y=temps, z=probs, colorscale="RdYlGn_r"))
    fig.update_layout(
        title="3D Dynamic Risk Surface",
        scene=dict(xaxis_title="Pressure (atm)", yaxis_title="Temperature (°C)", zaxis_title="Leak Probability")
    )
    return fig.to_html(full_html=False)

def plot_calibration():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=calib_df["prob_pred"], y=calib_df["prob_true"], mode="lines+markers", name="Calibrated"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Perfect"))
    fig.update_layout(title="Model Calibration Curve", xaxis_title="Predicted", yaxis_title="Observed")
    return fig.to_html(full_html=False)

# -----------------------------------------------------
# Flask route
# -----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    recs = []
    form_values = {
        "initialvolume": 1000,
        "transferredvolume": 800,
        "ambienttemp": 25,
        "chemicalevappoint": 78,
        "humidity": 50,
        "pressure": 1.0,
        "chemical_mw": 100
    }

    if request.method == "POST":
        try:
            for key in form_values:
                form_values[key] = float(request.form.get(key, form_values[key]))
            X_scaled, df_row = prepare_features(form_values)
            prob = float(model.predict_proba(X_scaled)[0][1])
            result = {"prob": prob}
            recs = safety_recommendations(prob, df_row)
        except Exception as e:
            result = {"error": str(e)}

    # Generate updated plots dynamically based on inputs
    heatmap_html = plot_heatmap(form_values["ambienttemp"], form_values["humidity"], form_values)
    surface_html = plot_3d(form_values["ambienttemp"], form_values["pressure"], form_values)
    calibration_html = plot_calibration()

    return render_template("index.html",
                           result=result,
                           recs=recs,
                           form_values=form_values,
                           heatmap_html=heatmap_html,
                           surface_html=surface_html,
                           calibration_html=calibration_html)

# -----------------------------------------------------
# Run app
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)



