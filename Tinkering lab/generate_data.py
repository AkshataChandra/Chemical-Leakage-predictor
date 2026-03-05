# generate_data.py
import pandas as pd, numpy as np
np.random.seed(42)
n = 4000

data = {
    "initialvolume": np.random.uniform(600, 2000, n),
    "transferredvolume": np.random.uniform(500, 2000, n),
    "ambienttemp": np.random.uniform(-10, 120, n),
    "chemicalevappoint": np.random.uniform(-20, 200, n),
    "humidity": np.random.uniform(5, 95, n),
    "pressure": np.random.uniform(0.7, 1.5, n),
    "chemical_mw": np.random.uniform(30, 250, n)
}
df = pd.DataFrame(data)

# derived features
df["volume_diff"] = df["initialvolume"] - df["transferredvolume"]
df["percent_loss"] = (df["volume_diff"] / (df["initialvolume"] + 1e-9)) * 100
df["temp_diff"] = df["ambienttemp"] - df["chemicalevappoint"]
df["evaporation_ratio"] = df["ambienttemp"] / (df["chemicalevappoint"] + 1e-9)
df["safety_margin"] = (df["chemicalevappoint"] - df["ambienttemp"]).clip(lower=0)

# vapor-pressure proxy
T = df["ambienttemp"] + 273.15
bp = df["chemicalevappoint"] + 273.15
mw = df["chemical_mw"]
vp_proxy = np.exp(-mw / 500.0) * np.exp((T - bp) / (50.0 + 0.01*np.abs(bp)))
df["vapor_pressure_proxy"] = vp_proxy

# risk score + noise
raw_score = (0.06*df["temp_diff"] + 0.04*df["percent_loss"]
             -0.02*df["humidity"] -0.6*df["pressure"]
             +0.45*df["vapor_pressure_proxy"]
             +0.02*(df["volume_diff"]/df["initialvolume"])) + np.random.normal(0,0.2,len(df))
leak_prob = 1/(1+np.exp(-raw_score))
df["leakprob"] = leak_prob

# balanced leak occurrence
threshold = np.percentile(leak_prob, 60)     # ≈ 40% leaks
df["leakoccurred"] = (leak_prob > threshold).astype(int)

df.to_csv("chemicaldata.csv", index=False)
print("✅ Dataset:", df.shape, " | Leak ratio:", df['leakoccurred'].mean().round(2))


