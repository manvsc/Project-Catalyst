import pandas as pd
import numpy as np
from scipy.stats import gamma

# --- Load and clean data ---

path = "data.csv"
df = pd.read_csv(path)

# Drop malformed header row and rename columns
df = df.iloc[1:].copy()
df.columns = ["stake_address", "voting_power"]

# Convert to numeric and drop invalid entries
stakes = pd.to_numeric(df["voting_power"], errors="coerce").dropna().values

# --- Gamma MLE fit (support on R_{>0}) ---

# We fix loc = 0 to enforce positive support
alpha_hat, loc_hat, theta_hat = gamma.fit(stakes, floc=0)

print(f"alpha = {alpha_hat:.6f}")
print(f"theta = {theta_hat:.6f}")
