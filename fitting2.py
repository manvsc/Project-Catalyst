import pandas as pd
import numpy as np
from scipy.stats import gamma


def main():
    path = "data2.csv"

    # Read CSV; handle spaces after commas and boolean parsing for 'voted'
    df = pd.read_csv(
        path,
        skipinitialspace=True,
        true_values=["true", "True", "1"],
        false_values=["false", "False", "0"],
    )

    # Normalize expected column names in case of spacing/casing variations
    cols = {c.strip().lower(): c for c in df.columns}
    addr_col = cols.get("voter address")
    power_col = cols.get("voting power")
    voted_col = cols.get("voted")

    if power_col is None or voted_col is None:
        raise ValueError(
            f"Expected columns 'voting power' and 'voted' not found. Got: {list(df.columns)}"
        )

    # Filter to only rows where voted == True
    voted_mask = df[voted_col]
    # If dtype isn't boolean, coerce via string match
    if voted_mask.dtype != bool:
        voted_mask = df[voted_col].astype(str).str.strip().str.lower().eq("true")

    df_true = df.loc[voted_mask]

    # Convert staking power to numeric, drop NaN and non-positive
    stakes = pd.to_numeric(df_true[power_col], errors="coerce").dropna().astype(float).values
    stakes = stakes[stakes > 0]

    if stakes.size == 0:
        print("No valid 'true' stakes found after filtering.")
        return

    # Basic statistics
    count_v = int(stakes.size)
    min_v = float(np.min(stakes))
    max_v = float(np.max(stakes))
    median_v = float(np.median(stakes))
    mean_v = float(np.mean(stakes))

    print("Statistics for TRUE stakes:")
    print(f"count = {count_v}")
    print(f"min = {min_v:.6f}")
    print(f"max = {max_v:.6f}")
    print(f"median = {median_v:.6f}")
    print(f"mean = {mean_v:.6f}")

    # Fit Gamma distribution with positive support by fixing loc=0
    alpha_hat, loc_hat, theta_hat = gamma.fit(stakes, floc=0)

    print("\nGamma MLE fit (loc fixed to 0):")
    print(f"alpha = {alpha_hat:.6f}")
    print(f"theta = {theta_hat:.6f}")


if __name__ == "__main__":
    main()
