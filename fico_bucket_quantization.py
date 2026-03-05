import pandas as pd
import numpy as np

# ============================================================
# Task 4: FICO Quantization (Bucket Optimization)
# Objective: find bucket boundaries that maximize log-likelihood
# ============================================================

def _bin_log_likelihood(n, k):
    """
    Binomial log-likelihood contribution for a bucket:
    n = number of observations in bucket
    k = number of defaults in bucket
    p = k/n
    """
    if n <= 0:
        return -1e18
    p = k / n
    # Avoid log(0) / log(1); if p=0 or 1 the term is 0 in the sample answer
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return k * np.log(p) + (n - k) * np.log(1.0 - p)

def build_fico_buckets_max_ll(
    df,
    n_buckets=10,
    fico_col="fico_score",
    default_col="default",
    fico_min=300,
    fico_max=850
):
    """
    Dynamic programming solution to choose bucket boundaries that maximize
    the sum of bucket-level binomial log-likelihoods.

    Returns:
      boundaries: sorted list of bucket upper-boundaries (inclusive)
      rating_map: dict mapping fico_score -> rating (1..n_buckets), where
                  rating 1 is best credit quality (highest FICO).
      bucket_table: dataframe with bucket stats (n, k, pd)
    """

    # Keep only relevant columns and drop missing
    d = df[[fico_col, default_col]].dropna().copy()
    d[fico_col] = d[fico_col].astype(int)
    d = d[(d[fico_col] >= fico_min) & (d[fico_col] <= fico_max)]

    # Count totals and defaults by FICO score
    scores = np.arange(fico_min, fico_max + 1)
    total = np.zeros(len(scores), dtype=int)
    defaults = np.zeros(len(scores), dtype=int)

    for s, grp in d.groupby(fico_col):
        idx = s - fico_min
        total[idx] = len(grp)
        defaults[idx] = int(grp[default_col].sum())

    # Prefix sums so bucket counts can be computed in O(1)
    total_cum = np.cumsum(total)
    default_cum = np.cumsum(defaults)

    def bucket_stats(i, j):
        """
        Bucket covering score indices (i+1..j) inclusive in prefix space.
        Here we use prefix sums:
          counts in (k..j] = cum[j] - cum[k]
        We'll represent a bucket as (k -> j), where k < j in prefix index.
        """
        n = total_cum[j] - total_cum[i]
        k = default_cum[j] - default_cum[i]
        return n, k

    # DP arrays:
    # dp[b][j] = best log-likelihood using b buckets to cover up to index j
    # back[b][j] = best split point i for dp[b][j]
    m = len(scores) - 1  # last prefix index for cum arrays
    dp = np.full((n_buckets + 1, m + 1), -1e18, dtype=float)
    back = np.full((n_buckets + 1, m + 1), -1, dtype=int)

    # Base: 0 buckets yields 0 likelihood at j=0 only
    dp[0, 0] = 0.0

    # Fill DP
    for b in range(1, n_buckets + 1):
        for j in range(1, m + 1):
            best_val = -1e18
            best_i = -1
            # try last split at i, forming bucket (i..j]
            for i in range(0, j):
                if dp[b - 1, i] <= -1e17:
                    continue
                n, k = bucket_stats(i, j)
                if n == 0:
                    continue
                val = dp[b - 1, i] + _bin_log_likelihood(n, k)
                if val > best_val:
                    best_val = val
                    best_i = i
            dp[b, j] = best_val
            back[b, j] = best_i

    # Recover boundaries by backtracking
    j = m
    boundaries_idx = []
    b = n_buckets
    while b > 0 and j > 0:
        i = back[b, j]
        if i < 0:
            break
        boundaries_idx.append(j)
        j = i
        b -= 1

    boundaries_idx = sorted(boundaries_idx)

    # Convert boundary indices to actual FICO scores (upper inclusive)
    boundaries = [fico_min + idx for idx in boundaries_idx]

    # Build bucket table and rating map
    bucket_rows = []
    rating_map = {}

    lower = fico_min
    for r, upper in enumerate(boundaries, start=1):
        # bucket = [lower..upper]
        mask = (d[fico_col] >= lower) & (d[fico_col] <= upper)
        n = int(mask.sum())
        k = int(d.loc[mask, default_col].sum())
        pd_bucket = (k / n) if n > 0 else 0.0

        bucket_rows.append({
            "rating_raw": r,               # 1..n_buckets in ascending FICO
            "fico_lower": lower,
            "fico_upper": upper,
            "n": n,
            "defaults": k,
            "pd": pd_bucket
        })

        # rating definition required by prompt:
        # "lower rating signifies a better credit score"
        # So highest FICO should map to rating 1 (best).
        # Our bucket construction is from low->high FICO, so we invert.
        for s in range(lower, upper + 1):
            rating_map[s] = None  # fill later

        lower = upper + 1

    bucket_table = pd.DataFrame(bucket_rows)

    # Invert ratings so that higher FICO gets smaller rating number
    # Example: if we have 10 buckets, the highest-FICO bucket should be rating 1.
    bucket_table["rating"] = (n_buckets + 1) - bucket_table["rating_raw"]
    bucket_table = bucket_table.drop(columns=["rating_raw"]).sort_values("rating").reset_index(drop=True)

    # Fill rating_map using inverted ratings
    for _, row in bucket_table.iterrows():
        r = int(row["rating"])
        for s in range(int(row["fico_lower"]), int(row["fico_upper"]) + 1):
            rating_map[s] = r

    return boundaries, rating_map, bucket_table


def build_fico_buckets_equal_width(n_buckets=10, fico_min=300, fico_max=850):
    """
    Simple baseline: equal-width buckets across the FICO range.
    Returns upper boundaries inclusive.
    """
    width = (fico_max - fico_min + 1) / n_buckets
    bounds = []
    for i in range(1, n_buckets + 1):
        upper = int(round(fico_min + i * width - 1))
        bounds.append(min(upper, fico_max))
    bounds[-1] = fico_max
    return sorted(list(dict.fromkeys(bounds)))


# =========================
# Example usage / testing
# =========================
if __name__ == "__main__":
    
    df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

    n_buckets = 10

    boundaries, rating_map, bucket_table = build_fico_buckets_max_ll(
        df,
        n_buckets=n_buckets,
        fico_col="fico_score",
        default_col="default",
        fico_min=300,
        fico_max=850
    )

    print("Optimized bucket upper boundaries (inclusive):")
    print(boundaries)

    print("\nBucket summary (rating 1 = best credit):")
    print(bucket_table)

    # Baseline boundaries (optional comparison)
    equal_bounds = build_fico_buckets_equal_width(n_buckets=n_buckets)
    print("\nEqual-width boundaries (baseline):")
    print(equal_bounds)

    # Example: map a FICO score to rating
    example_fico = 720
    print(f"\nExample mapping: FICO {example_fico} -> Rating {rating_map.get(example_fico)}")
