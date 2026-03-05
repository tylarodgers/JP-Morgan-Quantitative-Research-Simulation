import pandas as pd
import matplotlib.pyplot as plt
import math

# ============================================================
# PART 1
# Natural Gas Price Analysis and Estimation Model
# ============================================================

# Load historical natural gas price data
data = pd.read_csv("Nat_Gas.csv")

# Convert date column to datetime format
data["Dates"] = pd.to_datetime(data["Dates"])

# Ensure chronological ordering
data = data.sort_values("Dates").reset_index(drop=True)

# ------------------------------------------------------------
# Visualisation of historical price behaviour
# ------------------------------------------------------------

plt.figure()
plt.plot(data["Dates"], data["Prices"], marker="o")
plt.title("Natural Gas Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Price estimation function
# ------------------------------------------------------------

def estimate_price(input_date):

    """
    Estimates the price of natural gas for any requested date.

    If the date lies within the historical dataset, the price
    is estimated using linear interpolation between the
    nearest observed price points.

    If the date lies outside the available dataset, a simple
    linear extrapolation is performed based on the most recent
    price trend.
    """

    d = pd.to_datetime(input_date)

    dmin = data["Dates"].min()
    dmax = data["Dates"].max()

    # Case 1: date within dataset range
    if dmin <= d <= dmax:

        before = data[data["Dates"] <= d].iloc[-1]
        after = data[data["Dates"] >= d].iloc[0]

        if before["Dates"] == after["Dates"]:
            return float(before["Prices"])

        price = before["Prices"] + (
            (after["Prices"] - before["Prices"])
            * (d - before["Dates"]).days
            / (after["Dates"] - before["Dates"]).days
        )

        return float(price)

    # Case 2: extrapolate outside dataset
    n = min(12, len(data) - 1)

    if d > dmax:

        recent = data.tail(n + 1)

        days = (recent["Dates"].iloc[-1] - recent["Dates"].iloc[0]).days

        slope = (recent["Prices"].iloc[-1] - recent["Prices"].iloc[0]) / days

        return float(data["Prices"].iloc[-1] + slope * (d - dmax).days)

    if d < dmin:

        early = data.head(n + 1)

        days = (early["Dates"].iloc[-1] - early["Dates"].iloc[0]).days

        slope = (early["Prices"].iloc[-1] - early["Prices"].iloc[0]) / days

        return float(data["Prices"].iloc[0] - slope * (dmin - d).days)


# ============================================================
# PART 2
# Natural Gas Storage Contract Pricing Model
# ============================================================

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    rate,
    max_volume,
    storage_cost_per_month,
    injection_withdrawal_cost_per_unit,
    price_function
):

    """
    This function calculates the value of a natural gas storage contract.

    The model simulates the cash flows associated with purchasing gas
    during injection periods and selling gas during withdrawal periods.

    Contract value = sales revenue − purchase cost − storage cost −
    injection/withdrawal transaction costs.

    Assumptions:
    - No interest rate or discounting
    - No transportation delay
    - Injection/withdrawal occurs at a fixed rate
    """

    inj = [pd.to_datetime(d).date() for d in injection_dates]
    wd = [pd.to_datetime(d).date() for d in withdrawal_dates]

    all_dates = sorted(set(inj + wd))

    volume = 0
    purchase_cost = 0
    sale_revenue = 0

    for d in all_dates:

        # Injection event (purchase gas for storage)
        if d in inj:

            if volume + rate <= max_volume:

                price = price_function(d)

                purchase_cost += rate * price
                purchase_cost += rate * injection_withdrawal_cost_per_unit

                volume += rate

        # Withdrawal event (sell stored gas)
        if d in wd:

            if volume - rate >= 0:

                price = price_function(d)

                sale_revenue += rate * price
                sale_revenue -= rate * injection_withdrawal_cost_per_unit

                volume -= rate

    # Estimate total storage cost across the contract duration
    start = min(inj)
    end = max(wd)

    months = max(1, math.ceil((end - start).days / 30))

    storage_cost = months * storage_cost_per_month

    contract_value = sale_revenue - purchase_cost - storage_cost

    return float(contract_value)


# ============================================================
# Example Test Scenario
# ============================================================

if __name__ == "__main__":

    injection_dates = ["2022-01-31", "2022-02-28", "2022-03-31"]

    withdrawal_dates = ["2022-06-30", "2022-07-31", "2022-08-31"]

    rate = 100000
    max_volume = 500000

    storage_cost_per_month = 10000
    injection_withdrawal_cost_per_unit = 0.0005

    value = price_storage_contract(
        injection_dates,
        withdrawal_dates,
        rate,
        max_volume,
        storage_cost_per_month,
        injection_withdrawal_cost_per_unit,
        estimate_price
    )

    print("Estimated storage contract value:", value)
