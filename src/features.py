import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pre-payment features for fraud scoring.
    No post-payment balances, no step or hour, no account IDs.

    Parameters:
    df (pd.DataFrame): raw PaySim rows.

    Returns:
    pd.DataFrame: feature table with the same index as df.
    """
    out = pd.DataFrame(index=df.index)
    out["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    out["amount"] = df["amount"]
    out["oldbalanceOrg"] = df["oldbalanceOrg"]
    out["oldbalanceDest"] = df["oldbalanceDest"]

    out["orig_balance_zero"] = (out["oldbalanceOrg"] == 0).astype(int)
    out["dest_balance_zero"] = (out["oldbalanceDest"] == 0).astype(int)

    # Model A only (dropped for Model B)
    out["drains_account"] = (
        (out["oldbalanceOrg"] > 0) & (out["amount"] == out["oldbalanceOrg"])
    ).astype(int)
    out["amount_to_balance"] = out["amount"] / out["oldbalanceOrg"].where(out["oldbalanceOrg"] > 0)

    return out


MODEL_B_DROP = ["drains_account", "amount_to_balance"]