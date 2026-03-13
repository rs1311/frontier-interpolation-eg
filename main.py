import pandas as pd
import numpy as np


# This script uses numeric inputs stated in the case:
#   - Atlas 4.3 metrics
#   - Atlas-FC metrics
#   - Internal fairness targets from Appendix B
#   - Risk/cost figures from Sections 7 and Appendix A
#
# Logic:
# 1) Atlas 4.4 is a structured alternative between Atlas 4.3 and Atlas-FC.
# 2) The Board wants to preserve as much 4.3 performance as possible.
# 3) Atlas 4.4 must also materially improve fairness and cohort false positives.
# 4) Thus, Atlas 4.4 is defined as the MINIMUM shift from 4.3 toward FC
#    needed to satisfy all explicit internal fairness thresholds.


# 1. metrics given by the case
atlas_43 = {
    "AUC": 81.0,
    "Efficiency_Uplift_pct": 14.6,
    "FP_Avg_pct": 12.0,
    "FP_LongTenure_pct": 18.0,
    "OverPrediction_pct": 3.0,
    "CareerBreak_Odds": 1.4,
    "Fairness_Score": 40.0,
    "Titan_Net_Savings_GBPm": 400.0,
    "Expected_Legal_GBPm": 7.1,
    "Expected_Regulatory_GBPm": 4.0,
    "Reputational_Low_GBPm": 9.0,
    "Reputational_High_GBPm": 15.0,
}

atlas_fc = {
    "AUC": 71.0,
    "Efficiency_Uplift_pct": 9.8,
    "FP_Avg_pct": 9.0,
    "FP_LongTenure_pct": 11.0,
    "OverPrediction_pct": 1.0,
    "CareerBreak_Odds": 1.1,
    "Fairness_Score": 85.0,
    "Titan_Net_Savings_GBPm": 280.0,
    "Expected_Legal_GBPm": 4.1,
    "Expected_Regulatory_GBPm": 2.0,
    "Reputational_Low_GBPm": 4.0,
    "Reputational_High_GBPm": 8.0,
}

# 2. internal fairness targets
targets = {
    "AUC_min": 70.0,
    "FP_Avg_max": 10.0,
    "FP_LongTenure_max": 13.0,
    "OverPrediction_max": 2.0,
    "CareerBreak_Odds_max": 1.2,
}

# solve for minimum lambda shift (weighted interpolation)
# lambda = 0  -> Atlas 4.3
# lambda = 1  -> Atlas-FC
#
# for each and every fairness metric with a maximum allowed value:
#   metric_44 = metric_43 + lambda * (metric_fc - metric_43)
# Need metric_44 <= target
# Solve:
#   lambda >= (target - metric_43) / (metric_fc - metric_43)
#
# Because metric_fc < metric_43 for these fairness-risk metrics,
# this produces a positive threshold in [0, 1].
#
# We choose the maximum of these required lambdas.
# That yields the smallest move away from Atlas 4.3 that still
# satisfies every explicit fairness target.
def lambda_needed_upper_bound(v43, vfc, target_max):
    # solve v43 + λ(vfc-v43) <= target_max
    if v43 <= target_max:
        return 0.0
    if vfc == v43:
        return np.inf
    return (target_max - v43) / (vfc - v43)

def lambda_needed_lower_bound(v43, vfc, target_min):
    # solve v43 + lambda(vfc-v43) >= target_min
    if v43 >= target_min:
        return 0.0
    if vfc == v43:
        return np.inf
    return (target_min - v43) / (vfc - v43)

lambda_constraints = {
    "FP_Avg_pct": lambda_needed_upper_bound(
        atlas_43["FP_Avg_pct"], atlas_fc["FP_Avg_pct"], targets["FP_Avg_max"]
    ),
    "FP_LongTenure_pct": lambda_needed_upper_bound(
        atlas_43["FP_LongTenure_pct"], atlas_fc["FP_LongTenure_pct"], targets["FP_LongTenure_max"]
    ),
    "OverPrediction_pct": lambda_needed_upper_bound(
        atlas_43["OverPrediction_pct"], atlas_fc["OverPrediction_pct"], targets["OverPrediction_max"]
    ),
    "CareerBreak_Odds": lambda_needed_upper_bound(
        atlas_43["CareerBreak_Odds"], atlas_fc["CareerBreak_Odds"], targets["CareerBreak_Odds_max"]
    ),
    "AUC": lambda_needed_lower_bound(
        atlas_43["AUC"], atlas_fc["AUC"], targets["AUC_min"]
    ),
}

lambda_star = max(lambda_constraints.values())

if not (0.0 <= lambda_star <= 1.0):
    raise ValueError(
        f"No feasible Atlas 4.4 exists on the 4.3 ↔ FC frontier under the explicit targets. "
        f"Computed lambda*: {lambda_star:.4f}"
    )

#frontier interpolation
def interpolate(v43, vfc, lam):
    return v43 + lam * (vfc - v43)

atlas_44 = {
    k: interpolate(atlas_43[k], atlas_fc[k], lambda_star)
    for k in atlas_43.keys()
}

#logical classifications
atlas_44_labels = {
    "Projected_EU_AI_Act_Compliance": "Conditional / Likely Yes with human-in-the-loop + audit trail",
    "Volatility_Stability": "At least Moderate; improved toward Moderate–High vs Atlas 4.3",
    "Deployment_Form": "Human-reviewed restructuring support, not sole automated termination trigger",
    "Year0_Cost_Treatment": "Offset within Titan's existing £16m implementation budget",
}

gross_margin_contribution_gbpm = 13.0  # case given over 3 years

def rep_mid(low, high):
    return (low + high) / 2.0

summary = pd.DataFrame({
    "Metric": [
        "Predictive Accuracy (AUC, %)",
        "Restructuring Efficiency Uplift (%)",
        "False Positive Rate — Avg (%)",
        "False Positive Rate — Long Tenure >10y (%)",
        "Redundancy Over-Prediction (%)",
        "Career-Break Odds Ratio (x)",
        "Internal Fairness Score (0-100)",
        "Titan Net Savings (£m)",
        "Expected Legal Exposure (£m)",
        "Expected Regulatory Cost (£m)",
        "Reputational Impact Low (£m)",
        "Reputational Impact High (£m)",
        "Reputational Impact Mid (£m)",
        "3Y Gross Margin Contribution to HelixWorks (£m)",
        "Simple Risk-Adjusted Value = GM - Legal - Reg - RepMid (£m)"
    ],
    "Atlas 4.3": [
        atlas_43["AUC"],
        atlas_43["Efficiency_Uplift_pct"],
        atlas_43["FP_Avg_pct"],
        atlas_43["FP_LongTenure_pct"],
        atlas_43["OverPrediction_pct"],
        atlas_43["CareerBreak_Odds"],
        atlas_43["Fairness_Score"],
        atlas_43["Titan_Net_Savings_GBPm"],
        atlas_43["Expected_Legal_GBPm"],
        atlas_43["Expected_Regulatory_GBPm"],
        atlas_43["Reputational_Low_GBPm"],
        atlas_43["Reputational_High_GBPm"],
        rep_mid(atlas_43["Reputational_Low_GBPm"], atlas_43["Reputational_High_GBPm"]),
        gross_margin_contribution_gbpm,
        gross_margin_contribution_gbpm
        - atlas_43["Expected_Legal_GBPm"]
        - atlas_43["Expected_Regulatory_GBPm"]
        - rep_mid(atlas_43["Reputational_Low_GBPm"], atlas_43["Reputational_High_GBPm"])
    ],
    "Atlas 4.4": [
        atlas_44["AUC"],
        atlas_44["Efficiency_Uplift_pct"],
        atlas_44["FP_Avg_pct"],
        atlas_44["FP_LongTenure_pct"],
        atlas_44["OverPrediction_pct"],
        atlas_44["CareerBreak_Odds"],
        atlas_44["Fairness_Score"],
        atlas_44["Titan_Net_Savings_GBPm"],
        atlas_44["Expected_Legal_GBPm"],
        atlas_44["Expected_Regulatory_GBPm"],
        atlas_44["Reputational_Low_GBPm"],
        atlas_44["Reputational_High_GBPm"],
        rep_mid(atlas_44["Reputational_Low_GBPm"], atlas_44["Reputational_High_GBPm"]),
        gross_margin_contribution_gbpm,
        gross_margin_contribution_gbpm
        - atlas_44["Expected_Legal_GBPm"]
        - atlas_44["Expected_Regulatory_GBPm"]
        - rep_mid(atlas_44["Reputational_Low_GBPm"], atlas_44["Reputational_High_GBPm"])
    ],
    "Atlas-FC": [
        atlas_fc["AUC"],
        atlas_fc["Efficiency_Uplift_pct"],
        atlas_fc["FP_Avg_pct"],
        atlas_fc["FP_LongTenure_pct"],
        atlas_fc["OverPrediction_pct"],
        atlas_fc["CareerBreak_Odds"],
        atlas_fc["Fairness_Score"],
        atlas_fc["Titan_Net_Savings_GBPm"],
        atlas_fc["Expected_Legal_GBPm"],
        atlas_fc["Expected_Regulatory_GBPm"],
        atlas_fc["Reputational_Low_GBPm"],
        atlas_fc["Reputational_High_GBPm"],
        rep_mid(atlas_fc["Reputational_Low_GBPm"], atlas_fc["Reputational_High_GBPm"]),
        gross_margin_contribution_gbpm,
        gross_margin_contribution_gbpm
        - atlas_fc["Expected_Legal_GBPm"]
        - atlas_fc["Expected_Regulatory_GBPm"]
        - rep_mid(atlas_fc["Reputational_Low_GBPm"], atlas_fc["Reputational_High_GBPm"])
    ]
})

# fairness checks
target_check = pd.DataFrame({
    "Constraint": [
        "AUC >= 70",
        "FP Avg < 10",
        "FP Long Tenure < 13",
        "Over-Prediction < 2",
        "Career-Break Odds < 1.2"
    ],
    "Atlas 4.4 Value": [
        atlas_44["AUC"],
        atlas_44["FP_Avg_pct"],
        atlas_44["FP_LongTenure_pct"],
        atlas_44["OverPrediction_pct"],
        atlas_44["CareerBreak_Odds"]
    ],
    "Target": [
        ">= 70",
        "<= 10",
        "<= 13",
        "<= 2",
        "<= 1.2"
    ],
    "Pass": [
        atlas_44["AUC"] >= 70.0,
        atlas_44["FP_Avg_pct"] <= 10.0,
        atlas_44["FP_LongTenure_pct"] <= 13.0,
        atlas_44["OverPrediction_pct"] <= 2.0,
        atlas_44["CareerBreak_Odds"] <= 1.2
    ]
})

binding_constraints = (
    pd.Series(lambda_constraints, name="Required lambda")
    .sort_values(ascending=False)
    .to_frame()
)

#print
print("=")
print("ATLAS 4.4 DERIVATION MATH")
print("=")
print("Method:")
print("  Atlas 4.4 is defined as the minimum shift from Atlas 4.3 toward Atlas-FC")
print("  required to satisfy all explicit internal fairness thresholds from Appendix B.")
print()
print(f"lambda* (minimum required fairness shift) = {lambda_star:.6f}")
print()
print("Interpretation:")
print("  - lambda = 0.000000 -> Atlas 4.3")
print("  - lambda = 1.000000 -> Atlas-FC")
print("  - Atlas 4.4 sits at the smallest feasible point that clears every fairness target")
print("    while preserving as much 4.3 performance as possible.")
print()

print("-")
print("BINDING CONSTRAINTS (which fairness requirement forces the shift to occur?)")
print("-")
print(binding_constraints.to_string())
print()

print("-")
print("ATLAS 4.4 NON-NUMERIC BOARD CLASSIFICATIONS")
print("-")
for k, v in atlas_44_labels.items():
    print(f"{k}: {v}")
print()

print("-"*100)
print("ATLAS 4.3 vs ATLAS 4.4 vs ATLAS-FC")
print("-"*100)
with pd.option_context("display.float_format", "{:,.4f}".format):
    print(summary.to_string(index=False))
print()

print("-"*100)
print("ATLAS 4.4 FAIRNESS CHECKS")
print("-"*100)
print(target_check.to_string(index=False))
print()
