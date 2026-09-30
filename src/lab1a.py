"""
ECM-402: AI for Signal Processing
Lab 1A — Wireless Channel Modeling using Linear Regression

Problems covered:
1. MyLinearRegression from scratch
2. Joint estimation of FSPL(d0) and path-loss exponent
3. Shadowing standard deviation estimation
4. Visualization of fitted models
5. Received-power prediction and threshold probability

Roll Number: 523107
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from pathlib import Path
import warnings


# ============================================================
# 0. SETUP
# ============================================================

np.random.seed(0)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = True

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "pathloss.txt"

FIGURES_DIR = ROOT_DIR / "results" / "figures"
TABLES_DIR = ROOT_DIR / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. SYSTEM PARAMETERS AND DATA
# ============================================================

c = 3e8
fc = 3.5e9
d0 = 1.0

FSPL_d0_theory = 20 * np.log10(
    4 * np.pi * d0 * fc / c
)

print("=" * 70)
print("ECM-402 LAB 1A")
print("Wireless Channel Modeling using Linear Regression")
print("=" * 70)

print(
    f"\nTheoretical free-space path loss at d0 = {d0} m, "
    f"fc = {fc / 1e9} GHz:"
)
print(f"FSPL(d0) = {FSPL_d0_theory:.3f} dB")


def synthesize_pathloss(
    n_true,
    sigma_true,
    n_samples=150,
    d_min=10.0,
    d_max=1800.0,
    seed=1
):
    """
    Generate a synthetic measurement campaign using the CI path-loss model.

    PL(d) = FSPL(d0)
            + 10*n*log10(d/d0)
            + X_sigma
    """

    rng = np.random.default_rng(seed)

    log_d = rng.uniform(
        np.log10(d_min),
        np.log10(d_max),
        size=n_samples
    )

    d = 10 ** log_d

    shadowing = rng.normal(
        0,
        sigma_true,
        size=n_samples
    )

    PL = (
        FSPL_d0_theory
        + 10 * n_true * np.log10(d / d0)
        + shadowing
    )

    return d, PL


# Synthetic fallback values.
# These are used only when data/pathloss.txt is absent.
N_TRUE = 3.0
SIGMA_TRUE = 6.0

using_real_data = DATA_PATH.exists()

if using_real_data:

    data = np.loadtxt(
        DATA_PATH,
        comments="#"
    )

    d_meas = data[:, 0]
    PL_meas = data[:, 1]

    print(
        f"\nLoaded REAL measurement file: "
        f"{DATA_PATH} ({len(d_meas)} points)."
    )

else:

    d_meas, PL_meas = synthesize_pathloss(
        N_TRUE,
        SIGMA_TRUE
    )

    print(
        f"\n[pathloss.txt not found] "
        f"Using synthetic fallback data."
    )

    print(
        f"True n = {N_TRUE}, "
        f"true sigma = {SIGMA_TRUE} dB"
    )


# ============================================================
# Raw measurement plot
# ============================================================

plt.figure(figsize=(7, 4))

plt.scatter(
    d_meas,
    PL_meas,
    s=12,
    alpha=0.6
)

plt.xscale("log")

plt.xlabel("Distance d (m, log scale)")
plt.ylabel("Measured path loss (dB)")
plt.title("Raw Measurement Data")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "raw_measurements.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Problem 1 — MyLinearRegression
# ============================================================

class MyLinearRegression:
    """
    Simple linear regression:

        y = theta0 + theta1*x

    Implemented from scratch using NumPy.

    Supported methods:
        - closed_form
        - batch_gd
        - sgd
    """

    def __init__(self):

        self.theta = None

        self.loss_history_ = None


    @staticmethod
    def _design_matrix(x):

        x = np.asarray(x).reshape(-1)

        return np.column_stack(
            [
                np.ones_like(x),
                x
            ]
        )


    @staticmethod
    def _mse(Phi, y, theta):

        residual = Phi @ theta - y

        return np.mean(
            residual ** 2
        )


    def fit(
        self,
        x,
        y,
        method="closed_form",
        lr=1e-3,
        epochs=2000,
        random_state=0
    ):

        Phi = self._design_matrix(x)

        y = np.asarray(y).reshape(-1)

        n = len(y)


        # ----------------------------------------------------
        # Closed-form solution
        # ----------------------------------------------------

        if method == "closed_form":

            self.theta = np.linalg.solve(
                Phi.T @ Phi,
                Phi.T @ y
            )

            self.loss_history_ = None


        # ----------------------------------------------------
        # Batch Gradient Descent
        # ----------------------------------------------------

        elif method == "batch_gd":

            theta = np.zeros(2)

            history = np.empty(
                epochs
            )

            for epoch in range(epochs):

                residual = Phi @ theta - y

                grad = (
                    (2.0 / n)
                    * (Phi.T @ residual)
                )

                theta = theta - lr * grad

                history[epoch] = self._mse(
                    Phi,
                    y,
                    theta
                )

            self.theta = theta

            self.loss_history_ = history


        # ----------------------------------------------------
        # Stochastic Gradient Descent
        # ----------------------------------------------------

        elif method == "sgd":

            rng = np.random.default_rng(
                random_state
            )

            theta = np.zeros(2)

            history = np.empty(
                epochs
            )

            for epoch in range(epochs):

                order = rng.permutation(n)

                for i in order:

                    phi_i = Phi[i]

                    residual_i = (
                        phi_i @ theta
                        - y[i]
                    )

                    grad_i = (
                        2.0
                        * residual_i
                        * phi_i
                    )

                    theta = (
                        theta
                        - lr * grad_i
                    )

                history[epoch] = self._mse(
                    Phi,
                    y,
                    theta
                )

            self.theta = theta

            self.loss_history_ = history


        else:

            raise ValueError(
                f"Unknown method '{method}'. "
                "Use 'closed_form', 'batch_gd', or 'sgd'."
            )


        return self


    def predict(self, x):

        Phi = self._design_matrix(x)

        return Phi @ self.theta


    @property
    def intercept_(self):

        return self.theta[0]


    @property
    def slope_(self):

        return self.theta[1]


print(
    "\nMyLinearRegression implemented with:"
)

print(
    "closed_form, batch_gd, sgd"
)


# ============================================================
# Problem 2 — Estimate FSPL(d0) and Path-Loss Exponent
# ============================================================

x_feat = np.log10(
    d_meas / d0
)

y_target = PL_meas


# ============================================================
# Learning-rate sweep
# ============================================================

print("\n" + "=" * 70)
print("LEARNING-RATE SWEEP")
print("=" * 70)

lr_results = []

for lr_test in [
    0.5,
    0.2,
    0.1,
    0.05,
    0.01,
    1e-3
]:

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning
        )

        model_test = MyLinearRegression().fit(
            x_feat,
            y_target,
            method="batch_gd",
            lr=lr_test,
            epochs=500
        )

    final_loss = (
        model_test.loss_history_[-1]
    )

    diverged = (
        not np.isfinite(final_loss)
        or final_loss > 1e6
    )

    if diverged:

        loss_display = "DIVERGED"

    else:

        loss_display = f"{final_loss:.4f}"

    print(
        f"lr={lr_test:<8g} "
        f"final MSE (500 epochs) = "
        f"{loss_display}"
    )

    lr_results.append(
        {
            "Learning Rate": lr_test,
            "Final MSE": (
                np.nan
                if diverged
                else final_loss
            ),
            "Status": (
                "DIVERGED"
                if diverged
                else "CONVERGED"
            )
        }
    )


lr_table = pd.DataFrame(
    lr_results
)

lr_table.to_csv(
    TABLES_DIR / "learning_rate_sweep.csv",
    index=False
)


# ============================================================
# Final models
# ============================================================

LR = 0.05
EPOCHS = 1500

model_cf = MyLinearRegression().fit(
    x_feat,
    y_target,
    method="closed_form"
)

model_gd = MyLinearRegression().fit(
    x_feat,
    y_target,
    method="batch_gd",
    lr=LR,
    epochs=EPOCHS
)

model_sgd = MyLinearRegression().fit(
    x_feat,
    y_target,
    method="sgd",
    lr=LR,
    epochs=EPOCHS,
    random_state=0
)


results = pd.DataFrame(
    {
        "Method": [
            "Closed-form",
            "Batch GD",
            "SGD"
        ],

        "FSPL(d0) [dB]": [
            model_cf.intercept_,
            model_gd.intercept_,
            model_sgd.intercept_
        ],

        "n (path-loss exp.)": [
            model_cf.slope_ / 10,
            model_gd.slope_ / 10,
            model_sgd.slope_ / 10
        ]
    }
)


print(
    "\n" + results.to_string(
        index=False
    )
)


# ============================================================
# Problem 2 — Training Loss
# ============================================================

closed_form_mse = (
    MyLinearRegression._mse(
        MyLinearRegression._design_matrix(
            x_feat
        ),
        y_target,
        model_cf.theta
    )
)


plt.figure(figsize=(8, 4.5))

plt.semilogy(
    model_gd.loss_history_,
    label="Batch Gradient Descent"
)

plt.semilogy(
    model_sgd.loss_history_,
    label="SGD (per-epoch MSE)",
    alpha=0.8
)

plt.axhline(
    closed_form_mse,
    linestyle="--",
    linewidth=1,
    label="Closed-form MSE"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (log scale)")
plt.title("Training Loss: Batch GD vs SGD")

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Problem 3 — Shadowing Variance
# ============================================================

def estimate_sigma(
    model,
    x,
    y
):

    residuals = (
        y
        - model.predict(x)
    )

    sigma2_hat = np.mean(
        residuals ** 2
    )

    return (
        np.sqrt(sigma2_hat),
        residuals
    )


sigma_cf, resid_cf = estimate_sigma(
    model_cf,
    x_feat,
    y_target
)

sigma_gd, resid_gd = estimate_sigma(
    model_gd,
    x_feat,
    y_target
)

sigma_sgd, resid_sgd = estimate_sigma(
    model_sgd,
    x_feat,
    y_target
)


summary_table = pd.DataFrame(
    {
        "Method": [
            "Closed-form",
            "Batch GD",
            "SGD"
        ],

        "FSPL(d0) [dB]": [
            model_cf.intercept_,
            model_gd.intercept_,
            model_sgd.intercept_
        ],

        "n": [
            model_cf.slope_ / 10,
            model_gd.slope_ / 10,
            model_sgd.slope_ / 10
        ],

        "sigma [dB]": [
            sigma_cf,
            sigma_gd,
            sigma_sgd
        ]
    }
)


print(
    "\n" + summary_table.to_string(
        index=False
    )
)


summary_table.to_csv(
    TABLES_DIR / "regression_summary.csv",
    index=False
)


if not using_real_data:

    print(
        f"\nSynthetic-data ground truth:"
    )

    print(
        f"FSPL(d0) = "
        f"{FSPL_d0_theory:.3f} dB"
    )

    print(
        f"n = {N_TRUE}"
    )

    print(
        f"sigma = {SIGMA_TRUE} dB"
    )


# ============================================================
# Problem 4 — Visualization
# ============================================================

d_grid = np.logspace(
    np.log10(d_meas.min()),
    np.log10(d_meas.max()),
    200
)

x_grid = np.log10(
    d_grid / d0
)


plt.figure(figsize=(8, 5))

plt.scatter(
    x_feat,
    y_target,
    s=12,
    alpha=0.4,
    label="Measured path loss"
)

plt.plot(
    x_grid,
    model_cf.predict(x_grid),
    label="Closed-form fit",
    linewidth=2
)

plt.plot(
    x_grid,
    model_gd.predict(x_grid),
    "--",
    label="Batch GD fit",
    linewidth=2
)

plt.plot(
    x_grid,
    model_sgd.predict(x_grid),
    ":",
    label="SGD fit",
    linewidth=2.5
)

plt.xlabel(
    r"$\log_{10}(d/d_0)$"
)

plt.ylabel(
    "Path loss (dB)"
)

plt.title(
    "Measured Path Loss vs log-distance"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "regression_fits.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Problem 5 — Received Power Prediction
# ============================================================

Pt_watts = 100.0

Pt_dBm = (
    10
    * np.log10(
        Pt_watts * 1000
    )
)

print(
    "\n" + "=" * 70
)
print("PROBLEM 5 — RECEIVED POWER")
print("=" * 70)

print(
    f"\nTransmit power: "
    f"{Pt_watts} W = "
    f"{Pt_dBm:.3f} dBm"
)


d_query = 2000.0


if d_query > d_meas.max():

    print(
        f"\nNote: d = {d_query:.0f} m is beyond "
        f"the maximum measured distance "
        f"({d_meas.max():.1f} m)."
    )

    print(
        "This is a mild extrapolation, so "
        "the prediction carries more uncertainty."
    )


PL_pred_dB = model_cf.predict(
    np.array(
        [
            np.log10(
                d_query / d0
            )
        ]
    )
)[0]


Pr_mean_dBm = (
    Pt_dBm
    - PL_pred_dB
)


print(
    f"\nPredicted deterministic path loss "
    f"at d = {d_query:.0f} m: "
    f"{PL_pred_dB:.3f} dB"
)

print(
    f"Mean received power at "
    f"d = {d_query:.0f} m: "
    f"{Pr_mean_dBm:.3f} dBm"
)


# ============================================================
# Probability that received power exceeds -95 dBm
# ============================================================

threshold_dBm = -95.0

z = (
    threshold_dBm
    - Pr_mean_dBm
) / sigma_cf


prob_above_threshold = (
    1
    - stats.norm.cdf(z)
)


print(
    f"\nP(received power > "
    f"{threshold_dBm:.0f} dBm) "
    f"at d = {d_query:.0f} m "
    f"= {prob_above_threshold:.4f} "
    f"({prob_above_threshold * 100:.2f}%)"
)


z_abs = abs(z)

side = (
    "above"
    if threshold_dBm < Pr_mean_dBm
    else "below"
)


print(
    f"\nMean received power: "
    f"{Pr_mean_dBm:.2f} dBm"
)

print(
    f"Shadowing standard deviation: "
    f"{sigma_cf:.2f} dB"
)

print(
    f"The {threshold_dBm:.0f} dBm threshold "
    f"sits {z_abs:.2f} standard deviations "
    f"{side} the mean."
)


# ============================================================
# Final results
# ============================================================

final_results = pd.DataFrame(
    {
        "Quantity": [
            "Theoretical FSPL(d0)",
            "Estimated FSPL(d0)",
            "Estimated path-loss exponent n",
            "Estimated shadowing sigma",
            "Transmit power",
            "Predicted path loss at 2 km",
            "Mean received power at 2 km",
            "Probability Pr > -95 dBm"
        ],

        "Value": [
            FSPL_d0_theory,
            model_cf.intercept_,
            model_cf.slope_ / 10,
            sigma_cf,
            Pt_dBm,
            PL_pred_dB,
            Pr_mean_dBm,
            prob_above_threshold
        ]
    }
)


final_results.to_csv(
    TABLES_DIR / "final_results.csv",
    index=False
)


print(
    "\n" + "=" * 70
)

print(
    "LAB 1A COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nFigures saved in: "
    f"{FIGURES_DIR}"
)

print(
    f"Tables saved in: "
    f"{TABLES_DIR}"
)