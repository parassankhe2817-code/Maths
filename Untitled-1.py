# ============================================================
# SE MINI PROJECT
# Student Commute Time, Canteen Expenditure & Attendance
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Attendance Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Student Commute & Attendance Analysis")

st.write(
    "Statistical analysis of commute time, canteen expenditure "
    "and student attendance."
)


# ============================================================
# 2. DATA INPUT
# ============================================================

st.sidebar.header("📁 Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload your CSV file containing at least 35 student responses."
    )

    st.stop()

df = pd.read_csv(uploaded_file)


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Commute_Time_Min",
    "Canteen_Expenditure_Rs",
    "Attendance_Percent"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The following columns are missing: "
        + ", ".join(missing_columns)
    )

    st.write("Your CSV must contain:")

    for column in required_columns:
        st.write("•", column)

    st.stop()


df = df[required_columns].copy()


# ============================================================
# 4. DATA COLLECTION
# ============================================================

st.header("1. Data Collection")

# Convert columns to numeric
for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


col1, col2, col3 = st.columns(3)

col1.metric(
    "Number of Responses",
    len(df)
)

col2.metric(
    "Average Commute",
    f"{df['Commute_Time_Min'].mean():.2f} min"
)

col3.metric(
    "Average Attendance",
    f"{df['Attendance_Percent'].mean():.2f}%"
)


if len(df) < 35:

    st.warning(
        "The project requires at least 35 responses (N ≥ 35)."
    )

else:

    st.success(
        f"Requirement satisfied: {len(df)} responses available."
    )


st.subheader("Collected Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# ============================================================
# 5. DATA CLEANING & NORMALIZATION
# ============================================================

st.header("2. Data Cleaning & Normalization")


# ---------------- Missing Values ----------------

st.subheader("Missing Values")

missing_values = df.isnull().sum()

st.dataframe(
    missing_values.rename("Missing Values"),
    use_container_width=True
)


# Remove missing values
clean_df = df.dropna().copy()


# ---------------- Outlier Detection ----------------

st.subheader("Outlier Detection using IQR")

mask = pd.Series(
    True,
    index=clean_df.index
)

for column in required_columns:

    Q1 = clean_df[column].quantile(0.25)
    Q3 = clean_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    column_mask = (
        (clean_df[column] >= lower_limit)
        &
        (clean_df[column] <= upper_limit)
    )

    mask = mask & column_mask


original_count = len(clean_df)

clean_df = clean_df[mask].copy()

removed = original_count - len(clean_df)

st.write(
    f"Outliers removed: **{removed}**"
)


# ---------------- Cleaned Dataset ----------------

st.subheader("Cleaned Dataset")

st.dataframe(
    clean_df,
    use_container_width=True
)


# ---------------- Standardization ----------------

st.subheader("Standardized Data")

standardized_df = clean_df.copy()

for column in required_columns:

    mean = clean_df[column].mean()
    std = clean_df[column].std()

    if std != 0:

        standardized_df[column] = (
            clean_df[column] - mean
        ) / std


st.latex(
    r"Z = \frac{X-\bar{X}}{s}"
)

st.dataframe(
    standardized_df.round(3),
    use_container_width=True
)


# ============================================================
# 6. COVARIANCE MATRIX
# ============================================================

st.header("3. Covariance Analysis")

covariance_matrix = clean_df.cov()

st.subheader("Covariance Matrix")

st.dataframe(
    covariance_matrix.round(4),
    use_container_width=True
)


# ============================================================
# 7. BIVARIATE CORRELATION
# ============================================================

st.header("4. Bivariate Correlation")

correlation_matrix = clean_df.corr()

st.subheader("Pearson Correlation Matrix")

st.dataframe(
    correlation_matrix.round(4),
    use_container_width=True
)


X = clean_df["Commute_Time_Min"]
Y = clean_df["Canteen_Expenditure_Rs"]
Z = clean_df["Attendance_Percent"]


# Pearson correlations

r_xy, _ = stats.pearsonr(X, Y)

r_xz, _ = stats.pearsonr(X, Z)

r_yz, _ = stats.pearsonr(Y, Z)


col1, col2, col3 = st.columns(3)

col1.metric(
    "rₓᵧ : Commute ↔ Canteen",
    f"{r_xy:.4f}"
)

col2.metric(
    "rₓ𝓏 : Commute ↔ Attendance",
    f"{r_xz:.4f}"
)

col3.metric(
    "rᵧ𝓏 : Canteen ↔ Attendance",
    f"{r_yz:.4f}"
)


# ============================================================
# 8. PARTIAL CORRELATION
# ============================================================

st.header("5. Partial Correlation")

st.write(
    "Relationship between commute time and attendance "
    "while controlling for canteen expenditure."
)

st.latex(
    r"""
    r_{XZ.Y} =
    \frac{r_{XZ}-r_{XY}r_{YZ}}
    {\sqrt{(1-r_{XY}^{2})(1-r_{YZ}^{2})}}
    """
)

numerator = r_xz - (r_xy * r_yz)

denominator = np.sqrt(
    (1 - r_xy ** 2) *
    (1 - r_yz ** 2)
)

partial_correlation = (
    numerator / denominator
)

st.metric(
    "Partial Correlation",
    f"{partial_correlation:.4f}"
)


# ============================================================
# 9. LINEAR REGRESSION
# ============================================================

st.header("6. Regression Analysis")

st.write(
    "Regression equation of attendance on commute duration."
)


regression = stats.linregress(
    X,
    Z
)

slope = regression.slope

intercept = regression.intercept


predicted = (
    intercept +
    slope * X
)

residuals = Z - predicted


st.latex(
    r"\hat{Z} = a + bX"
)

st.success(
    f"Regression Equation: "
    f"Attendance = {intercept:.4f} "
    f"{slope:+.4f} × Commute Time"
)


# ============================================================
# 10. COEFFICIENT OF DETERMINATION
# ============================================================

st.header("7. Coefficient of Determination")

st.write(
    "R² represents the proportion of variation in attendance "
    "explained by commute time."
)

st.latex(
    r"""
    R^2 =
    1 -
    \frac{\sum(Z_i-\hat Z_i)^2}
    {\sum(Z_i-\bar Z)^2}
    """
)

SS_res = np.sum(
    (Z - predicted) ** 2
)

SS_total = np.sum(
    (Z - Z.mean()) ** 2
)

R_squared = 1 - (
    SS_res / SS_total
)


col1, col2 = st.columns(2)

col1.metric(
    "R²",
    f"{R_squared:.4f}"
)

col2.metric(
    "Variance Explained",
    f"{R_squared * 100:.2f}%"
)


# ============================================================
# 11. ERROR ANALYSIS
# ============================================================

st.header("8. Error Analysis")

st.latex(
    r"e_i = Z_i-\hat Z_i"
)

n = len(Z)

if n > 2:

    residual_variance = (
        SS_res / (n - 2)
    )

else:

    residual_variance = np.nan


st.metric(
    "Residual Variance",
    f"{residual_variance:.4f}"
)


# ============================================================
# 12. CORRELATION HEATMAP
# ============================================================

st.header("9. Data Visualization")

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(
    figsize=(8, 5)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    ax=ax
)

ax.set_title(
    "Pearson Correlation Heatmap"
)

st.pyplot(fig)


# ============================================================
# 13. REGRESSION PLOT
# ============================================================

st.subheader(
    "Commute Time vs Attendance"
)

plot_df = clean_df.copy()

plot_df["Predicted_Attendance"] = predicted


fig_plot = px.scatter(
    plot_df,
    x="Commute_Time_Min",
    y="Attendance_Percent",
    trendline="ols",
    title="Commute Time vs Attendance",
    labels={
        "Commute_Time_Min": "Commute Time (minutes)",
        "Attendance_Percent": "Attendance (%)"
    }
)

st.plotly_chart(
    fig_plot,
    use_container_width=True
)


# ============================================================
# 14. RESIDUAL PLOT
# ============================================================

st.subheader("Residual Plot")

residual_df = pd.DataFrame({
    "Predicted Attendance": predicted,
    "Residual": residuals
})

fig_residual = px.scatter(
    residual_df,
    x="Predicted Attendance",
    y="Residual",
    title="Residual Plot",
    labels={
        "Predicted Attendance":
        "Predicted Attendance (%)",
        "Residual":
        "Residual"
    }
)

fig_residual.add_hline(
    y=0,
    line_dash="dash"
)

st.plotly_chart(
    fig_residual,
    use_container_width=True
)


# ============================================================
# 15. ATTENDANCE PREDICTION
# ============================================================

st.header("10. Attendance Prediction")

st.write(
    "Enter daily commute time to estimate attendance."
)

commute_input = st.slider(
    "Daily Commute Time (minutes)",
    min_value=0,
    max_value=180,
    value=45,
    step=1
)

estimated_attendance = (
    intercept +
    slope * commute_input
)

st.success(
    f"Estimated Attendance: "
    f"**{estimated_attendance:.2f}%**"
)


# ============================================================
# END OF PROJECT
# ============================================================