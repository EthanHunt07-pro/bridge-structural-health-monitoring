"""
Bridge Structural Health Monitoring System
------------------------------------------

Streamlit application for predicting bridge damage
using vibration sensor data and a trained Neural Network model.
"""

# ============================================
# Import Libraries
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import os


from config import DATA_DIR
from predictor import predict_damage
from class_labels import CLASS_LABELS
from report_generator import generate_pdf_report


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Bridge Structural Health Monitoring",
    page_icon="🌉",
    layout="wide"
)

# ============================================
# Load Demo Dataset
# ============================================

X_test = np.load(DATA_DIR / "processed" / "X_test.npy")
y_test = np.load(DATA_DIR / "processed" / "y_test.npy")

# ============================================
# Sidebar
# ============================================

st.sidebar.title("📘 Project Information")

st.sidebar.markdown("""
### Model

- Neural Network
- TensorFlow / Keras

### Dataset

- Z24 Bridge Dataset
- 27 Accelerometers
- 6000 Time Steps
- 17 Damage Classes

### Feature Engineering

For each sensor:

- Mean
- Standard Deviation
- RMS
- Maximum
- Minimum
- Peak-to-Peak
- Energy

Total Features = **189**
""")

st.sidebar.info(
    "This application predicts the bridge damage class "
    "using vibration sensor data."
)

# ============================================
# Main Title
# ============================================

st.title("🌉 Bridge Structural Health Monitoring")

st.markdown("""
This application predicts the structural health condition of a bridge
using vibration sensor data and a trained Neural Network model.

The application supports two prediction modes:

- 🧪 Demo Mode (Unseen Test Samples)
- 📤 Upload Your Own Data
""")

# ============================================
# Prediction Mode
# ============================================

st.divider()

st.header("Prediction Mode")

prediction_mode = st.radio(
    "Select how you want to predict the bridge condition:",
    (
        "🧪 Demo Mode",
        "📤 Upload Your Own Data"
    )
)


# ============================================
# Demo Mode
# ============================================

if prediction_mode == "🧪 Demo Mode":

    st.subheader("🧪 Demo Prediction")

    st.info(f"""
        ### 🧪 About Demo Mode

        This mode uses the **20% unseen test dataset** that was **never used during model training**.

        It demonstrates how well the neural network generalizes to new bridge vibration signals.

        **Dataset Information**

        - Unseen Test Samples: **{len(X_test)}**
        - Sensors per Sample: **27**
        - Time Steps per Sensor: **6000**
        - Damage Classes: **17**
        """)

    sample_index = st.selectbox(
        "Choose a Test Sample",
        range(len(X_test)),
        format_func=lambda x: f"Test Sample {x:03d}"
    )

    st.caption(
        f"""
        Selected Sample: **#{sample_index}**

        Signal Shape: **{X_test[sample_index].shape[0]} Sensors × {X_test[sample_index].shape[1]} Time Steps**
        """
        )

    if st.button("🔍 Predict Damage"):

        sample = X_test[sample_index]

        predicted_class, confidence, probabilities = predict_damage(sample)

        with st.container(border=True):
            st.subheader("📊Prediction Result")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.metric(
                    "Predicted Damage Class",
                    CLASS_LABELS[predicted_class]
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{float(confidence)*100:.2f}%"
                )
        
        # ============================================
        # Prediction Evaluation
        # ============================================

        actual_class = int(y_test[sample_index])

        with st.container(border=True):
            st.subheader("✅ Ground Truth Verification")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Actual Damage Class",
                    CLASS_LABELS[actual_class]
                )

            with col2:  
                if predicted_class == actual_class:
                    st.success("✅ Correct Prediction")
                else:
                    st.error("❌ Incorrect Prediction")

        # ============================================
        # Prediction Probability Distribution
        # ============================================

        probability_df = pd.DataFrame({
            "Damage Class": list(CLASS_LABELS.values()),
            "Probability": probabilities * 100
        })

        probability_df["Prediction"] = [
            "Predicted"
            if i == predicted_class
            else "other"
            for i in range(len(probability_df))
        ]

        probability_df = probability_df.sort_values(
            by="Probability",
            ascending=False
        )

        probability_df = probability_df.head(5)

        st.subheader("📈 Top 5 Prediction Probabilities")

        chart = (
            alt.Chart(probability_df)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                x=alt.X(
                    "Probability:Q",
                    title="Confidence (%)"
                ),
                y=alt.Y(
                    "Damage Class:N",
                    sort=None,
                    title=""
                ),
                color=alt.Color(
                    "Prediction:N",
                    scale=alt.Scale(
                        domain=["Predicted", "Other"],
                        range=["#2ECC71", "#8E8E93"]
                    ),
                    legend=None
                ),
                tooltip=[
                    "Damage Class",
                    alt.Tooltip(
                        "Probability:Q",
                        format=".2f",
                        title="Confidence (%)"
                    )
                ]
            )
            .properties(
                height=500
            )
        )

        st.altair_chart(chart, width='stretch')

# ==========================================
# Upload Your Own Data
# ==========================================

elif prediction_mode == "📤 Upload Your Own Data":

    st.subheader("📤 Upload Bridge Vibration Data")

    st.info(
        """
        Upload a NumPy (.npy) file containing raw bridge vibration signals.

        Expected format:

        • 27 Sensors
        • 6000 Time Steps
        • Shape = (27, 6000)
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a NumPy (.npy) file",
        type=["npy"]
    )

    if uploaded_file is None:
        st.warning("Please upload a .npy file to continue.")
        st.stop()
    

    try:

        sample = np.load(uploaded_file)

        st.success("File uploaded successfully!")

    except Exception:

        st.error("Unable to read the uploaded NumPy file.")

        st.stop()

    st.subheader("Uploaded Data")

    st.write(f"Shape : {sample.shape}")

    st.write(f"Data Type : {sample.dtype}")

    st.subheader("Signal Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Minimum", f"{sample.min():.4f}")

    with col2:
        st.metric("Maximum", f"{sample.max():.4f}")

    with col3:
        st.metric("Mean", f"{sample.mean():.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Standard Deviation", f"{sample.std():.4f}")

    with col2:
        st.metric("RMS", f"{np.sqrt(np.mean(sample**2)):.4f}")

    statistics = {
        "Minimum": float(sample.min()),
        "Maximum": float(sample.max()),
        "Mean": float(sample.mean()),
        "Standard Deviation": float(sample.std()),
        "RMS": float(np.sqrt(np.mean(sample**2)))
    }

    # Validate uploaded data
    if sample.shape != (27, 6000):
        st.error("❌ Invalid input shape!")
        st.error(f"Expected: (27, 6000)")
        st.error(f"Received: {sample.shape}")
        st.stop()

    st.success("✅ File format validated successfully!")

    if st.button("🔍 Predict Damage", key="upload_predict_button"):

        predicted_class, confidence, probabilities = predict_damage(sample)

        st.success("Prediction completed successfully!")

        pdf_path = generate_pdf_report(
            sample=sample,
            predicted_class=CLASS_LABELS[predicted_class],
            confidence=float(confidence) * 100,
            probabilities=probabilities,
            statistics=statistics,
            filename=uploaded_file.name.replace(".npy", "")
        )

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Inspection Report",
                data=pdf_file,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )

        st.divider()

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Damage Class",
                CLASS_LABELS[predicted_class]
            )

        with col2:
            st.metric(
                "Confidence",
                f"{float(confidence)*100:.2f}%"
            )

        # ============================================
        # Prediction Probability Distribution
        # ============================================
        
        probability_df = pd.DataFrame({
            "Damage Class": list(CLASS_LABELS.values()),
            "Probability": probabilities * 100
        })
        
        probability_df["Prediction"] = [
            "Predicted"
            if i == predicted_class
            else "other"
            for i in range(len(probability_df))
        ]
        
        probability_df = probability_df.sort_values(
            by="Probability",
            ascending=False
        )
        
        probability_df = probability_df.head(5)
        
        st.subheader("📈 Top 5 Prediction Probabilities")
        
        chart = (
            alt.Chart(probability_df)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                x=alt.X(
                    "Probability:Q",
                    title="Confidence (%)"
                ),
                y=alt.Y(
                    "Damage Class:N",
                    sort=None,
                    title=""
                ),
                color=alt.Color(
                    "Prediction:N",
                    scale=alt.Scale(
                        domain=["Predicted", "Other"],
                        range=["#2ECC71", "#8E8E93"]
                    ),
                    legend=None
                ),
                tooltip=[
                    "Damage Class",
                    alt.Tooltip(
                        "Probability:Q",
                        format=".2f",
                        title="Confidence (%)"
                    )
                ]
            )
            .properties(
                height=500
            )
        )
        
        st.altair_chart(chart, width='stretch')

        st.subheader("🧠 Model Interpretation")

        if confidence >= 0.95:
            st.success(
                f"""
        ### Very High Confidence Prediction

        The neural network is highly confident that the uploaded bridge vibration
        signal corresponds to:

        **{CLASS_LABELS[predicted_class]}**

        Prediction Confidence: **{confidence*100:.2f}%**

        This indicates the uploaded signal closely matches vibration patterns
        learned during model training.
        """
            )

        elif confidence >= 0.80:
            st.warning(
                f"""
        ### Moderate Confidence Prediction

        The model predicts:

        **{CLASS_LABELS[predicted_class]}**

        Prediction Confidence: **{confidence*100:.2f}%**

        The uploaded signal contains characteristics similar to this damage class,
        although there is some uncertainty.
        """
            )

        else:
            st.error(
                f"""
        ### Low Confidence Prediction

        Prediction Confidence: **{confidence*100:.2f}%**

        The uploaded signal does not strongly resemble any damage class.

        Consider checking:

        • Sensor placement

        • Signal quality

        • Noise level

        • Sampling process
        """
            )