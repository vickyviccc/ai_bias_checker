import streamlit as st
from transformers import pipeline

# Load Fine-Tuned DistilBERT Model
fine_tuned_model_path = "./fine_tuned_distilbert"
fine_tuned_classifier = pipeline(
    "text-classification",
    model=fine_tuned_model_path,
    tokenizer=fine_tuned_model_path
)

# Load d4data Bias Detection Model
d4data_model = pipeline(
    "text-classification",
    model="d4data/bias-detection-model",
    tokenizer="d4data/bias-detection-model",
    framework="tf"
)

# Load Toxic-BERT Model
toxic_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    tokenizer="unitary/toxic-bert"
)

# Streamlit App
st.title("AI-Powered Multi-Model Bias Checker")
st.write("Analyze text for potential biases using multiple specialized models.")

# Input text
uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded Text:", text, height=200)
else:
    text = st.text_area("Enter the text to analyze:")

# Set thresholds
distilbert_threshold = 0.50  # Lowered threshold
d4data_threshold = 0.45
toxicbert_threshold = 0.50

if st.button("Analyze"):
    if text.strip():
        st.subheader("Bias Analysis Results")

        # Fine-Tuned DistilBERT
        fine_tuned_result = fine_tuned_classifier(text)
        distilbert_label = fine_tuned_result[0]['label']
        distilbert_confidence = fine_tuned_result[0]['score']
        st.write("### Fine-Tuned DistilBERT Results")
        st.write(f"Raw Output: {fine_tuned_result}")
        if distilbert_label == "LABEL_1" and distilbert_confidence >= distilbert_threshold:
            st.error(f"DistilBERT: Biased content detected with confidence {distilbert_confidence:.2f}.")
        else:
            st.success("DistilBERT: No bias detected.")

        # d4data Bias Detection Model
        d4data_result = d4data_model(text)
        bias_label = d4data_result[0]['label']
        bias_confidence = d4data_result[0]['score']
        st.write("### d4data Bias Detection Results")
        st.write(f"Raw Output: {d4data_result}")
        if bias_label == "Biased" and bias_confidence >= d4data_threshold:
            st.error(f"d4data: Biased content detected with confidence {bias_confidence:.2f}.")
        else:
            st.success("d4data: No bias detected.")

        # Toxic-BERT
        toxic_result = toxic_model(text)
        toxic_label = toxic_result[0]['label']
        toxic_confidence = toxic_result[0]['score']
        st.write("### Toxic-BERT Results")
        st.write(f"Raw Output: {toxic_result}")
        if toxic_label == "LABEL_1" and toxic_confidence >= toxicbert_threshold:
            st.error(f"Toxic-BERT: Toxic content detected with confidence {toxic_confidence:.2f}.")
        else:
            st.success("Toxic-BERT: No toxic content detected.")

        # Comparison Across Models with Manual Review
        st.subheader("Comparison Across Models")
        bias_detected = []

        if distilbert_label == "LABEL_1" and distilbert_confidence >= distilbert_threshold:
            bias_detected.append("Fine-Tuned DistilBERT")

        if bias_label == "Biased" and bias_confidence >= d4data_threshold:
            bias_detected.append("d4data Bias Detection Model")

        if toxic_label == "LABEL_1" and toxic_confidence >= toxicbert_threshold:
            bias_detected.append("Toxic-BERT")

        if bias_detected:
            st.warning(f"Bias detected by: {', '.join(bias_detected)}. Review manually.")
        else:
            st.success("No bias detected across all models.")

        # Rule-based flagging for additional checks (e.g., stereotypes)
        st.subheader("Rule-Based Checks")
        if "men can't" in text.lower() or "women can't" in text.lower():
            st.warning("Rule-based check: Stereotype detected. Review manually.")
    else:
        st.error("Please enter or upload text for analysis.")
