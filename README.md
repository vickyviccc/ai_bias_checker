AI Bias Checker


Overview

The AI Bias Checker is a powerful multi-model tool designed to analyze text for potential bias or toxic content. This tool integrates multiple AI models to provide a comprehensive perspective on bias detection, including a fine-tuned DistilBERT model, d4data Bias Detection, and Toxic-BERT.

Features

Multi-Model Integration: Combines multiple AI models to ensure diverse perspectives on bias detection.
Fine-Tuned Model: Includes a custom fine-tuned DistilBERT model trained on specific bias-related tasks.
Rule-Based Checks: Additional rule-based validation for common stereotypes.
Detailed Results: Displays raw outputs and confidence scores for in-depth analysis.
Customizable Thresholds: Adjust model thresholds for more precise detection.

How It Works

The bias checker uses a combination of AI models:
Fine-Tuned DistilBERT: Custom-trained for bias detection using labeled data.
d4data Bias Detection: Pretrained on the MBAD dataset to detect fairness and bias.
Toxic-BERT: Identifies toxic and harmful content in text.

Example Usage

Input: "Women are less capable of leadership roles."
Output:
- DistilBERT: Bias detected with confidence 0.85
- d4data: Bias detected with confidence 0.88
- Toxic-BERT: No toxic content detected

Project Files

bias_checker.py: Main script for running the Streamlit-based app.
fine_tune_distilbert.py: Code for fine-tuning the DistilBERT model.
requirements.txt: Python dependencies for running the project.

Setup Instructions

Clone this repository:
git clone https://github.com/yourusername/ai-bias-checker.git
cd ai-bias-checker
Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run bias_checker.py

Fine-Tuned Model
The fine-tuned DistilBERT model used in this project was trained on a dataset focusing on bias in leadership, gender stereotypes, and toxic behavior.

Future Enhancements

Explore larger datasets for fine-tuning.
Add multilingual bias detection.
Incorporate more pretrained models for broader analysis
