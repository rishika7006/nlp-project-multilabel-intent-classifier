# you can directly run this app using the command !streamlit run multilabel_classifier_app.py
# on your terminal

# import the required libraries
import os
import math
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import XLNetModel, XLNetTokenizerFast
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

#######################################################################################################################

def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings - 2], text_list))
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids

def create_attn_masks(input_ids):
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
    def __init__(self, num_labels=2):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        last_hidden_state = self.xlnet(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        return torch.mean(last_hidden_state, 1)

#######################################################################################################################

tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased', do_lower_case=True)

label_cols = ['Problem in recharge', 'Problem in reward/redeem points',
              'Problem in registration/login/username/password', 'Problem with customer care service',
              'Other complaints', 'Bad/Irrelevant comments', 'Appreciation']
num_labels = len(label_cols)

model_save_path = 'classifier_model1.pt'
device = torch.device('cpu')
model = XLNetForMultiLabelSequenceClassification(num_labels)
model.load_state_dict(torch.load(model_save_path, map_location=device))

def generate_predictions(model, df, num_labels, device="cpu", batch_size=32):
    num_iter = math.ceil(df.shape[0] / batch_size)
    pred_probs = np.array([]).reshape(0, num_labels)

    model.to(device)
    model.eval()

    for i in range(num_iter):
        df_subset = df.iloc[i * batch_size:(i + 1) * batch_size, :]
        X = df_subset["features"].values.tolist()
        masks = df_subset["masks"].values.tolist()
        X = torch.tensor(X)
        masks = torch.tensor(masks, dtype=torch.long)
        X = X.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            logits = model(input_ids=X, attention_mask=masks)
            logits = logits.sigmoid().detach().cpu().numpy()
            pred_probs = np.vstack([pred_probs, logits])

    return pred_probs

##################################################################################################################################

# Custom CSS for styling - Compact version
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 1rem 2rem !important;
        }
        .stTextArea>div>textarea {
            min-height: 120px;
            font-size: 14px;
            padding: 12px !important;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .stButton>button {
            background-color: #4a4e69;
            color: white;
            border-radius: 6px;
            padding: 10px 24px;
            font-size: 14px;
            margin: 10px 0;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3a3e59;
        }
        .header-text {
            font-size: 1.8rem;
            font-weight: 600;
            color: #22223b;
            margin-bottom: 1rem;
            text-align: center;
        }
        .label-card {
            background-color: white;
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 13px;
            line-height: 1.3;
            text-align: center;
            color: #22223b;
        }
        .label-container {
            margin: 0 -5px;
        }
        .result-card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: #000 !important;
        }
        .progress-bar {
            height: 16px;
            border-radius: 8px;
            background-color: #e9ecef;
            margin: 12px 0;
            color: #000 !important;
        }
        .progress-fill {
            height: 100%;
            border-radius: 8px;
            background-color: #4a4e69;
            color: #000 !important;
        }
        .sidebar .sidebar-content {
            background-color: #22223b;
            color: white;
            padding: 1.5rem !important;
        }
        .sidebar .stMarkdown h1, 
        .sidebar .stMarkdown h2, 
        .sidebar .stMarkdown h3 {
            color: white !important;
        }
        .sidebar .stInfo {
            background-color: #4a4e69 !important;
            color: white !important;
            border-radius: 8px;
            padding: 12px;
        }
        @media screen and (max-width: 768px) {
            .main {
                padding: 0.5rem 1rem !important;
            }
            .label-card {
                font-size: 12px;
                padding: 6px 8px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("## About App")
    st.info(
        "This tool classifies banking app reviews by detecting sentiment and multiple user intents simultaneously, providing a confidence score for each prediction."
    )
    st.markdown("---")
    st.markdown("### Contact")
    st.markdown("`rishikavaish321@gmail.com`")

# Main content
st.markdown('<p style="font-size: 2.4rem; font-weight: 700; color: #222;">Multi-label Intent Classifier</p>', unsafe_allow_html=True)

# Labels
st.markdown("<h4 style='color: #444444; font-size: 1.1rem;'>Labels to be predicted:</h4>", unsafe_allow_html=True)
st.markdown('<div class="label-container">', unsafe_allow_html=True)

cols1 = st.columns(4)
label_data1 = [
    "Problem in recharge",
    "Problem in reward/redeem points", 
    "Problem in customer care service",
    "Problem in registration/login"
]

for col, label in zip(cols1, label_data1):
    with col:
        st.markdown(f'<div class="label-card">{label}</div>', unsafe_allow_html=True)

cols2 = st.columns(3)
label_data2 = [
    "Other complaints",
    "Appreciation",
    "Bad/Irrelevant comments"
]

for col, label in zip(cols2, label_data2):
    with col:
        st.markdown(f'<div class="label-card">{label}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input
# st.markdown("<h4 style='color: #444444; font-size: 1.1rem;'>Enter your text</h4>", unsafe_allow_html=True)
comment = st.text_area("", placeholder="Type your text here to analyze sentiment and intents...", 
                      height=80, key="user_input", label_visibility="collapsed")

analyze_btn = st.button("Analyze Text", key="analyze_button")

if analyze_btn and comment:
    with st.spinner('Analyzing your text...'):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("<h2 style='color: black;'>Classification Results</h2>", unsafe_allow_html=True)

        # Sentiment
        st.markdown("<h3 style='color: black;'>Sentiment Analysis</h3>", unsafe_allow_html=True)
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(comment)
        p = abs(vs['compound']) * 100
        p = np.round(p, 2)

        if vs['compound'] > 0:
            sentiment = "positive"
            sentiment_color = "#2a9d8f"
        elif vs['compound'] < 0:
            sentiment = "negative"
            sentiment_color = "#e63946"
        else:
            sentiment = "neutral"
            sentiment_color = "#457b9d"

        st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; color: black;">
                    <span style="font-size: 14px;">Sentiment: <strong>{sentiment.capitalize()}</strong></span>
                    <span style="font-size: 14px;">{p}% {sentiment}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {p}%; background-color: {sentiment_color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Intent classification
        st.markdown("<h3 style='color: black;'>Intent Classification</h3>", unsafe_allow_html=True)
        df1 = pd.DataFrame([[comment]], columns=['content'])
        df1_input_ids = tokenize_inputs(df1["content"].values, tokenizer, num_embeddings=250)
        df1_attention_masks = create_attn_masks(df1_input_ids)
        df1["features"] = df1_input_ids.tolist()
        df1["masks"] = df1_attention_masks
        pred_probs = generate_predictions(model, df1, num_labels, device="cpu", batch_size=1)
        pred_probs = np.round(pred_probs * 100, 2)
        probsList = [item for elem in pred_probs for item in elem]

        THRESHOLD = 50
        for label, prediction in zip(label_cols, probsList):
            if prediction >= THRESHOLD:
                bar_color = "#2a9d8f"
                st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; color: black;">
                            <span style="font-size: 14px;">{label}</span>
                            <span style="font-size: 14px;">{prediction}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {prediction}%; background-color: {bar_color};"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)


        st.markdown('</div>', unsafe_allow_html=True)

elif analyze_btn and not comment:
    st.warning("Please enter some text to analyze!")