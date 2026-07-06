import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------
# Configuration
# ------------------------------------

MODEL = "model.keras"
TOKENIZER = "tokenizer.pkl"

# ------------------------------------
# Load Model
# ------------------------------------

model = load_model(MODEL)

with open(TOKENIZER, "rb") as f:
    tokenizer = pickle.load(f)

# Reverse Dictionary
index_word = {}

for word, index in tokenizer.word_index.items():
    index_word[index] = word

max_sequence_len = model.input_shape[1] + 1


# ------------------------------------
# Predict Next Word
# ------------------------------------

def predict_next_word(text):

    text = text.lower()

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(prediction)

    return index_word.get(predicted_index, "Unknown")


# ------------------------------------
# Streamlit UI
# ------------------------------------

st.set_page_config(
    page_title="Many-to-Many RNN",
    page_icon="🧠"
)

st.title("🧠 Next Word Prediction using Simple RNN")

st.write(
    "Enter a sentence and let the RNN predict the next word."
)

sentence = st.text_input(
    "Enter Sentence",
    placeholder="Example : machine learning"
)

if st.button("Predict"):

    if sentence.strip() == "":
        st.warning("Please enter a sentence.")

    else:

        next_word = predict_next_word(sentence)

        st.success(f"Predicted Next Word : {next_word}")