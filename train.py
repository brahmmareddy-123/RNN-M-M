import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# Configuration
# -----------------------------------

DATASET = "corpus.txt"
MODEL = "model.keras"
TOKENIZER = "tokenizer.pkl"

# -----------------------------------
# Read Dataset
# -----------------------------------

with open(DATASET, "r", encoding="utf-8") as f:
    corpus = f.read().lower()

# -----------------------------------
# Tokenizer
# -----------------------------------

tokenizer = Tokenizer()

tokenizer.fit_on_texts([corpus])

total_words = len(tokenizer.word_index) + 1

print("Vocabulary Size :", total_words)

# -----------------------------------
# Create Sequences
# -----------------------------------

input_sequences = []

for line in corpus.split("\n"):

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):

        ngram = token_list[:i + 1]

        input_sequences.append(ngram)

# -----------------------------------
# Padding
# -----------------------------------

max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(

    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )

)

# -----------------------------------
# Split X and y
# -----------------------------------

X = input_sequences[:, :-1]

y = input_sequences[:, -1]

print("Input Shape :", X.shape)

print("Output Shape :", y.shape)

# -----------------------------------
# Model
# -----------------------------------

model = Sequential()

model.add(

    Embedding(

        input_dim=total_words,

        output_dim=64,

        input_length=max_sequence_len - 1

    )

)

model.add(

    SimpleRNN(

        128,

        activation="tanh"

    )

)

model.add(

    Dense(

        64,

        activation="relu"

    )

)

model.add(

    Dense(

        total_words,

        activation="softmax"

    )

)

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# -----------------------------------
# Train
# -----------------------------------

model.fit(

    X,

    y,

    epochs=300,

    batch_size=16,

    verbose=1

)

# -----------------------------------
# Save
# -----------------------------------

model.save(MODEL)

with open(TOKENIZER, "wb") as f:

    pickle.dump(tokenizer, f)

print("Training Completed Successfully!")