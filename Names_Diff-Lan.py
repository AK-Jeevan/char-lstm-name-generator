# This script trains a character-level LSTM language-conditioned name generator.
# It reads name lists per language, builds a character vocabulary and embeddings,
# Train a LSTM to predict next characters, and generates new names.

import os                                  # interact with the filesystem
import random                              # basic randomness control
import numpy as np                         # numerical arrays and utilities
import nltk                                # minimal NLP utilities (tokenizers, corpora)
from gensim.models import Word2Vec        # train small char2vec embeddings
import tensorflow as tf                    # deep learning framework
from tensorflow import keras               # high-level Keras API
from keras import layers, models, callbacks  # model building blocks and callbacks

# ----------------------------
# Download small NLTK resources if not already available (required once)
# ----------------------------
''' nltk.download('punkt', quiet=True)         # tokenizer models
nltk.download('wordnet', quiet=True)       # lemmatizer corpora (not used heavily here)
nltk.download('omw-1.4', quiet=True)       # optional multilingual resources
'''

# ----------------------------
# Configuration / hyperparameters
# ----------------------------
DATA_DIR = r"C:\Users\akjee\Documents\AI\NLP\NLP - DL\LSTM-RNN\names\names"  # folder with per-language .txt files
MIN_NAME_LEN = 1               # minimum name length to keep (characters, after cleaning)
MAX_NAME_LEN = 30              # maximum name length to keep (characters)
TEST_SPLIT = 0.1               # fraction of examples held for validation
RANDOM_SEED = 42               # reproducible randomness seed

EMBEDDING_SIZE = 64            # dimensionality for char embeddings (and language embedding)
LSTM_UNITS = 128               # LSTM hidden units
BATCH_SIZE = 64                # training batch size
EPOCHS = 10                    # training epochs

TEMPERATURE = 0.8              # sampling temperature when generating names
MAX_GENERATE_LEN = 20          # max characters to generate (stops at end token)
NAMES_TO_GENERATE = 8          # how many names to generate per language after training

# Fix random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ----------------------------
# Utilities: load and preprocess files
# ----------------------------
def load_language_files(data_dir):
    """
    Read all .txt files in data_dir. Each file should contain one name per line.
    Returns dict mapping language (filename without extension) -> list of names.
    """
    lang_to_names = {}
    for fname in os.listdir(data_dir):                   # iterate files in directory
        if not fname.lower().endswith(".txt"):           # skip non-txt files
            continue
        lang = os.path.splitext(fname)[0]                # language id from filename
        path = os.path.join(data_dir, fname)             # full path
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            names = [line.strip() for line in f if line.strip()]  # read non-empty lines
        if names:
            lang_to_names[lang] = names                  # store list under language key
    return lang_to_names

def clean_name(name):
    """
    Minimal cleaning for a name:
    - strip whitespace
    - lower-case for normalization
    - collapse multiple spaces
    Keeps Unicode letters intact.
    """
    name = name.strip().lower()
    name = " ".join(name.split())
    return name

def tokenize_chars(name):
    """
    Tokenize a name at the character level.
    We expect start/end tokens to be added by caller.
    """
    return list(name)

def preprocess(lang_to_names):
    """
    Convert raw names to token lists and attach start/end tokens.
    Returns a list of (language, token_list) and a sorted list of language names.
    """
    data = []
    languages = sorted(lang_to_names.keys())
    for lang in languages:
        for raw in lang_to_names[lang]:
            s = clean_name(raw)                            # clean name text
            if len(s) < MIN_NAME_LEN or len(s) > MAX_NAME_LEN:
                continue                                   # skip names outside length bounds
            s = "<s>" + s + "</s>"                         # add explicit start/end tokens
            tokens = tokenize_chars(s)                     # split into characters
            data.append((lang, tokens))                    # store pair
    return data, languages

# ----------------------------
# Vocab and embeddings
# ----------------------------
def build_char_vocab(data):
    """
    Build character vocabulary from tokenized data.
    Adds special tokens <pad> and <unk> at the front.
    Returns list of all chars, mapping char->id and id->char.
    """
    chars = set()
    for _, tokens in data:
        chars.update(tokens)
    # Ensure special tokens are included
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]    # pad id 0, unk id 1
    all_chars = special_tokens + sorted(c for c in chars if c not in special_tokens)
    char_to_id = {ch: i for i, ch in enumerate(all_chars)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}
    return all_chars, char_to_id, id_to_char

def build_lang_vocab(languages):
    """
    Map language names to integer ids and return reverse mapping.
    """
    lang_to_id = {lang: i for i, lang in enumerate(languages)}
    id_to_lang = {i: lang for lang, i in lang_to_id.items()}
    return lang_to_id, id_to_lang

def train_char_word2vec(data, embedding_size=EMBEDDING_SIZE):
    """
    Train a small Word2Vec model on character sequences.
    This provides initial character vectors used as the embedding matrix.
    """
    sentences = [tokens for _, tokens in data]            # list of token lists (characters)
    model = Word2Vec(sentences=sentences,
                     vector_size=embedding_size,
                     window=5,
                     min_count=1,
                     workers=1,
                     sg=1,                # skip-gram (works well for small corpora)
                     epochs=20)
    return model

def build_embedding_matrix(w2v, all_chars, char_to_id, embedding_size=EMBEDDING_SIZE):
    """
    Create an embedding matrix of shape (vocab_size, embedding_size).
    If a character appears in the Word2Vec model, use that vector; otherwise random-init.
    """
    mat = np.random.normal(scale=0.01, size=(len(all_chars), embedding_size)).astype(np.float32)
    for ch in all_chars:
        idx = char_to_id[ch]
        if ch in w2v.wv:
            mat[idx] = w2v.wv[ch]                         # replace random vector with trained vector
    return mat

# ----------------------------
# Prepare supervised sequences for next-character prediction
# ----------------------------
def prepare_sequences(data, char_to_id, lang_to_id, max_len=None):
    """
    For each tokenized name (with start token), create many training examples:
    - inputs: previous characters (padded to max_len) and language id
    - target: next character id
    This converts sequence generation into a supervised next-character prediction task.
    """
    if max_len is None:
        max_len = max(len(tokens) for _, tokens in data)  # longest sequence length
    pad = char_to_id["<pad>"]
    Xc, Xl, y = [], [], []
    for lang, tokens in data:
        lang_id = lang_to_id[lang]
        ids = [char_to_id.get(t, char_to_id["<unk>"]) for t in tokens]  # convert tokens->ids
        for i in range(1, len(ids)):                      # for each position predict ids[i] from ids[:i]
            prev = ids[:i]                                # previous character ids
            target = ids[i]                               # next character id to predict
            padded = prev + [pad] * (max_len - len(prev)) # right-pad to fixed length
            Xc.append(padded)                             # char input (padded)
            Xl.append(lang_id)                            # language id input
            y.append(target)                              # target char id
    # return arrays and the max sequence length used
    return (np.array(Xc, dtype=np.int32),
            np.array(Xl, dtype=np.int32),
            np.array(y, dtype=np.int32),
            max_len)

def train_test_split(Xc, Xl, y, test_split=TEST_SPLIT):
    """
    Simple randomized train/validation split for the prepared example arrays.
    """
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    ts = int(n * test_split)
    test_idx = idx[:ts]
    train_idx = idx[ts:]
    return Xc[train_idx], Xl[train_idx], y[train_idx], Xc[test_idx], Xl[test_idx], y[test_idx]

# ----------------------------
# Model definition
# ----------------------------
def build_model(vocab_size, embedding_size, embedding_matrix, num_langs, lstm_units, max_len):
    """
    Build a model that takes:
      - char_in: (max_len,) sequence of char ids (padded)
      - lang_in: scalar language id
    The model embeds chars and language, concatenates them, processes with an LSTM,
    and predicts the next character using a softmax over the char vocabulary.
    """
    # character input (padded sequence of ids)
    char_in = layers.Input(shape=(max_len,), name="char_in")
    # pre-initialized embedding layer for characters (trainable)
    emb = layers.Embedding(input_dim=vocab_size, output_dim=embedding_size,
                           weights=[embedding_matrix], input_length=max_len,
                           trainable=True, name="char_emb")(char_in)
    # language id input (single int)
    lang_in = layers.Input(shape=(), dtype="int32", name="lang_in")
    # learn a small embedding for the language id
    lang_emb = layers.Embedding(input_dim=num_langs, output_dim=embedding_size, name="lang_emb")(lang_in)
    # replicate language embedding across sequence length so it can be concatenated with char embeddings
    lang_rep = layers.RepeatVector(max_len)(lang_emb)
    # concatenate character and language embeddings along feature axis
    x = layers.Concatenate()([emb, lang_rep])
    # LSTM processes the concatenated sequence and outputs a single vector
    x = layers.LSTM(lstm_units, name="lstm")(x)
    # a small dense bottleneck
    x = layers.Dense(128, activation="relu")(x)
    # final softmax over character vocabulary to predict next char id
    out = layers.Dense(vocab_size, activation="softmax", name="next_char")(x)
    model = models.Model([char_in, lang_in], out)     # instantiate the Keras functional model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ----------------------------
# Sampling / generation utilities
# ----------------------------
def sample_with_temperature(probs, temperature=1.0):
    """
    Sample an index from a probability distribution after applying temperature scaling.
    Lower temperature -> more greedy; higher temperature -> more random.
    """
    p = np.asarray(probs).astype("float64")
    # avoid log(0)
    p = np.log(p + 1e-8) / max(temperature, 1e-6)
    p = np.exp(p)
    p = p / np.sum(p)
    return np.random.choice(len(p), p=p)

def generate_name(model, char_to_id, id_to_char, lang_to_id, language, max_len,
                  temperature=TEMPERATURE, max_generate_len=MAX_GENERATE_LEN):
    """
    Generate a single name for a given language using the trained model.
    Uses <s> and </s> tokens to mark start and end; stops when end token predicted.
    """
    if language not in lang_to_id:
        raise ValueError("Unknown language")
    pad = char_to_id["<pad>"]
    start = char_to_id.get("<s>")
    end = char_to_id.get("</s>")
    if start is None or end is None:
        raise RuntimeError("Start/end tokens missing in vocab")
    seq = [start]                                       # initial sequence begins with start token
    for _ in range(max_generate_len):
        padded = seq + [pad] * (max_len - len(seq))     # pad current sequence to max_len
        # model expects batch inputs; provide single example arrays
        pred = model.predict({"char_in": np.array([padded]), "lang_in": np.array([lang_to_id[language]])}, verbose=0)[0]
        nxt = sample_with_temperature(pred, temperature)  # sample next char id
        seq.append(int(nxt))
        if nxt == end:                                   # stop if end token produced
            break
        if len(seq) >= max_len:                          # safety: don't exceed max_len
            break
    # convert ids to characters and remove special tokens
    chars = [id_to_char[i] for i in seq if i in id_to_char]
    name = "".join(ch for ch in chars if ch not in ["<s>", "</s>", "<pad>", "<unk>"]).strip()
    return name.capitalize() if name else ""             # capitalize first letter for readability

# ----------------------------
# Main training and generation flow
# ----------------------------
def main():
    # load name lists per-language
    lang_to_names = load_language_files(DATA_DIR)
    if not lang_to_names:
        raise RuntimeError(f"No .txt files in {DATA_DIR}")   # guard if data directory empty
    print("Languages:", list(lang_to_names.keys()))

    # preprocess into (lang, token_list) pairs
    data, languages = preprocess(lang_to_names)
    print("Total names (after cleaning):", len(data))

    # build character and language vocabularies
    all_chars, char_to_id, id_to_char = build_char_vocab(data)
    lang_to_id, id_to_lang = build_lang_vocab(languages)
    print("Vocab size (chars):", len(all_chars), "Languages:", len(languages))

    # train lightweight character Word2Vec to initialize embeddings
    w2v = train_char_word2vec(data, embedding_size=EMBEDDING_SIZE)
    emb_mat = build_embedding_matrix(w2v, all_chars, char_to_id, embedding_size=EMBEDDING_SIZE)

    # prepare supervised training examples
    Xc, Xl, y, max_len = prepare_sequences(data, char_to_id, lang_to_id)
    Xc_tr, Xl_tr, y_tr, Xc_val, Xl_val, y_val = train_test_split(Xc, Xl, y, TEST_SPLIT)
    print("Train examples:", len(y_tr), "Val examples:", len(y_val), "Max sequence length:", max_len)

    # build and summarize the model
    model = build_model(len(all_chars), EMBEDDING_SIZE, emb_mat, len(languages), LSTM_UNITS, max_len)
    model.summary()

    # early stopping to avoid overfitting; restore best weights
    es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # train the model; inputs are (char sequence padded, language id) and target is next char id
    model.fit({"char_in": Xc_tr, "lang_in": Xl_tr}, y_tr,
              validation_data=({"char_in": Xc_val, "lang_in": Xl_val}, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)

    # generate sample names for each language
    for lang in languages:
        print(f"\nGenerated names for {lang}:")
        for _ in range(NAMES_TO_GENERATE):
            print(" -", generate_name(model, char_to_id, id_to_char, lang_to_id, lang, max_len))

    # save the trained model to disk for later use
    model.save("char_lstm_name_generator_optimized.h5")
    print("Model saved as char_lstm_name_generator_optimized.h5")

# Run main when executed as a script
if __name__ == "__main__":
    main()