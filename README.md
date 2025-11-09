# 🔤 Character-Level LSTM Name Generator (Language-Conditioned)

This project trains a **character-level LSTM model** to generate realistic names across multiple languages.  
It learns from per-language name lists, builds character and language embeddings, and predicts names using a next-character generation approach.

---

## ✨ Key Features
✅ Language-conditioned name generation  
✅ Character-level Word2Vec embeddings  
✅ LSTM-based next-character prediction  
✅ Temperature-based sampling for diversity  
✅ Automatic start/end token handling  
✅ Generates multiple names per language  

---

## 🧠 Model Pipeline
1) Load per-language name lists  
2) Clean & tokenize characters  
3) Build vocab + embeddings  
4) Prepare next-char supervised sequences  
5) Train language-conditioned LSTM  
6) Generate names with `<s> ... </s>` tokens  

---

## 🏗 Architecture
- Character input → Embedding  
- Language ID → Embedding  
- Concatenation → LSTM → Dense → Softmax  
- Predicts next character until end token or max length  

---

## 📦 Installation

git clone https://github.com/<your-username>/char-lstm-name-generator
cd char-lstm-name-generator
pip install -r requirements.txt

## 🔮 Generate Names

After training, the script will automatically print generated names per language.
To customize generation:

Change NAMES_TO_GENERATE

Adjust TEMPERATURE

Switch target language

## 📁 Output

Console: generated names per language

Saved model:

char_lstm_name_generator_optimized.h5

## 📈 Improvements

Beam search decoding

Transformer-based generation

GUI / web demo

Add phonetic constraints

## 📄 License

MIT
