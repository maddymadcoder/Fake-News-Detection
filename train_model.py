import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # Fake
true["label"] = 1   # Real

# Combine & shuffle
data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# Text cleaning
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    return text.lower()

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully")
