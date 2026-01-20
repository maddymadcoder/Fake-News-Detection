import streamlit as st
import pickle
import re
from groq import Groq

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    return text.lower()

def llama_explanation(news, result):
    prompt = f"""
    You are an AI fake news analyst.
    The article is classified as {result}.
    Explain clearly why this news may be fake or real.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

st.title("📰 Fake News Detection System")

news_input = st.text_area("Paste News Article Here", height=250)

if st.button("Analyze News"):
    if news_input.strip() == "":
        st.warning("Please enter news text.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 0:
            st.error("❌ Fake News Detected")
            result = "FAKE"
        else:
            st.success("✅ Real News Detected")
            result = "REAL"

        with st.spinner("Analyzing with LLaMA..."):
            explanation = llama_explanation(news_input, result)

        st.subheader("AI Explanation")
        st.write(explanation)
