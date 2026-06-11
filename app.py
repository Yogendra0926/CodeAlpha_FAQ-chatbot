import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
st.set_page_config(
    page_title="VIPEY FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

data = pd.read_csv("faqs.csv")
questions = data["Question"]
answers = data["Answer"]
stop_words = set(stopwords.words("english"))
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_words = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            filtered_words.append(word)
    return " ".join(filtered_words)
processed_questions = questions.apply(preprocess)
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)
def get_answer(user_question):
    processed_input = preprocess(user_question)
    user_vector = vectorizer.transform([processed_input])
    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )
    best_match = similarity.argmax()
    score = similarity[0][best_match]
    if score > 0.30:
        return answers[best_match], score
    return "Sorry, I couldn't find answer.", score
st.markdown("""
<style>
.chat-title{
    text-align:center;
    color:#4CAF50;
    font-size:40px;
    font-weight:bold;
}
.user-msg{
    background:#DCF8C6;
    padding:12px;
    border-radius:10px;
    margin:5px;
}
.bot-msg{
    background:#F1F0F0;
    padding:12px;
    border-radius:10px;
    margin:5px;
}
</style>
""", unsafe_allow_html=True)
st.markdown(
    "<h1 class='chat-title'>🤖 VIPEY FAQ Chatbot</h1>",
    unsafe_allow_html=True
)
st.write("Ask any question from the FAQ database.")
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_inp = st.chat_input(
    "Type your question..."
)
if user_inp:
    st.session_state.messages.append({
        "role":"user",
        "content":user_inp
    })
    with st.chat_message("user"):
        st.write(user_inp)
    answer, score = get_answer(user_inp)
    bot_reply = f"""
{answer}
🔍 Confidence Score: {score:.2f}
"""
    st.session_state.messages.append({
        "role":"assistant",
        "content":bot_reply
    })
    with st.chat_message("assistant"):
        st.write(bot_reply)