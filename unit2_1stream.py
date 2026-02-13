import streamlit as st
import codecs
from llama_cpp import Llama

st.set_page_config(page_title="Ice Cream Chatbot", layout="centered")
st.title("Ice Cream Chatbot")
st.write("Ask questions based on the dataset")


@st.cache_resource
def load_embedding_model():
    embedding_model_path = (r"C:\Gen Ai and Prompt eng\Practical 2\bge-base-en-v1.5-q4_k_m.gguf")
    return Llama(
        model_path=embedding_model_path,
        embedding=True,
        verbose=False
    )

embed = load_embedding_model()


def get_embedding(text):
    return embed.create_embedding(text)["data"][0]["embedding"]


@st.cache_data
def load_dataset(path):
    with codecs.open(path, encoding="utf-8", errors="ignore") as f:
        return f.readlines()

dataset_path = "ice cream.txt"
dataset = load_dataset(dataset_path)


@st.cache_resource
def build_vector_db(dataset):
    vector_db = []
    for chunk in dataset:
        embedding = get_embedding(chunk)
        vector_db.append((chunk, embedding))
    return vector_db

vector_db = build_vector_db(dataset)


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
    query_embedding = get_embedding(query)

    similarities = []
    for chunk, embedding in vector_db:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


@st.cache_resource
def load_llm():
    llm_model_path = (r"C:\Gen Ai and Prompt eng\Practical 2\Llama-3.2-1B-Instruct-Q6_K.gguf")
    return Llama(
        model_path=llm_model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

llm = load_llm()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_query = st.text_input("Ask me a question about ice cream ")

if st.button("Ask") and user_query:
    retrieved_knowledge = retrieve(user_query)

    

    instruction_prompt = f"""
You are a helpful chatbot.
Use ONLY the information below to answer.
Do not make up new information.
{"".join([f"- {chunk}" for chunk, _ in retrieved_knowledge])}


Question: {user_query}

Answer:
"""
    output = llm(
        instruction_prompt,
        max_tokens=300,
        echo=False
    )

    response = output["choices"][0]["text"].strip()
    st.session_state.chat_history.append((user_query, response))

    for q , a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**bot:** {a}")