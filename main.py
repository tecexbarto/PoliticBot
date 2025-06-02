#--------------------IMPORTED LIBRARIES-----------------------------

import os

os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf'
os.environ['HF_HOME'] = '/tmp/hf'

import streamlit as st
import base64
import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import httpx
from huggingface_hub import hf_hub_download

# ---------------------- INITIAL CONFIGURATION ----------------------

st.set_page_config(page_title="PoliticBot", layout="wide")

with open("fondo.jpeg", "rb") as f:
    img_bytes = f.read()
    encoded_img = base64.b64encode(img_bytes).decode()

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: scroll;
    }}
    section[data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 50, 0.6);
        color: white;
    }}
    h1, h2, h3, h4, h5, h6, p, label, div, span {{
        color: white !important;
    }}
    textarea {{
        color: white !important;
        background-color: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid #ccc !important;
        border-radius: 8px !important;
        padding: 0.5em !important;
        min-height: 100px !important;
        transition: none !important;
    }}
    ::placeholder {{
        color: #ccc !important;
    }}
    pre, code {{
        background-color: rgba(0, 0, 0, 0.4) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5em !important;
    }}
    div[data-testid="stExpander"] {{
        transition: none !important;
    }}
    * {{
        transition: none !important;
        animation: none !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
        background-color: #526366 !important;
        color: white !important;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6em;
        width: 80% !important;
        margin-bottom: 0.5em;
    }}
    div[data-testid="stButton"] > button {{
        background-color: #526366 !important;
        color: white !important;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6em;
        margin-top: 1em;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- LIBRARIES AND MODELS ----------------------

ideology_families = ["Communism", "Liberalism", "Conservatism", "Fascism", "Radical_Left"]

ideology_keywords = {
    "Communism": ["communism", "marxism", "marxist", "anarcho-communism", "leninism"],
    "Liberalism": ["liberalism", "libertarianism", "classical liberal"],
    "Conservatism": ["conservatism", "traditional conservatism", "neoconservatism"],
    "Fascism": ["fascism", "nazism", "national socialism"],
    "Radical_Left": ["radical left", "far-left", "revolutionary socialism", "anarchism"]
}

@st.cache_resource
def load_encoder():
    model_name = "intfloat/e5-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

tokenizer, model = load_encoder()

def mean_pooling(output, mask):
    token_embeddings = output.last_hidden_state
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def embed_query(query):
    prefixed = f"query: {query}"
    inputs = tokenizer(prefixed, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = mean_pooling(outputs, inputs["attention_mask"])
    return F.normalize(pooled, p=2, dim=1).cpu().numpy().astype("float32")

@st.cache_resource
def load_data_global():
    chunks_path = hf_hub_download(repo_id="Bartix84/politicbot-data", filename="chunks.jsonl", repo_type="dataset")
    index_path = hf_hub_download(repo_id="Bartix84/politicbot-data", filename="faiss_index.index", repo_type="dataset")
    metadata_path = hf_hub_download(repo_id="Bartix84/politicbot-data", filename="metadata_titles.json", repo_type="dataset")

    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    return index, metadata, chunks

def search_in_global_index(query_embedding, index, metadata, chunks, selected_ideology, k=5):
    _, indices = index.search(query_embedding, k * 8)
    results = []
    keywords = ideology_keywords.get(selected_ideology, [])
    seen_titles = set()

    for i in range(indices.shape[1]):
        idx = indices[0][i]
        title = metadata[idx]
        if title in seen_titles:
            continue
        seen_titles.add(title)
        match = next((chunk for chunk in chunks if chunk["title"] == title), None)
        if match:
            title_text = title.lower()
            if any(keyword in title_text for keyword in keywords):
                results.append(match)
        if len(results) >= k:
            break
    return results

def generate_rag_response(ideology, user_query, context_chunks):
    context = "\n\n".join(chunk["chunk"] for chunk in context_chunks)[:1500]

    system_prompt = f"You are a political assistant who thinks and reasons like a {ideology} thinker."

    user_prompt = f"""
    Answer the following political or ethical question based strictly on the CONTEXT provided.
    Think according to the principles and values of {ideology}. If the context is insufficient, clearly say so or explain its limitations.
    Avoid always starting your answer the same way. Vary the introduction while staying formal and ideologically grounded.
CONTEXT:
{context}
QUESTION:
{user_query}
ANSWER:"""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "❌ Error: Missing `OPENROUTER_API_KEY` in Hugging Face Space secrets."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://yourappname.streamlit.app",
        "X-Title": "PoliticBot"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.9,
        "max_tokens": 768,
        "top_p": 0.95
    }

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
    except httpx.RequestError as e:
        return f"❌ Connection error: {e}"
    except httpx.HTTPStatusError as e:
        return f"❌ API error: {e.response.status_code} - {e.response.text}"

    return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------- STREAMLIT INTERFACE ----------------------

st.image('portada3.jpg', use_container_width=True)
st.title('🗳️ PoliticBot')
st.subheader('Reasoning with political ideologies')

with st.sidebar:
    st.header("Choose a political ideology")

    if "selected_ideology" not in st.session_state:
        st.session_state.selected_ideology = None

    for ideology in ideology_families:
        if st.button(ideology):
            st.session_state.selected_ideology = ideology

selected_ideology = st.session_state.selected_ideology

if selected_ideology:
    st.write(f"You have selected: **{selected_ideology}**")

user_query = st.text_area("Write your question or political dilemma:", height=100, key="user_input")

if selected_ideology and st.button("Send question"):
    if user_query.strip() == "":
        st.warning("Write a question before continuing.")
    else:
        with st.spinner("Thinking like that ideology..."):
            query_emb = embed_query(user_query + " in the context of " + selected_ideology)
            index, metadata, chunks = load_data_global()
            context = search_in_global_index(query_emb, index, metadata, chunks, selected_ideology, k=5)
            response = generate_rag_response(selected_ideology, user_query, context)

            st.session_state.response = response
            st.session_state.context = context
            st.session_state.last_query = user_query  

if "response" in st.session_state and st.session_state.selected_ideology:
    st.subheader("🤖 Generated response:")
    st.markdown(f"> {st.session_state.response}")

    with st.expander("🌐 Display the context used"):
        context = st.session_state.context
        if not context:
            st.markdown("*No relevant context found.*")
        else:
            for chunk in context:
                st.markdown(f"**{chunk['title']}**")
                st.code(chunk["chunk"][:500] + "...")