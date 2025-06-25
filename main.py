import streamlit as st
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ✅ Configure Gemini SDK
genai.configure(api_key=GOOGLE_API_KEY)

# Load vector store created earlier
@st.cache_resource
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return FAISS.load_local("gita_faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vector_store()

# Streamlit UI
st.title("🧘‍♂️ Gita Life Guide")
st.write("Reflect on your challenges through the wisdom of the Bhagavad Gita.")

user_input = st.text_area("🗣️ What's troubling you right now?", height=150)

if st.button("🪷 Seek Gita Wisdom"):
    if not user_input.strip():
        st.warning("Please enter something first.")
    else:
        with st.spinner("Searching for Gita's response..."):
            # 🔍 Get top matching verse using embeddings
            results = vectorstore.similarity_search(user_input, k=1)
            verse_doc = results[0]
            meta = verse_doc.metadata

            # 🪔 Display the verse
            st.markdown(f"### 📖 Verse: BG {meta['chapter']}.{meta['verse']}")
            st.markdown(f"**Sanskrit**: \n\n{meta['sanskrit']}")
            st.markdown(f"**Translation**: _{meta['translation']}_")

            # ✨ Generate reflection using Gemini 2.5 (Vertex API)
            prompt = f"""You are a spiritual counselor helping someone with life issues.

Problem: "{user_input}"

Below is a Bhagavad Gita verse translation:
"{meta['translation']}"

Based on this verse, provide an insightful and compassionate reflection, staying true to the Gita's philosophy."""

            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)

            # 🧘 Show Gemini’s reflection
            st.markdown("### 🧠 Gita’s Reflection:")
            st.write(response.text)
