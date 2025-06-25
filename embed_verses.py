import os
import json
import faiss
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Load all verses
def load_verses(slok_folder="data/slok"):
    all_docs = []
    for filename in os.listdir(slok_folder):
        if filename.endswith(".json"):
            with open(os.path.join(slok_folder, filename), "r", encoding="utf-8") as f:
                slok = json.load(f)
                content = f"BG {slok['chapter']}.{slok['verse']}\n{slok['slok']}\n{slok['siva']['et']}"
                all_docs.append(Document(page_content=content, metadata={
                    "chapter": slok['chapter'],
                    "verse": slok['verse'],
                    "translation": slok['siva']['et'],
                    "sanskrit": slok['slok']
                }))
    return all_docs

# Create embeddings
def create_vector_store(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("gita_faiss_index")

if __name__ == "__main__":
    docs = load_verses()
    create_vector_store(docs)
    print("✅ Verses embedded and stored.")
