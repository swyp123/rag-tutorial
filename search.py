from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client_emb = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

class MyEmbedding:
    def embed_query(self, text):
        res = client_emb.embeddings.create(
            input=text,
            model="BAAI/bge-large-zh-v1.5"
        )
        return res.data[0].embedding

embeddings = MyEmbedding()

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

test_questions = [
    "什么是调制？",
    "信噪比是什么？",
    "OFDM有什么好处？",
    "5G用了什么技术？"
]

for q in test_questions:
    print(f"\n{'='*50}")
    print(f"❓ 问题：{q}")
    results = db.similarity_search(q, k=2)
    for i, doc in enumerate(results):
        print(f"\n  【第{i+1}相关】")
        print(f"  {doc.page_content}")