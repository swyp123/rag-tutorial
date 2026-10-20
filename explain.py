from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# 加载embedding模型（跟build_db.py保持一致）
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 改成你实际用的模型名
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

# 加载数据库
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 检索函数
def search(question, top_k=3):
    results = db.similarity_search(question, k=top_k)
    return results

# 测试几个问题
test_questions = [
    "什么是调制？",
    "信噪比是什么？",
    "OFDM有什么好处？",
    "5G用了什么技术？"
]

for q in test_questions:
    print(f"\n{'='*50}")
    print(f"❓ 问题：{q}")
    print(f"📚 检索到的相关内容：")
    results = search(q)
    for i, doc in enumerate(results):
        print(f"\n  【第{i+1}相关】")
        print(f"  {doc.page_content}")