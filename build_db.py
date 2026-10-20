from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# 自定义Embedding
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

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


embeddings = MyEmbedding()

# 1. 读取知识库
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. 切分文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks = text_splitter.split_text(text)

print(f"✅ 切分成了 {len(chunks)} 个文本块")

# 3. 存入向量数据库
db = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ 向量数据库创建成功！保存在 chroma_db 文件夹")