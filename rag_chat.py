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

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

SYSTEM_PROMPT = """你是一个知识问答助手。请严格基于提供的参考资料回答用户的问题。
规则：
1. 只能使用参考资料中的信息，不能编造
2. 如果参考资料里没有答案，直接说"抱歉，我的知识库中没有相关信息"
3. 回答简洁清晰，用自己的话总结
4. 结合对话历史理解用户的问题
5. 如果不确定就说不确定不要猜"""


class RAGChat:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def ask(self, question, top_k=2):
        docs = db.similarity_search(question, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)

        user_prompt = f"参考资料：\n{context}\n\n用户问题：{question}"
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=messages,
            temperature=0.3
        )

        answer = response.choices[0].message.content

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return answer, docs

    def clear_history(self):
        self.history = []
        print("🔄 对话历史已清空")


if __name__ == "__main__":
    print("=" * 50)
    print("📚 RAG 多轮对话助手")
    print("   输入 clear 清空历史，输入 exit 退出")
    print("=" * 50)

    chat = RAGChat(max_history=5)

    while True:
        question = input("\n你：")

        if question.lower() in ["exit", "退出", "quit"]:
            print("👋 再见！")
            break

        if question.lower() in ["clear", "清空"]:
            chat.clear_history()
            continue

        answer, sources = chat.ask(question)
        print(f"\n🤖 AI：{answer}")
        print(f"   📖 参考了 {len(sources)} 段内容")