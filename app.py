import streamlit as st
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv
import os


# ==================== 初始化（只运行一次） ====================
# Streamlit 有个特点：每次用户交互都会重新运行整个脚本
# 所以用 @st.cache_resource 让这些初始化只做一次

@st.cache_resource
def init_everything():
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

    embeddings = MyEmbedding()

    # 加载向量数据库
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # 大模型客户端
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    return db, client


db, client = init_everything()

SYSTEM_PROMPT = """你是一个知识问答助手。请严格基于提供的参考资料回答用户的问题。
规则：
1. 只能使用参考资料中的信息，不能编造
2. 如果参考资料里没有答案，直接说"抱歉，我的知识库中没有相关信息"
3. 回答简洁清晰，用自己的话总结
4. 结合对话历史理解用户的问题
5. 如果不确定就说不确定不要猜"""


# ==================== RAG 核心逻辑 ====================

def rag_answer(question, history, top_k=2):
    # 检索
    docs = db.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 构造 messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)  # 历史对话
    messages.append({
        "role": "user",
        "content": f"参考资料：\n{context}\n\n用户问题：{question}"
    })

    # 调用大模型
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=messages,
        temperature=0.3
    )

    answer = response.choices[0].message.content
    return answer, docs


# ==================== 页面布局 ====================

# 页面标题
st.set_page_config(page_title="RAG 知识问答助手", page_icon="📚")
st.title("📚 通信原理知识问答助手")
st.caption("基于 RAG 技术，用你的知识库回答问题")

# 侧边栏：设置
with st.sidebar:
    st.header("⚙️ 设置")
    top_k = st.slider("检索段落数", min_value=1, max_value=5, value=2)
    temperature = st.slider("回答温度", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    max_history = st.slider("保留历史轮数", min_value=1, max_value=10, value=5)

    st.divider()
    if st.button("🗑️ 清空对话", type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("💡 提示：")
    st.caption("- 检索段落越多，信息越全，但可能引入干扰")
    st.caption("- 温度越低，回答越保守稳定")

# 初始化对话历史（用 st.session_state 保存状态）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            # 只传历史对话（不含系统prompt和参考资料，纯对话历史）
            answer, sources = rag_answer(
                prompt,
                st.session_state.messages[-max_history * 2:],  # 只保留最近N轮
                top_k=top_k
            )
            st.markdown(answer)

            # 展开显示参考来源
            with st.expander(f"📖 参考了 {len(sources)} 段内容"):
                for i, doc in enumerate(sources):
                    st.caption(f"**[{i + 1}]** {doc.page_content[:100]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})