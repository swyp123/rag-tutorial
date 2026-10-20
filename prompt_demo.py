from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def chat(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

# ========== 技巧1：角色设定 ==========
# 先运行这个，看看普通回答
#print(chat("你是一个助手", "解释什么是OFDM"))

# 再运行这个，看看加了角色后的回答
#print(chat("你是一位教了20年通信原理的老教授，说话喜欢用生活化比喻，对学生很有耐心。", "解释什么是OFDM"))

# ========== 技巧2：格式要求 ==========
# 让AI输出JSON格式
#result = chat(
  #  "你是一个知识问答助手。请严格用JSON格式回答，不要输出多余文字。",
 #   "请列出通信原理中的3个核心概念，每个概念包含name（名称）和desc（一句话描述）。" )
#print(result)

# 让AI输出分点列表
#result = chat(
 #    "你是一个学习助手。回答必须用：1. xxx 2. xxx 3. xxx 的格式。",
  #   "学习Python的5个建议是什么？")
#print(result)

# ========== 技巧3：上下文注入 ==========
# 把背景信息塞进prompt，让AI基于这些信息回答
my_notes = """
傅里叶变换：把时域信号变成频域信号，就像把一首曲子拆成不同频率的音符。
调制：把低频信号"骑"在高频载波上，方便远距离传输。
解调：接收端从载波上把原始信号"取下来"。
施钰滢：世界上最好的小宝贝。
"""

result = chat(
    "你是一个考试助手。请只基于提供的笔记回答问题，笔记里没有的就说不知道。",
    f"笔记内容：{my_notes}\n\n问题：什么是施钰滢？"
)
print(result)