from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_URL"))
with open("knowledge.txt.txt", "r",encoding="utf-8") as f:
    knowledge = f.read()

question="世界上最可爱的小宝贝是谁"
prompt=f"""请基于以下参考资料回答用户的问题。
如果参考资料里没有答案，就说“我不知道”。

参考资料:{knowledge}

用户问题:{question}
回答:
"""
response=client.chat.completions.create(model="deepseek-chat",messages=[{"role":"user","content":prompt}])
print(response.choices[0].message.content)