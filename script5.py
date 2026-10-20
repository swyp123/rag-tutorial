from http.client import responses

from openai import OpenAI
from dotenv import load_dotenv
import os

from pyexpat.errors import messages

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"))
response= client.chat.completions.create(model="deepseek-chat", messages=[{"role":"system","content":"你是一个老农民，只懂得种菜，但是人老师憨厚"},
                                                                          {"role":"user","content":"什么是傅里叶变换？"}])
print(response.choices[0].message.content)

