from flask import Flask, request, abort 
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import openai
import traceback
import os
import json
import random 
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document 
import pdfplumber






os.environ["TOKENIZERS_PARALLELISM"] = "false"


# GPT API Key 設定（openai 0.28.1 寫法）
openai.api_key = 'sk-kVraVp5JrS0q3DLd1202F329D8C943938cAfDa071f966b29'
openai.api_base = 'https://free.v36.cm/v1'  # 自訂 API server URL


# LINE 設定
CHANNEL_SECRET = '74630b154d9d0cf1823c5c32db2bcf4f'
CHANNEL_ACCESS_TOKEN = 'iqYgdqANm0V1UVbC+0jYZqXQNATimJvJRU+esv0RR5TlngqFDmytCT3aVyiyW3mj2BZBoRK6UYoAY8Y2D1L2iVizgzRwU3Q2QblOcdFlf58fK70AZIJ+TtCyb+zvjlwHcEn0TubFwY851pNcJVOEiwdB04t89/1O/w1cDnyilFU='


line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)


def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-multilingual")

# === STEP 2: 讀取 PDF 檔 ===
def load_documents(filepath: str):
    documents = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"page": i + 1}))
    return documents

# === STEP 3: 切割文件 ===
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            texts.append(Document(page_content=chunk, metadata=doc.metadata))
    return texts

# === STEP 4: 建立向量資料庫 ===
def create_vectorstore(chunks, embedding_model):
    return FAISS.from_documents(chunks, embedding_model)



# 步驟 5：使用者提問 → 相似內容 → 餵給 ChatGPT
def ask_gpt_with_context(query: str, vectorstore: FAISS) -> str:
    # 取得最相似的文件內容
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 建構 ChatGPT 提問內容
    system_prompt = "你是一個知識豐富的專業助理，根據以下內容回答使用者的問題。"
    user_prompt = f"以下是相關知識內容：\n\n{context}\n\n使用者問題：{query}"

    # 使用 ChatGPT（OpenAI 0.28.1 方式）
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.98,
        max_tokens=300,
    )
    return response["choices"][0]["message"]["content"].strip()




print("🔍 建立向量資料庫...")
embeddings = load_embedding_model()

print("📄 載入知識文件...")
docs = load_documents("00.pdf")

print("✂️ 分割文件...")
chunks = split_documents(docs)

print("🔍 建立向量資料庫...")
embeddings = load_embedding_model()
vectorstore = create_vectorstore(chunks, embeddings)

app = Flask(__name__)







@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    print(f"[Webhook 接收到訊息] Body:\n{body}")  # 印出訊息內容以確認

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("[簽章錯誤] Signature 無效")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_input = event.message.text
    user_id = event.source.user_id

    try:
        # 所有訊息都用向量資料庫查找內容 + GPT 回答
        reply = ask_gpt_with_context(user_input, vectorstore)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )

        print(f"[使用者 ID] {user_id}")
        print(f"[使用者提問] {user_input}")
        print(f"[AI 回答] {reply}")

    except Exception as e:
        print("⚠️ 錯誤發生：", e)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="抱歉～剛剛有點小狀況，哥哥可以再說一次嗎？")
        )



if __name__ == "__main__":
    print("[啟動] Flask App 執行中")
    app.run(host="0.0.0.0", port=5000)







