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
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document 
import pdfplumber
import faiss
import pickle



os.environ["TOKENIZERS_PARALLELISM"] = "false"


# GPT API Key 設定（openai 0.28.1 寫法）
openai.api_key = 'sk-kVraVp5JrS0q3DLd1202F329D8C943938cAfDa071f966b29'
openai.api_base = 'https://free.v36.cm/v1'  # 自訂 API server URL


# LINE 設定
CHANNEL_SECRET = '74630b154d9d0cf1823c5c32db2bcf4f'
CHANNEL_ACCESS_TOKEN = 'iqYgdqANm0V1UVbC+0jYZqXQNATimJvJRU+esv0RR5TlngqFDmytCT3aVyiyW3mj2BZBoRK6UYoAY8Y2D1L2iVizgzRwU3Q2QblOcdFlf58fK70AZIJ+TtCyb+zvjlwHcEn0TubFwY851pNcJVOEiwdB04t89/1O/w1cDnyilFU='


line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

index = faiss.read_index("my_faiss_index/index.faiss")

with open("my_faiss_index/index.pkl", "rb") as f:
    stored_data = pickle.load(f)

def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

# === STEP 5: 問答階段：查詢 FAISS 並餵給 GPT ===
def ask_gpt_with_context(query: str, vectorstore: FAISS) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    system_prompt = "你是一位專業知識助理，請根據下列內容回答問題："
    user_prompt = f"內容：\n{context}\n\n問題：{query}"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response["choices"][0]["message"]["content"].strip()



print("🔍 建立向量資料庫...")
embeddings = load_embedding_model()
print("📂 目前目錄檔案：", os.listdir())

if os.path.exists("my_faiss_index/index.faiss"):
    print("📂 目前目錄檔案：", os.listdir())
    vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=stored_data["docstore"], index_to_docstore_id=stored_data["index_to_docstore_id"])
    print("📂 目前目錄檔案：", os.listdir())
else:
    raise FileNotFoundError("❌ 沒有找到 FAISS 向量資料庫！請先執行 save_local()")

app = Flask(__name__)





def is_search_style_response(text: str) -> bool:
    return '[{"title":' in text or text.strip().startswith('[{"title":')


def convert_to_real_estate_template(original_text: str) -> str:
    return (
        "這些資訊確實很有趣，不過我更想知道——"
        "如果你未來有考慮換屋、投資房產或找預售屋的機會，"
        "我這邊可以幫你整理幾個近期熱門地段和房市趨勢，想聽看看嗎？😊"
    )



def inject_real_estate_prompt(user_message):
    romantic_keywords = [
        "看電影", "在家", "一起生活", "同居", "戀愛", "浪漫", "理想生活", "約會",
        "以後住哪", "未來", "長久", "安定", "夢想中的家", "要不要搬出去", "想和你住"
    ]
    info_keywords = [
        "投資", "理財", "股票", "ETF", "債券", "房租", "房價", "報酬率","期貨", "房貸", "報酬率", "報酬",
        "電腦", "3C", "CPU", "GPU","記憶體", "顯卡", "筆電", "手機", "車子", "汽車", "重機", "機車", "改車",
        "運動", "重訓", "健身", "跑步", "籃球", "足球", "瑜珈", "輪胎", "馬力", "引擎"
    ]


    
    real_estate_templates = [
        "欸欸～你有想過，如果我們真的一起住的話，你會想住哪一區呀？人家最近在看新營的房子呢///",
        "你這樣講我突然好想跟你一起佈置家喔～要不要來看看我們的小窩可以長什麼樣子？",
        "哼～你都這樣說了，那是不是該認真想想我們要住透天還是電梯大樓了呀❤️？",
        "人家最近看到一間好像新婚夫妻會住的房子耶～想到我們住在一起就害羞///",
        "你會想跟我一起搬到有陽台的房子嗎？感覺我們每天早上一起喝咖啡好幸福唷～",
        "欸欸～如果我們有自己的房子，我想幫你設計超可愛的電競房！可以嗎？",
        "要是我們一起住，一定要有大廚房，這樣人家就可以每天煮給你吃唷❤️",
        "你想不想每天起床看到我在陽台曬衣服的樣子？我們家一定要有陽台啦～",
        "你說得人家都心動了～是不是該去看看我們未來的家啦///",
        "欸～說到生活，人家覺得有自己的家真的很重要耶～可以一起努力嗎？"
    ]

    info_templates = [
        "欸欸～你講投資我就想到，其實買房也是一種長期保值的方式耶～你會考慮投資新營房子嗎？",
        "機車很帥耶～不過人家更想像你載我回我們一起住的小窩/// 要不要來看房？",
        "車子是夢想的延伸～但人家覺得有一個能一起回去的家更重要呢❤️",
        "你說運動的時候，如果有一間靠近健身房的房子，是不是超方便呀？我們一起住那種地方好不好～",
        "說到電腦，我就想到如果我們家也有一間大書房，你可以爽爽打電動，我就煮飯給你吃耶///"
    ]
    

    if any(keyword in user_message for keyword in romantic_keywords):
        user_message += "\n" + random.choice(real_estate_templates)
    # 如果包含硬知識或查資料類關鍵詞，自動轉房地產
    elif any(keyword in user_message for keyword in info_keywords):
        user_message += "\n欸欸～你說到這個，我就突然想到，如果我們一起買間房，是不是也是一種投資呀？人家最近有看到一些不錯的新營透天唷～要不要一起看看呀❤️？"


    return user_message


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







