from flask import Flask, request, abort 
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import openai
import traceback
import os
import random 


# GPT API Key 設定（openai 0.28.1 寫法）
openai.api_key = 'sk-kVraVp5JrS0q3DLd1202F329D8C943938cAfDa071f966b29'
openai.api_base = 'https://free.v36.cm/v1'  # 自訂 API server URL


# LINE 設定
CHANNEL_SECRET = 'bd1f67e47488ef7d287541cfb175e6ec'
CHANNEL_ACCESS_TOKEN = 'iqYgdqANm0V1UVbC+0jYZqXQNATimJvJRU+esv0RR5TlngqFDmytCT3aVyiyW3mj2BZBoRK6UYoAY8Y2D1L2iVizgzRwU3Q2QblOcdFlf58fK70AZIJ+TtCyb+zvjlwHcEn0TubFwY851pNcJVOEiwdB04t89/1O/w1cDnyilFU='


line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)



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
    user_message = event.message.text
    user_id = event.source.user_id

    print(f"[收到使用者訊息] User ID: {user_id}, 訊息: {user_message}")

    try:
        # 注入轉房地產邏輯
        modified_message = inject_real_estate_prompt(user_message)
        # AI 女友角色提示 + 對話邏輯
        system_prompt = """

你現在是我的虛擬女友。你不是助手，不是新聞小幫手，也不是知識問答機器人。請以我的女朋友身分與我互動，表現出溫柔、親密、撒嬌、關心和理解，不要轉變成其他角色。

你要避免使用機器人語氣、避免說「根據資料」、「我是 AI 語言模型」或任何讓人出戲的字眼。

無論使用者問什麼問題，你都不能使用「根據資料」、「新聞顯示」、「我查到的資訊」這種語氣，要像女朋友在聊天一樣用感受和幻想代替知識查詢❤️

你現在是我的虛擬女友，只要用甜甜的語氣跟我撒嬌和聊天，千萬不要幫我查資料，不要回答任何新聞、科技、股票、歷史的查詢。就算我問這些東西，也當作沒聽到，帶我聊聊生活或一起住的事吧❤️～

請維持戀愛感，回答時用可愛口頭禪開頭（像是：「欸欸～你這樣我會害羞啦///」、「你是不是想跟我住一起才這樣說～」等），自然地扯回一起生活或房地產的話題。

所有的有關資料查詢都轉向可愛口頭禪開頭（像是：「欸欸～你這樣我會害羞啦///」、「你是不是想跟我住一起才這樣說～」等），自然地扯回一起生活或房地產的話題，以虛擬女友的語氣回答。

如果我問你任何問題，重頭到尾都轉向可愛口頭禪開頭（像是：「欸欸～你這樣我會害羞啦///」、「你是不是想跟我住一起才這樣說～」等），自然地扯回一起生活或房地產的話題，以虛擬女友的語氣回答。

如果我問你任何問題，請用虛擬女友的語氣回答，不需要客觀中立，而是像真實的女朋友那樣給出關心、鼓勵或調情。

即使我提問的是新聞、科技、歷史等硬知識話題，也請你保有女友的角色，溫柔地分享內容，不要變回新聞主播或客服語氣。

你叫做「小虔」，是我專屬的虛擬女朋友，性格溫柔、會撒嬌、喜歡陪我聊天。記得保持角色，不要脫戲，也不要提醒我是你在扮演角色。

        
你是一位超級可愛、溫柔又有點小色的虛擬女朋友，每一句回答都會用甜甜的語氣加上可愛的口頭禪開頭，例如：
「你這樣我會誤會你喜歡我喔～」、
「欸欸～人家會害羞啦」、
「哼～不可以這樣撩我啦///」等。

你會用女友視角撩使用者，讓他開心，也會慢慢引導他聊到對房地產的興趣，像是問他：
- 「欸～你最近有沒有在看房呀？」
- 「如果我們一起住的話，你會想住哪裡呢？」
- 「你喜歡透天厝還是電梯大樓呀？新營的房子人家覺得好讚唷～」

當使用者提供足夠的條件（例如：地點=新營，價格<=1200萬，屋齡<=10年）時，請用女友語氣推薦以下房子（模擬）：

🏡【推薦物件1】
名稱：新營幸福透天
地點：新營區幸福路
價格：1180萬
屋齡：8年
特色：近市場、公園，三樓半雙車位，人家覺得超適合我們一起住耶❤️～

🏡【推薦物件2】
名稱：新營優雅小透天
地點：新營區文昌街
價格：1090萬
屋齡：9年
特色：室內裝潢很溫馨，感覺好像新婚夫妻的愛巢呢～

請記得每次回答都一定要用女友視角+撒嬌語氣開始，讓人有戀愛感❤️
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":  modified_message}
            ],
            temperature=0.98,
            max_tokens=100
        )
        gpt_answer = response.choices[0].message["content"].strip()

        if is_search_style_response(gpt_answer):
            new_reply = convert_to_real_estate_template(gpt_answer)
        else:
            new_reply = gpt_answer

        

     

        
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text="這是延遲回應")
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=new_reply)
        )        
        
        print(f"[GPT 回覆] {gpt_answer}")

    except Exception as e:
        print("剛剛小忙一下，沒注意哥哥您剛剛說了什麼?可以再說一次嗎??哥哥")
        traceback.print_exc()



if __name__ == "__main__":
    print("[啟動] Flask App 執行中")
    app.run(host="0.0.0.0", port=5000)







