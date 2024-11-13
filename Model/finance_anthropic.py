import anthropic
import re
import json
from util import load_docs
from prompt_template import system_prompt


# Use Anthropic API to get the prediction results
def extract_output_format_regex(text):
    pattern = r'<output_format>(.*?)</output_format>'
    match = re.search(pattern, text, re.DOTALL)  
    if match:
        return match.group(1).strip()
    else:
        return text

image_folder = './reference/processed_finance/processed_finance_image_resize'
use_first_stage = True
bm25_threshold = 0.30
first_stage_path = './preliminary_test/bm25_rewrite.json'


# Load input questions
print('Loading QUERY')
question_path = './preliminary_test/questions_preliminary.json'
with open(question_path, 'rb') as f:
    qs_ref = json.load(f) 

# Answer dict for storing answer
answer_dict = {"answers": []}
output_path = './preliminary_test/pred/finance.json'

client = anthropic.Anthropic(
    api_key= "put your Anthropic API key here"
    )

print('Loading first stage results')
with open(first_stage_path, 'rb') as f:
    first_stage_retrieved = json.load(f)


for q_dict in qs_ref['questions']:
    if q_dict['category'] != 'finance':
        continue
    print(f'============回答第{q_dict['qid']}題============')
    user_query = q_dict['query']

    c_ids = None
    for answer in first_stage_retrieved["answers"]:
        if answer["qid"] == q_dict['qid']:
            '''
            Thresholding: 使用BM25之分數，只保留分數大於 "排名第二影像之BM25分數* bm25_threshold"
            ''' 
            threshold = bm25_threshold * answer["scores"][1]
            top_count = sum(score > threshold for score in answer["scores"])
            top_count = min(top_count, len(answer["retrieve"]))

            c_ids = answer["retrieve"][:top_count]
            str_c_ids = [str(c_id) for c_id in c_ids]
            print("選取的 c_ids:", str_c_ids)
            break
    
    base64_images, filenames = load_docs(image_folder, str_c_ids, use_first_stage, truncate_num=12) # 最多只給模型top-12影像，避免過多資訊
    print(f"sorted filenams: {filenames}")

    '''生成user prompt'''
    user_content = []
    for idx, (base64_img, filename) in enumerate(zip(base64_images, filenames), 1):
        # 影像編號
        user_content.append({
            "type": "text",
            "text": f"Image {idx}:"
        })
        # 圖片內容
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64_img
            }
        })

    # 使用者查詢(query)
    user_content.append({
        "type": "text",
        "text": f"query: {user_query}"
    })

    '''Call API'''
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2046,
        temperature=0,
        system=system_prompt, # 放入定義好的 system prompt
        messages=[
            {
                "role": "user",
                "content": user_content
            }
        ]
    )

    text_content = message.content[0].text
    print(text_content)

    '''後處理模型輸出'''
    output_format = extract_output_format_regex(text_content)
    try:
        output_dict = json.loads(output_format)
        top_1_value = output_dict["top 1"]  # 獲取 "8"
        print(f"最相關的影像是：{top_1_value}")
    except Exception as e:
        print(f"發生錯誤：{str(e)}")
        top_1_value = output_format

    # map回原本的影像編號
    try:
        # 因為模型返回的編號是從1開始，但列表索引是從0開始，所以要減1
        index = int(top_1_value) - 1
        top_1_value_map = filenames[index]
        print(f"對應的檔案名稱是：{top_1_value_map}")
    except (ValueError, IndexError):
        print(f"無法找到對應的檔案名稱，模型返回值：{top_1_value}")
        top_1_value_map = top_1_value

    '''將答案將入answer_dict，並儲存成json檔案'''
    answer_dict['answers'].append({
        "qid": q_dict['qid'], 
        "retrieve": top_1_value_map
    })
    # 立即更新 JSON 文件
    with open(output_path, 'w', encoding='utf8') as f:
                    json.dump(answer_dict, f, ensure_ascii=False, indent=4) 

