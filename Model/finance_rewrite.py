import json

'''Step 1'''
question_path = './preliminary_test/questions_preliminary.json'
print('Loading QUERY')
with open(question_path, 'rb') as f:
    qs_ref = json.load(f) 

user_queries = {}
for q_dict in qs_ref['questions']:
    if q_dict['category'] == 'finance':
        user_queries[q_dict["qid"]] = q_dict["query"]


output_path = './preliminary_test/finance_query.json'
with open(output_path, 'w', encoding='utf8') as f:
    json.dump(user_queries, f, ensure_ascii=False, indent=4)  


'''Step 2、Rewrite to increase BM25 performance'''
### 使用免費之 Antropic web 介面 (https://claude.ai/new)，輸入以下prompt，並且將'./preliminary_test/finance_query.json'之結果附於{user query}中
### 一次丟入不超過50個query，並使用Claude 3.5 Sonnet，rewrite的結果較好
'''
你是一個RAG系統中的rewriter。你需要rewrite一組user query，以利後續的retriever檢索可以更精準。每組user query會包含一個唯一的識別碼 (qid) 和查詢內容 (user query)。具體來說，你需要嚴格地遵循以下規則:

刪去user query中的公司名稱
刪去無關緊要的stop words :「在、?、多少、之、請問、中、的」
將user query中的西元年份改成用「民國xxx年」描述
若遇到user query中存在「第x季」，請在user query後加上幾個關鍵字，包含「第x季」在財務報表的表格中常會出現的日期。例如「第3季」，則加入「7月1日」、「9月30日」
使用者輸入的格式為字典結構，例如：
{ "1": "和泰車在2022年第3季度的基本每股盈餘(虧損)是多少？", "2": "台積電在2021年營收有多少？"} 

請依照上述規則處理每個user query，並返回以下格式的結果：
{"1": "rewrite結果1", "2": "rewrite_query": "rewrite結果2"} 

現在開始，{user query}:
'''
### Copy and paste the rewrite results in ./preliminary_test/finance_query_rewrite.json
