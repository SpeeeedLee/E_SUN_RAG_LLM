import json
from sentence_transformers import SentenceTransformer
from Model.util import get_top_k_indices

cache_dir= './Model/cache' # repo for storing HuggingFace Model

'''Load the embedding model'''
model_name = 'intfloat/multilingual-e5-large'
print(model_name)
model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder=cache_dir, device='cuda:1')


'''Preprocess FAQ data'''
faq_path = '../reference/faq/pid_map_content.json'
print('Loading FAQ json')
with open(faq_path, 'rb') as f_s:
    key_to_source_dict = json.load(f_s)
    faq_dict = {}
    for key, qas in key_to_source_dict.items():
        faq_key = int(key)
        content_parts = []
        for qa in qas:
            # 把 answers 列表合併成一個字串
            answer_text = ' '.join(qa['answers'])
            content_parts.append(f"question: {qa['question']} answer: {answer_text}")
        faq_dict[faq_key] = ' '.join(content_parts)

'''Embed FAQ data'''
faq_texts = ["passage:" + text for text in faq_dict.values()]
faq_embeddings = model.encode(faq_texts)


'''Load user QUERY'''
query_path = '.preliminary_test/questions_preliminary.json'
## Load question json to dict
print('Loading QUERY')
with open(query_path, 'rb') as f:
    query_ref = json.load(f)

'''For all query, get most relavant FAQ in embedding space'''
answer_dict = {"answers": []}
for q_dict in query_ref['questions']:
    if q_dict['category'] == 'finance':
        continue
    elif q_dict['category'] == 'insurance':
        continue
    elif q_dict['category'] == 'faq':
        # Embed the query 
        query_embedding = model.encode('query: ' + q_dict['query'])
        candidate_faq_embeddings = faq_embeddings[q_dict['source']]
        retrieved_indexes = get_top_k_indices(candidate_faq_embeddings, query_embedding, 1)
        real_retrieved_indexes = [q_dict['source'][i] for i in retrieved_indexes]
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": real_retrieved_indexes})
        print(f'qid : {q_dict["qid"]}, retrieved index : {real_retrieved_indexes}')


'''Store the answer to json file'''
output_path = './preliminary_test/pred/faq.json'
with open(output_path, 'w', encoding='utf8') as f:
    json.dump(answer_dict, f, ensure_ascii=False, indent=4)