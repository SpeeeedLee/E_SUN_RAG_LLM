import json
from tqdm import tqdm
import pdfplumber
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from Model.util import get_chunks_by_headers, split_content_by_length, get_top_k_indices_insurance

cache_dir= './Model/cache' # repo for storing HuggingFace Model

'''Load the embedding model'''
model_name = 'intfloat/multilingual-e5-large'
print(model_name)
model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder=cache_dir, device='cuda:1')

'''Load the Insurance PDFs'''
def load_data(source_path):
    '''
    Load all PDF, return a dict
    [source_path]: A path that stores all Insurnace PDFs
    '''
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict
    
def read_pdf(pdf_loc, page_infos: list = None):
    '''
    讀取單個PDF，對其中文字進行Chunking
    [pdf_loc]: PDF路徑
    [page_infos]: 考慮的PDF頁面
    '''
    chunks = get_chunks_by_headers(pdf_loc, page_infos, 8)
    processed_chunks = split_content_by_length(chunks, 512)
    merged_texts = [(chunk['header'] + chunk['content']).replace('\n', '') if chunk['header'] is not None else chunk['content'].replace('\n', '') 
                for chunk in processed_chunks]    
    return merged_texts

print('Loading Insurance PDF')  
source_path_insurance = './reference/insurance'  
processed_insurance_texts = load_data(source_path_insurance)


'''Embed Insurance Text'''
insurance_embeddings = {}
for doc_id, texts in processed_insurance_texts.items():
    texts = ["passage:" + text for text in texts]
    insurance_embeddings[doc_id] = model.encode(texts)

'''Embed Query'''
query_path = '../dataset/preliminary/questions_preliminary.json'
print('Loading QUERY')
with open(query_path, 'rb') as f:
    query_ref = json.load(f)

'''For all query, get most relavant Chunk in embedding space'''
# Use suggested candidate docs
answer_dict = {"answers": []}
for q_dict in query_ref['questions']:
    if q_dict['category'] == 'finance':
        continue
    elif q_dict['category'] == 'insurance':
        # Embed the query 
        query_embedding = model.encode('query: ' + q_dict['query'])
        candidate_insurance_embeddings = {doc_id: insurance_embeddings[str(doc_id)] for doc_id in q_dict['source']}
        real_retrieved_indexes = get_top_k_indices_insurance(candidate_insurance_embeddings, query_embedding, 1)
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": real_retrieved_indexes[0]})
        print(f'qid : {q_dict["qid"]}, retrieved index : {real_retrieved_indexes[0]}')
    elif q_dict['category'] == 'faq':
        continue

'''Store the answer to json file'''
output_path = './preliminary_test/pred/insurance.json'
with open(output_path, 'w', encoding='utf8') as f:
    json.dump(answer_dict, f, ensure_ascii=False, indent=4)