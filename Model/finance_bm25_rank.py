import os
import json
import jieba 
import pdfplumber
from rank_bm25 import BM25Okapi



def load_data(source_path, candidate_ids):
    '''
    獲取所有candidate PDF的文字，返回字典
    [source_path]: 儲存所有PDF檔案的資料夾
    [candidate_ids]: 可能與user query相關的所有檔案名稱
    '''
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    candidate_ids_set = {str(id) for id in candidate_ids}

    # 過濾符合條件的檔案
    filtered_files = [
        file for file in masked_file_ls
        if file.endswith('.pdf') and any(file.startswith(f"{id}_") for id in candidate_ids_set)
    ]
    print(f'Filetered PDFs : {filtered_files}')
    # 讀取每個符合條件的PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    corpus_dict = {
        os.path.splitext(file)[0]: read_pdf(os.path.join(source_path, file))
        for file in filtered_files
    }
    return corpus_dict

def read_pdf(pdf_loc, page_infos = None):
    '''
    讀取PDF中的文字
    [pdf_loc]: pdf檔案路徑
    [page_infos]: 限制讀取某幾頁，無則設置為None
    '''
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    print('==================pdf_text==================')
    print(f'{pdf_text[:10]}....')
    pdf.close()  # 關閉PDF文件

    return pdf_text 



def BM25_retrieve(qs, corpus_dict):
    '''
    返回BM25排序

    [qs] : 使用者query, 
    [corpus_dict] : 存所有candidate PDF 文字的dict
    '''

    filtered_corpus = list(corpus_dict.values()) 

    # 將每篇文檔進行分詞
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    
    # 獲取所有文檔的BM25得分
    scores = bm25.get_scores(tokenized_query)
    
    # 將得分與檔案名對應起來
    scored_docs = [(key, score) for key, score in zip(corpus_dict.keys(), scores)]
    
    # 根據得分排序，從高到低
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # 提取排序後的檔案名，並過濾掉得分為0的檔案
    sorted_docs_filtered = [(key, round(score,4)) for key, score in sorted_docs]

    # 返回檔案名和BM25分數的排序
    sorted_file_names, sorted_scores = zip(*sorted_docs_filtered) if sorted_docs_filtered else ([], [])

    return sorted_file_names, sorted_scores

if __name__ == "__main__":
    question_path = './preliminary_test/questions_preliminary.json'
    rewrite_question_path = './preliminary_test/finance_query_rewrite.json'
    source_path_finance = './reference/processed_finance/processed_finance_pdf'
    output_path = './preliminary_test/bm25_rewrite.json'

    answer_dict = {"answers": []}

    # Load question json to dict
    print('Loading QUERY')
    with open(question_path, 'rb') as f:
        qs_ref = json.load(f)
        
    print('Loading rewrite QUERY')
    with open(rewrite_question_path, 'rb') as f:
        qs_rewrite = json.load(f)  
        qs_rewrite = {int(key): value for key, value in qs_rewrite.items()}
    
    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            print(f'Ansewering Question {q_dict['qid']} : {q_dict['query']}')
            
            # 獲取cnadidate pdf
            candidate_ids = q_dict['source']
            print(f'Loading candidate PDFs...{candidate_ids}')
            corpus_dict_finance = load_data(source_path_finance, candidate_ids)

            # 用rewrite query進行檢索
            retrieved, scores = BM25_retrieve(qs_rewrite[q_dict['qid']], corpus_dict_finance)
            print(f'Retrieved...{retrieved}')
            print(f'Scores...{scores}')
            
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved, "scores": scores})


    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)