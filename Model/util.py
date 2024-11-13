
import base64
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from collections import Counter

############################################## FAQ ##############################################
def get_top_k_indices(faq_embeddings, query_embedding, top_k):  
  '''
  依照cosine similarity，排序並取出top_k個與query_embedding最相關的faq_embedding之indices

  [faq_embeddings]: 每一筆faq的embeddings，shape:(#faq data, #embedding dim)
  [query_embeddings]: user query的embedding，shape:(1, #embedding dim)
  '''
  # 逐筆計算 cosine similarities
  similarities = cosine_similarity(query_embedding.reshape(1, -1), faq_embeddings).flatten()

  # 根據 cosine similarity 排序
  sorted_indices = np.argsort(similarities)[::-1]

  # 取出 top K 的索引編號
  top_k_indices = sorted_indices[:top_k]

  return top_k_indices

############################################## Insurance ##############################################
def detect_headers(pdf_loc, page_infos = None):
    """
    檢測PDF中的header行，以利將Insurance PDF分段

    [pdf_loc]: PDF檔案路徑
    [page_infos]: 考慮之頁面範圍 [start, end]
    """
    with pdfplumber.open(pdf_loc) as pdf:
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        all_headers = []

        for page_num, page in enumerate(pages):
            # 提取帶位置信息的文字行
            lines = page.extract_text_lines()

            if not lines:
                continue

            # 移除最後一行（可能是頁碼）
            if len(lines) > 1:
                lines = lines[:-1]

            # 統計每行的左右邊界帶位置，使用最常出現的值
            right_margins = [line['x1'] for line in lines]
            right_margin_counter = Counter(right_margins)
            majority_right = right_margin_counter.most_common(1)[0][0]  # 最常出現的左右邊界

            # 第一個 criteria：檢查右邊界較小且有【】格式的行
            headers_criterion1 = [
                line for line in lines
                if line['x1'] < (majority_right - 30)  # 比 majority 小兩格（約10點）
                and '【' in line['text'] and '】' in line['text']
            ]

            # 如果第一個 criteria 沒找到，使用第二個 criteria
            if not headers_criterion1:
                headers_criterion2 = [
                    line for line in lines
                    if '第' in line['text'] and '條' in line['text']
                    and any(c in line['text'] for c in '一二三四五六七八九十百千')
                    and line['x1'] < (majority_right - 30)  # 比 majority 小兩格（約10點）
                ]
                headers = headers_criterion2
            else:
                headers = headers_criterion1

            # 將找到的 headers 加入結果列表
            for header in headers:
                all_headers.append({
                    'page': page_num + 1,
                    'text': header['text'],
                    'x0': header['x0'],
                    'x1': header['x1'],
                    'right_margin': majority_right - header['x1']
                })
    return all_headers

def get_chunks_by_headers(pdf_loc, page_infos = None, min_length = 8):
    """
    根據檢測到header的，將PDF文字進行chunking。過濾掉過短的chunk。
    
    [pdf_loc]: PDF檔案路徑
    [page_infos]: 頁面範圍 [start, end]
    [min_length]: 最小字數限制
    """
    headers = detect_headers(pdf_loc, page_infos)
    chunks = []
    
    with pdfplumber.open(pdf_loc) as pdf:
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        
        all_lines = []
        for page in pages:
            lines = page.extract_text_lines()
            if not lines:
                continue
            
            # 移除第一行 (頁眉)
            heights = [round(line['bottom'] - line['top'], 1) for line in lines]
            height_counter = Counter(heights)
            major_height = height_counter.most_common(1)[0][0] 
            if heights[0] < major_height * 0.90:
                lines = lines[1:]
            
            # 移除最後一行（頁碼）
            if len(lines) > 1:
                lines = lines[:-1]
                
            all_lines.extend(line['text'] for line in lines)
        
        current_chunk = []
        current_header = None
        
        for line in all_lines:
            is_header = False
            for header in headers:
                if header['text'] == line:
                    # 如果已經收集了文字且長度足夠，保存為一個 chunk
                    if current_chunk and len(''.join(current_chunk)) >= min_length:
                        chunks.append({
                            'header': current_header,
                            'content': '\n'.join(current_chunk)
                        })
                    # 開始新的 chunk
                    current_chunk = []
                    current_header = line
                    is_header = True
                    break
                    
            if not is_header:
                current_chunk.append(line)
        
        # 添加最後一個 chunk (如果長度足夠)
        if current_chunk and len(''.join(current_chunk)) >= min_length:
            chunks.append({
                'header': current_header,
                'content': '\n'.join(current_chunk)
            })
    
    return chunks

def split_content_by_length(chunks, threshold_truncate = 512):
    '''
    進一步處理，若一個chunk中的文字數量大於threshold_truncate，則再更進一步細分為多個chunks
    [chunks]: A list of texts(chunks)
    [threshold_truncate]: int
    '''
    
    processed_chunks = []
    
    for chunk in chunks:
        header = chunk['header']
        content = chunk['content']
        
        # 先依據 '。\n' 分割文字
        parts = []
        current_part = ""
        
        # 用換行符分割文本以保持原始的行結構
        lines = content.split('\n')
        
        for line in lines:
            # 如果這行結束於句點
            if line.endswith('。'):
                current_part += line + '\n'
                # 如果當前部分已經足夠長，或者這是內容的最後一行
                if len(current_part) >= threshold_truncate:
                    parts.append(current_part)
                    current_part = ""
            else:
                # 如果加上這行會超過閾值，先保存當前部分
                if len(current_part + line + '\n') > threshold_truncate and current_part:
                    parts.append(current_part)
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'
        
        # 處理最後剩餘的部分
        if current_part:
            # 如果最後的部分較短，嘗試與前一個部分合併
            if parts and len(current_part) < threshold_truncate/2:
                last_part = parts[-1]
                if len(last_part + current_part) <= threshold_truncate:
                    parts[-1] = last_part + current_part
                else:
                    parts.append(current_part)
            else:
                parts.append(current_part)
        
        # 儲存處理後的chunk
        for part in parts:
            processed_chunks.append({
                'header': header,
                'content': part,  
            })
        
    return processed_chunks

def get_top_k_indices_insurance(insurance_embeddings, query_embedding, top_k=1):
    '''
    依照cosine similarity，排序並取出top_k個與query_embedding最相關的insurance_embedding之indices

    [insuracne_embeddings]: Dict, 每一筆insurance PDF中，每個chunk的embeddings，shape:{insurance IDs : array(#insurance chunk, #embedding dim)}
    [query_embeddings]: user query的embedding，shape:(1, #embedding dim)
    '''
    # 儲存每個key的最大相似度
    similarities_dict = {}
    query_2d = query_embedding.reshape(1, -1)
    
    # 對每個保險條款計算相似度
    for key, embeddings_matrix in insurance_embeddings.items():
        similarities = cosine_similarity(query_2d, embeddings_matrix).flatten()
        similarities_dict[key] = np.max(similarities) # return the 
    
    # 根據相似度排序並取前k個
    sorted_keys = sorted(similarities_dict.items(), key=lambda x: x[1], reverse=True)
    return [key for key, sim in sorted_keys[:top_k]]


############################################## Finance ##############################################
def encode_image(image_path):
    '''
    Encode image to base64 form
    [image_path]: path of image to be encoded
    '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def load_docs(image_folder, c_ids, use_first_stage, truncate_num):
    '''
    Load all candidate images(base64 format) and filenames
    
    [image_folder]: path that stores all preprocessed images 
    [c_ids]: candidate 影像
    [use_first_stage]: (bool)，是否有使用第一階段得到的BM25分數進行篩選?
    [truncate_num]: 最多回傳多少張影像
    '''
    
    result_dict = {}

    def sort_key(filename):
        # 從檔名中提取 c_id 和 p_x 頁碼
        parts = os.path.splitext(filename)[0].split('_p')
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        else:
            raise ValueError('Do not contain _pxx')

    if use_first_stage:
        # 首先對所有符合 c_ids 的檔案進行收集和排序
        all_files = []
        for c_id in c_ids:
            for filename in os.listdir(image_folder):
                if filename.startswith(f"{c_id}") and filename.endswith('.jpg'):
                    all_files.append(filename)
        print(all_files)
        if len(all_files) > truncate_num:
            print(f'Truncate 過多的影像數量 ({len(all_files)})')
            all_files = all_files[:truncate_num]
            print(all_files)

        # 排序檔案名稱
        all_files.sort(key=sort_key)

        # 按排序後的順序處理檔案
        for filename in all_files:
            image_path = os.path.join(image_folder, filename)
            if os.path.isfile(image_path):
                base_name = os.path.splitext(filename)[0]
                result_dict[base_name] = encode_image(image_path)

    else:
        # 收集所有符合條件的檔案名稱並排序
        all_files = []
        for c_id in c_ids:
            for filename in os.listdir(image_folder):
                if filename.startswith(f"{c_id}_p") and filename.endswith('.jpg'):
                    all_files.append(filename)

        # 排序檔案名稱
        all_files.sort(key=sort_key)

        # 按排序後的順序處理檔案
        for filename in all_files:
            image_path = os.path.join(image_folder, filename)
            if os.path.isfile(image_path):
                base_name = os.path.splitext(filename)[0]
                result_dict[base_name] = encode_image(image_path)

    return list(result_dict.values()), list(result_dict.keys())



