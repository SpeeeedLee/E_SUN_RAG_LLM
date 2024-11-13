import os
import fitz  
import os
import glob
import pdfplumber
import subprocess
from pdf2image import convert_from_path
from tqdm import tqdm
from PIL import Image
import math


def split_pdf_by_pages(input_folder, output_dir):
    '''
    Split every PDF document to one-page format
    exK 123.pdf --> 123_p1.pdf, 123_p2.pdf, 123_p3.pdf...

    [input_folder]: folder that contains PDF documents
    [output_dir]: output folder for storing PDF documents after splitting
    '''
    os.makedirs(output_dir, exist_ok=True)

    # 遍歷輸入資料夾中的所有PDF文件
    for filename in os.listdir(input_folder):
        print(filename)
        if filename.endswith('.pdf'):
            input_pdf = os.path.join(input_folder, filename)

            # 取得原始PDF檔名（不含擴展名）
            base_name = os.path.splitext(filename)[0]

            pdf_document = fitz.open(input_pdf)
            for page_num in range(len(pdf_document)):
                pdf_writer = fitz.open()  # 新建一個空的PDF
                pdf_writer.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)  # 插入單頁

                # 使用原始檔名加上 _p? 的格式
                output_pdf = os.path.join(output_dir, f'{base_name}_p{page_num + 1}.pdf')
                pdf_writer.save(output_pdf)  # 儲存為新的PDF文件
                pdf_writer.close()

            pdf_document.close()

def preprocess_pdf(input_pdf, output_pdf, page_infos=None):
    '''
    對於掃描得到的pdf --> 進行OCR
    對於非掃描得到的pdf --> 移除頁面中的影像(很可能是印章，會影響模型判讀)
    
    [input_pdf]: path of pdf to preprocess
    [output_pdf]: path of pdf to save after preprocess
    '''
    # 首先打開pdfplumber以提取圖像信息
    with pdfplumber.open(input_pdf) as pdf:
        # 檢查每一頁
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        # for page_number in range(len(pdf.pages)):
        for page_number, page in enumerate(pages): 
            # page = pdf.pages[page_number]
            # 獲取頁面尺寸
            page_width = page.width
            page_height = page.height
            page_area = page_width * page_height
            # print(f"頁面尺寸: {page_width} x {page_height}")

            # 獲取頁面中的所有圖片
            images = page.images

            # 如果有某張圖片很大，代表這個PDF很可能是掃描得到的，不進行圖像刪除
            for img in images:
                img_width = img['x1'] - img['x0']  # 圖片的寬度
                img_height = img['y1'] - img['y0']  # 圖片的高度
                img_area = img_width * img_height

                # 計算圖片佔頁面面積的比例
                if img_area / page_area > 0.8:
                    print(f"頁面 {page_number + 1} 是一張大圖片，直接對其進行OCR")
                    # 使用subprocess呼叫ocrmypdf
                    subprocess.run(["ocrmypdf", "--force-ocr", "-l", "chi_tra", input_pdf, output_pdf], check=True)
                    return  # 退出函數，不進行圖像移除

            # 打開該頁面進行圖像移除
            with fitz.open(input_pdf) as doc:
                pdf_page = doc[page_number]

                # 移除頁面中的所有圖片
                for img_index in range(len(images)):
                    print(f"移除第{img_index + 1}張影像")
                    xref = pdf_page.get_images(full=True)[img_index][0]  # 獲取圖片的xref
                    pdf_page.delete_image(xref)

                # 保存修改後的PDF
                doc.save(output_pdf)

def process_all_pdfs(input_folder, output_folder):
    '''
    Calling preprocees_pdf function on all one-page finance PDF
    [input_pdf]: path of one-page pdf to preprocess
    [output_pdf]: path of pdf to save after preprocess
    '''
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 獲取所有PDF文件
    pdf_files = glob.glob(os.path.join(input_folder, '*.pdf'))

    for pdf_file in pdf_files:
        # 定義輸出的PDF文件名
        output_pdf = os.path.join(output_folder, os.path.basename(pdf_file))
        print(f"處理文件: {pdf_file} -> {output_pdf}")

        # 調用預處理函數
        preprocess_pdf(pdf_file, output_pdf)


def pdf_to_images(input_folder, output_folder):
    '''
    Conver one-page PDF to image(.jpeg)
    [input_pdf]: path of one-page pdf to
    [output_pdf]: path of images to save
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    poppler_path = r"C:\Users\arthu\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin" # change to your own poppler path here !

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
            pdf_name = os.path.splitext(filename)[0]

            output_image_path = os.path.join(output_folder, f"{pdf_name}.jpg")
            images[0].save(output_image_path, "JPEG")
        

def resize_image_with_constraints(img, max_dimension=1568, max_megapixels=1.15):
   # 獲取原始尺寸
   original_width, original_height = img.size
   
   # 計算哪一邊需要縮放到 1568
   width_ratio = max_dimension / original_width
   height_ratio = max_dimension / original_height
   
   # 選擇較小的縮放比例（較大的邊會剛好縮放到 1568）
   scale_ratio = min(width_ratio, height_ratio)
   
   # 如果圖片尺寸已經都小於 1568，保持原尺寸
   if scale_ratio >= 1:
       scale_ratio = 1
   
   # 計算新的尺寸，保持長寬比
   new_width = int(original_width * scale_ratio)
   new_height = int(original_height * scale_ratio)
   
   # 檢查是否超過 1.15 megapixels
   megapixels = (new_width * new_height) / 1_000_000
   if megapixels > max_megapixels:
       # 需要進一步縮小
       additional_scale = math.sqrt(max_megapixels / megapixels)
       new_width = int(new_width * additional_scale)
       new_height = int(new_height * additional_scale)
   
   return img.resize((new_width, new_height))

def resize_images_in_folder(input_folder, output_folder):
   # 確保輸出資料夾存在
   os.makedirs(output_folder, exist_ok=True)
   
   # 計數器
   processed = 0
   errors = 0
   
   # 遍歷輸入資料夾中的所有檔案
   for filename in os.listdir(input_folder):
       if filename.lower().endswith('.jpg'):
           input_path = os.path.join(input_folder, filename)
           output_path = os.path.join(output_folder, filename)
           
           try:
               # 開啟並處理圖片
               with Image.open(input_path) as img:
                   # 使用新的 resize 函數
                   resized_img = resize_image_with_constraints(img)
                   resized_img.save(output_path)
               
               # 印出處理資訊
               original_size = os.path.getsize(input_path) / 1024  # KB
               new_size = os.path.getsize(output_path) / 1024  # KB
               print(f"處理: {filename}")
               print(f"原始尺寸: {img.size}, {original_size:.1f}KB")
               print(f"調整後尺寸: {resized_img.size}, {new_size:.1f}KB")
               print("-" * 50)
               
               processed += 1
               
           except Exception as e:
               errors += 1
               print(f"處理 {filename} 時發生錯誤: {str(e)}")
   
   # 輸出處理結果
   print(f"\n處理完成:")
   print(f"成功處理: {processed} 張圖片")
   print(f"處理失敗: {errors} 張圖片")




# Step 1: Split every finance PDF to one-page
input_folder = '../reference/finance'
output_dir = '../reference/finance_split'
split_pdf_by_pages(input_folder, output_dir)

# Step 2: Preprocess every one-page PDF (OCR or Remove Stamp)
input_folder = '../reference/finance_split/'  
output_folder = '../reference/processed_finance/processed_finance_pdf'  
process_all_pdfs(input_folder, output_folder)

# Step 3: .pdf --> .jpg 
intput_folder = '../reference/processed_finance/processed_finance_pdf'  
output_folder = '../reference/processed_finance/processed_finance_image'
pdf_to_images(input_folder, output_folder)

# Step 4: Resize to reduce tokens needed when using Antropic API
input_folder = './reference/processed_finance/processed_finance_image'
output_folder = './reference/processed_finance/processed_finance_image_resize'
resize_images_in_folder(input_folder, output_folder)
