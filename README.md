# 環境建置
### 建立conda虛擬環境
```bash
conda create -n your_env_name python=3.8.20
```
### 啟動虛擬環境並安裝必要套件
```bash
conda activate your_env_name
pip install -r requirements.txt
```
[注意] 程式流程繁雜，涵蓋資料預處理及使用HuggingFace模型、Antropic API生成預測。若按照requirement.txt安裝後，仍然遇到套件依賴問題，請依照抱錯安裝或調整套件版本。


# 資料前處理

## 文件存放
### 請將PDF檔案先存放至
* **Finance: `./reference/finance`**
* **Insurance: `./reference/insurance`**

### 前處理
```bash
cd Preprocess
python3 finance.py # 將PDF分割成one-page，進行OCR，轉成影像(.jpg)
```




# 生成預測
```bash
cd ../
```

## Finance
```bash
python3 Model/finance_rewrite.py # 共有兩步驟，Step2部分需使用Antropic網頁介面。請依照指示執行 
python3 Model/finance_bm25_rank.py # 使用 BM25 進行第一階段排序
python3 Model/finance_anthropic.py # 使用 Anthropic API 進行第二階段reranking
```
預測結果將儲存於: ```./preliminary_test/pred/finance.json```


## Insurance
```bash
python3 Model/insurance.py # 使用開源embedding model，進行預測
```
預測結果將儲存於: ```./preliminary_test/pred/insurance.json```

## Faq
```bash
python3 Model/faq.py # 使用開源embedding model，進行預測
```
預測結果將儲存於: ```./preliminary_test/pred/faq.json```