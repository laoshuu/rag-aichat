from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

samples = {
    "Prose Paragraph (英文散文)": (
        "Tokenization is the foundational step in natural language processing, "
        "converting raw text into numerical sequences that large language models can understand."
    ),
    "Code Snippet (程式碼)": (
        "def calculate_ratio(text: str, ids: list[int]) -> float:\n"
        "    return len(text) / len(ids) if ids else 0.0"
    ),
    "Markdown Table (Markdown 表格)": (
        "| Model | Parameters | Context |\n"
        "| :--- | :---: | ---: |\n"
        "| Qwen2.5 | 7B | 128K |"
    ),
    "Chinese Sentence (中文句子)": (
        "人工智慧正在快速改變軟體開發的工作流程，讓工程師能更專注於系統架構與邏輯設計。"
    ),
}

# 3. 逐一進行編碼並印出統計數據與切分結果
def print_token_stats():
    """encode each text and print tokenization results and statistics"""
    for label, text in samples.items():
        enc = tokenizer.encode(text)
        
        num_tokens = len(enc.ids)
        chars_per_token = len(text) / num_tokens
        
        print(f"=== {label} ===")
        print(f"Token 數量 (len(enc.ids))          : {num_tokens}")
        print(f"平均每 Token 字元數 (chars/token)  : {chars_per_token:.2f}")
        print(f"切分結果 (enc.tokens)              : {enc.tokens}")
        print()

if __name__ == "__main__":
    print_token_stats()