import pypdf
import chromadb
import ollama
import tempfile
import os
import time

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời: """

def embed(texts, embed_model="bge-m3"): 
    """Chuyển danh sách chuỗi text thành danh sách vector."""
    return ollama.embed(model=embed_model, input=texts)["embeddings"]

def chunk_text(text, size=1000, overlap=200): 
    """Cắt nhỏ văn bản thành các chunk có độ dài tối đa với vùng đệm overlap."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.append(p[:size].strip())
            p = p[size - overlap:]
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def process_pdf(uploaded_file, embed_model="bge-m3"):
    """Đọc file PDF từ Streamlit, cắt nhỏ và đẩy vào Vector Database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    os.unlink(path) # Xóa file tạm lập tức để tránh chiếm dung lượng ổ cứng
    
    chunks = chunk_text(text)
    
    # Sử dụng PersistentClient để lưu cơ sở dữ liệu lâu dài xuống ổ cứng thay vì bị xóa mất khi rerun
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Tạo tên bộ sưu tập duy nhất dựa trên mốc thời gian thực tránh trùng lặp
    col = client.get_or_create_collection(f"rag_{int(time.time())}")
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks, embed_model=embed_model)
    )
    return col, len(chunks)

def rag(question, collection, llm_model="vicuna:7b-v1.5-q5_1", embed_model="bge-m3", k=4):
    """Thực hiện tìm kiếm ngữ cảnh liên quan và chuyển tiếp cho LLM sinh câu trả lời."""
    res = collection.query(
        query_embeddings=embed([question], embed_model=embed_model), 
        n_results=k
    )
    
    context = "\n\n".join(res["documents"][0])
    
    resp = ollama.chat(
        model=llm_model,
        messages=[{
            "role": "user", 
            "content": PROMPT.format(context=context, question=question)
        }],
        options={"temperature": 0}
    )
    
    return resp["message"]["content"]