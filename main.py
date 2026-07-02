import pypdf
import chromadb
import ollama



def chunk_text(text, size=1000, overlap=200): 
    paras =  [p.strip() for p in text.split("\n") if p.strip()]
    chunks,cur = [],""
    for p in paras:
        while len(p) > size:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.append(p[:size].strip())
            p = p[size-overlap:]
        if len(cur) + len(p) + 1 <= size:
            cur +=p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p + "\n") if overlap else (p+"\n")
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
chunks = chunk_text(full_text)
print("Số chunks:", len(chunks))
print(chunks[0][:300])



# đoạn code thực hiện embed vô database collection
def embed(texts): 
    return ollama.embed(model ="bge-m3",input= texts)["embeddings"]
client = chromadb.Client()
collection = client.get_or_create_collection("rag")
# Thêm tất cả chunks vào database
collection.add(
            ids=[str(i) for i in range(len(chunks))], # ID duy nhất cho mỗi chunk
documents=chunks, #noi dung text gốc
embeddings=embed(chunks), #vector tương ứng
)
print("Đã index:", collection.count(), "chunks")

        
def retrieve(query, k = 4):
    "Tìm k đoạn văn bản liên quan nhất với câu hỏi"
    res = collection.query(
        query_embeddings = embed([query]),
        n_results = k
    )
    return res["documents"][0]
QUERY = "YOLOv10 dùng để làm gì?"
for doc in retrieve(QUERY):
    print(doc[:200])
    print("-" * 40)

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời: """

def rag(question, collection, k=4):
    res = collection.query(
        query_embeddings=embed([question]), 
        n_results=k
    )
    
    context = "\n\n".join(res["documents"][0])
    
    resp = ollama.chat(
        model="vicuna:7b-v1.5-q5_1",
        messages=[{
            "role": "user", 
            "content": PROMPT.format(context=context, question=question)
        }],
        options={"temperature": 0}
    )
    
    return resp["message"]["content"]
print(rag("YOLOv10 là gì?"))

def process_pdf(uploaded_file, embed_model="bge-m3"):
    """Đọc file PDF từ Streamlit, cắt nhỏ và đẩy vào Vector Database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    os.unlink(path) # Xóa file tạm lập tức để tránh chiếm dung lượng ổ cứng
    
    chunks = chunk_text(text)
    
    # Cải tiến: Sử dụng PersistentClient để lưu cơ sở dữ liệu lâu dài xuống ổ cứng thay vì in-memory
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Tạo tên bộ sưu tập duy nhất dựa trên mốc thời gian thực tránh trùng lặp
    col = client.get_or_create_collection(f"rag_{int(time.time())}")
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks, embed_model=embed_model)
    )
    return col, len(chunks)