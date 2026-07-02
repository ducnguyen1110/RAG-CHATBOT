import streamlit as st
# Import chính xác 2 hàm xử lý cốt lõi từ file main.py
from main import process_pdf, rag

# Cấu hình hiển thị giao diện rộng tràn màn hình chuyên nghiệp
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("PDF RAG Assistant: Native Pipeline")

# Khởi tạo bộ nhớ Session State theo cấu trúc mẫu y chang trong tài liệu
for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
    st.session_state.setdefault(k, v)

# Xây dựng thanh điều khiển Sidebar bên trái
with st.sidebar:
    st.subheader("Cấu hình mô hình")
    
    # Cải tiến chuyên sâu: Cho phép người dùng linh động chọn LLM và Embedding ngay trên giao diện
    llm_model = st.selectbox(
        "Chọn mô hình LLM", 
        ["vicuna:7b-v1.5-q5_1", "llama3.2:latest", "qwen2.5:3b", "gemma2:9b"]
    )
    embed_model = st.selectbox(
        "Chọn mô hình Embedding", 
        ["bge-m3", "nomic-embed-text"]
    )
    
    st.markdown("---")
    st.subheader("Upload tài liệu")
    f = st.file_uploader("Chọn file PDF", type="pdf")
    
    if f and st.button("Xử lý PDF", use_container_width=True):
        with st.spinner("Đang phân tích và vector hóa tài liệu..."):
            # Truyền mô hình embedding đã chọn vào hàm xử lý
            st.session_state.collection, n = process_pdf(f, embed_model=embed_model)
            st.session_state.pdf_name = f.name
            st.session_state.chat_history = [] # Reset lịch sử khi nạp tài liệu mới
            st.success(f"Đã xử lý xong: {n} chunks")
            
    st.info(f"Tài liệu hiện tại: {st.session_state.pdf_name}" if st.session_state.pdf_name else "Chưa có tài liệu")
    
    if st.button("Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []

# Hiển thị lịch sử chat động từ bộ nhớ session_state
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Kiểm soát trạng thái ô nhập tin nhắn dựa trên tài liệu nạp vào
if st.session_state.collection is None:
    st.info("Upload và xử lý PDF trước khi chat.")
    st.chat_input("Hộp chat đang bị khóa...", disabled=True)
else:
    q = st.chat_input("Nhập câu hỏi của bạn...")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
            
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                # Gọi hàm RAG kết hợp đầy đủ các cấu hình mô hình động từ giao diện
                ans = rag(
                    question=q, 
                    collection=st.session_state.collection, 
                    llm_model=llm_model, 
                    embed_model=embed_model
                )
                st.write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})