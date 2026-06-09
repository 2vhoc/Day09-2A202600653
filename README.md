# Student Scaffold

`src/` là bản để sinh viên hoàn thiện.

Giữ sẵn:
- provider abstraction trong `src/provider/`
- embedding loader thật trong `src/rag/embeddings.py`
- config và state cơ bản

Sinh viên cần tự làm:
- parse policy markdown
- build Chroma index
- viết tools lookup
- viết supervisor routing
- viết graph orchestration bằng LangGraph
- chạy test với `data/test.json`
