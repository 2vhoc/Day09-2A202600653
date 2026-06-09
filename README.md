

# 🛒 Multi-Agent Shopping Assistant
## Log test: src/artifacts
> Trợ lý mua sắm AI sử dụng kiến trúc đa tác tử (Multi-Agent) với LangGraph, RAG và LLM.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-00C853)
![License](https://img.shields.io/badge/License-MIT-blue)

## ✨ Tổng quan

Shopping Assistant là hệ thống trả lời câu hỏi khách hàng về **đơn hàng**, **voucher**, và **chính sách** mua sắm. Hệ thống sử dụng kiến trúc **Supervisor – Worker** với 4 agent phối hợp xử lý:

```
User Question
     │
     ▼
┌─────────────┐
│  Supervisor  │  ← Route câu hỏi đến worker phù hợp
└──────┬──────┘
       │
  ┌────┴────┐
  ▼         ▼
┌──────┐ ┌──────┐
│Policy│ │ Data │  ← RAG search policy / Lookup order, customer, voucher
│Worker│ │Worker│
└──┬───┘ └──┬───┘
   │        │
   └───┬────┘
       ▼
┌─────────────┐
│  Response    │  ← Tổng hợp + LLM viết câu trả lời tự nhiên
│  Agent       │
└─────────────┘
```

## 🚀 Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| **Multi-Agent Routing** | Supervisor tự động route đến Policy, Data, hoặc cả hai |
| **RAG Policy Search** | Tìm kiếm chính sách bằng semantic search (ChromaDB + sentence-transformers) |
| **Data Lookup Tools** | 4 tools tra cứu đơn hàng, khách hàng, voucher |
| **Chatbot UI** | Giao diện chat dark-mode premium với glassmorphism & animations |
| **Natural Language** | LLM viết lại câu trả lời thành giọng văn tự nhiên, thân thiện |
| **Clarification** | Tự động hỏi lại khi thiếu thông tin (order_id, customer_id) |
| **Batch Testing** | Chạy test hàng loạt với `data/test.json` |

## 📁 Cấu trúc dự án

```
├── app/
│   ├── graph.py          # LangGraph multi-agent orchestration
│   ├── web.py            # Web server + Chatbot UI
│   ├── data_access.py    # Data lookup tools (order, customer, voucher)
│   ├── config.py         # Settings & environment config
│   ├── state.py          # LangGraph state schema
│   ├── prompts.py        # System prompts cho các agent
│   ├── cli.py            # Command-line interface
│   └── utils.py          # Utility functions
├── provider/
│   ├── __init__.py       # LLM provider router
│   ├── gemini.py         # Google Gemini
│   ├── openai.py         # OpenAI
│   ├── openrouter.py     # OpenRouter
│   ├── ollama.py         # Ollama (local)
│   └── custom.py         # Custom LLM endpoint
├── rag/
│   ├── embeddings.py     # Sentence-transformers embeddings
│   ├── parser.py         # Policy markdown parser
│   └── vector_store.py   # ChromaDB vector store
├── requirements.txt
└── README.md
```

## ⚙️ Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/2vhoc/Day09-2A202600653.git
cd Day09-2A202600653
```

### 2. Tạo virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Cài dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình `.env`

Tạo file `.env` ở **thư mục gốc của repo cha** (ngang hàng với `data/`):

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your-api-key-here
```

Hoặc dùng OpenAI/Custom:

```env
LLM_PROVIDER=custom
CUSTOM_LLM_MODEL=deepseek-v4-flash
CUSTOM_LLM_BASE_URL=https://your-endpoint/v1
CUSTOM_LLM_API_KEY=your-key
```

## 🖥️ Chạy ứng dụng

### Web UI (Chatbot)

```bash
python -m app.web --port 8000
```

Mở trình duyệt tại **http://localhost:8000** → giao diện chatbot AI.

### CLI

```bash
# Hỏi một câu
python -m app.cli --question "Chính sách hoàn trả hàng ra sao?"

# Chạy batch test
python -m app.cli --batch --test-file ../data/test.json

# Xuất JSON
python -m app.cli --question "Đơn hàng 1971 bao giờ giao?" --json
```

## 💬 Ví dụ câu hỏi

| Loại | Câu hỏi mẫu |
|------|-------------|
| **Policy** | "Chính sách hoàn trả hàng ra sao?" |
| **Data** | "Đơn hàng 1971 bao giờ được giao?" |
| **Policy + Data** | "Đơn hàng 1971 có được hoàn trả không?" |
| **Voucher** | "Voucher của khách hàng C001 còn mã nào dùng được?" |
| **Clarification** | "Voucher của tôi còn dùng được không?" → Hỏi lại mã KH |
| **Not Found** | "Kiểm tra đơn hàng 9999 giúp tôi" → Không tìm thấy |

## 🛠️ Tech Stack

- **LangGraph** – Multi-agent orchestration
- **ChromaDB** – Vector store cho policy search
- **sentence-transformers** – Embedding model (`all-MiniLM-L6-v2`)
- **LangChain** – LLM integration (Gemini, OpenAI, Ollama, Custom)
- **Python HTTP Server** – Lightweight web server, no framework needed

## 📊 API Endpoints

| Method | Path | Mô tả |
|--------|------|--------|
| `GET` | `/` | Chatbot UI |
| `GET` | `/api/health` | System health & stats |
| `POST` | `/api/chat` | Chat với natural language response |
| `POST` | `/api/ask` | Structured response (backward compat) |
| `POST` | `/api/batch` | Chạy batch test |

## 👤 Author

**Vũ Văn Học** – VinUni AI20k Cohort 2

---

*Built with ❤️ for Day 09 – Multi-Agent Architecture*
