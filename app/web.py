from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

from app.config import Settings
from app.data_access import ShoppingDataStore
from app.graph import ShoppingAssistant


_ASSISTANT: ShoppingAssistant | None = None


def get_assistant() -> ShoppingAssistant:
    global _ASSISTANT
    if _ASSISTANT is None:
        _ASSISTANT = ShoppingAssistant()
    return _ASSISTANT


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "ShoppingAssistantDemo/2.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(HTML_PAGE)
            return
        if path == "/api/health":
            self._send_json(_build_health_payload())
            return
        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/ask":
            self._handle_ask()
            return
        if path == "/api/chat":
            self._handle_chat()
            return
        if path == "/api/batch":
            self._handle_batch()
            return
        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_ask(self) -> None:
        payload = self._read_json()
        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json(
                {"error": "question_required"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        started_at = time.perf_counter()
        try:
            assistant = get_assistant()
            result = assistant.ask(
                question,
                rebuild_index=bool(payload.get("rebuild_index", False)),
            )
            result["runtime"] = {
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
                "embedding_backend": getattr(assistant.embedding_model, "backend", "unknown"),
                "llm_ready": assistant.llm is not None,
                "llm_error": assistant.llm_error,
            }
        except Exception as exc:
            self._send_json(
                {
                    "error": "ask_failed",
                    "message": str(exc),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(result)

    def _handle_chat(self) -> None:
        """Chat endpoint: runs the graph then rewrites the answer naturally."""
        payload = self._read_json()
        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json(
                {"error": "question_required"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        started_at = time.perf_counter()
        try:
            assistant = get_assistant()
            result = assistant.ask(
                question,
                rebuild_index=bool(payload.get("rebuild_index", False)),
            )

            # Generate a natural chatbot response via LLM
            policy_summary = (result.get("policy_result") or {}).get("summary", "")
            data_facts = (result.get("data_result") or {}).get("facts", [])
            natural_answer = assistant.generate_natural_response(
                question=question,
                structured_answer=result.get("final_answer", ""),
                policy_summary=policy_summary,
                data_facts=data_facts,
            )

            result["natural_answer"] = natural_answer
            result["runtime"] = {
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
                "embedding_backend": getattr(assistant.embedding_model, "backend", "unknown"),
                "llm_ready": assistant.llm is not None,
                "llm_error": assistant.llm_error,
            }
        except Exception as exc:
            self._send_json(
                {
                    "error": "chat_failed",
                    "message": str(exc),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(result)

    def _handle_batch(self) -> None:
        payload = self._read_json()
        started_at = time.perf_counter()
        try:
            assistant = get_assistant()
            settings = assistant.settings
            test_file = Path(payload.get("test_file") or settings.root_dir / "data" / "test.json")
            output_dir = Path(payload.get("output_dir") or settings.traces_dir)
            summary = assistant.run_batch(
                test_file,
                output_dir,
                rebuild_index=bool(payload.get("rebuild_index", False)),
            )
            summary["runtime"] = {
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
                "trace_dir": str(output_dir),
                "embedding_backend": getattr(assistant.embedding_model, "backend", "unknown"),
            }
        except Exception as exc:
            self._send_json(
                {
                    "error": "batch_failed",
                    "message": str(exc),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(summary)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw_body = self.rfile.read(length)
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def _build_health_payload() -> dict[str, Any]:
    settings = Settings.load()
    store = ShoppingDataStore(settings.orders_path)
    return {
        "app": "Multi-Agent Shopping Assistant",
        "provider": settings.provider,
        "model": settings.model,
        "top_k": settings.top_k,
        "counts": {
            "customers": len(store.customers),
            "orders": len(store.orders),
            "vouchers": len(store.vouchers),
        },
        "paths": {
            "policy": str(settings.policy_path),
            "orders": str(settings.orders_path),
            "traces": str(settings.traces_dir),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the shopping assistant demo web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo app is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Premium Chatbot HTML
# ---------------------------------------------------------------------------

HTML_PAGE = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Multi-Agent Shopping Assistant – Trợ lý mua sắm thông minh với kiến trúc đa tác tử, RAG và LLM.">
  <title>Shopping Assistant – AI Chatbot</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    /* ── Reset & Variables ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg-deep: #0a0a1a;
      --bg-primary: #0f0f23;
      --bg-secondary: #1a1a2e;
      --bg-tertiary: #16213e;
      --bg-glass: rgba(26, 26, 46, 0.72);
      --bg-glass-light: rgba(30, 30, 60, 0.55);
      --border-glass: rgba(255, 255, 255, 0.08);
      --border-active: rgba(99, 179, 237, 0.35);

      --text-primary: #e8eaf6;
      --text-secondary: #9ca3c2;
      --text-muted: #636b8a;
      --text-bright: #ffffff;

      --accent-1: #38bdf8;
      --accent-2: #818cf8;
      --accent-3: #a78bfa;
      --accent-gradient: linear-gradient(135deg, #38bdf8, #818cf8, #a78bfa);
      --accent-gradient-h: linear-gradient(90deg, #38bdf8, #818cf8);

      --user-bubble: linear-gradient(135deg, #1e3a5f 0%, #2d2b55 100%);
      --bot-bubble: rgba(22, 33, 62, 0.85);

      --success: #34d399;
      --warning: #fbbf24;
      --error: #f87171;

      --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
      --shadow-md: 0 8px 32px rgba(0,0,0,0.4);
      --shadow-lg: 0 20px 60px rgba(0,0,0,0.5);
      --shadow-glow: 0 0 20px rgba(56, 189, 248, 0.15);

      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 16px;
      --radius-xl: 20px;
      --radius-full: 9999px;

      --sidebar-width: 300px;
      --font: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
    }

    html { height: 100%; }

    body {
      font-family: var(--font);
      background: var(--bg-deep);
      color: var(--text-primary);
      height: 100%;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }

    button, input, textarea { font: inherit; }

    /* ── App Shell ── */
    .app {
      display: grid;
      grid-template-columns: var(--sidebar-width) 1fr;
      height: 100vh;
    }

    /* ── Sidebar ── */
    .sidebar {
      background: var(--bg-primary);
      border-right: 1px solid var(--border-glass);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .sidebar-header {
      padding: 24px 20px 16px;
      border-bottom: 1px solid var(--border-glass);
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 6px;
    }

    .logo-icon {
      width: 40px;
      height: 40px;
      border-radius: var(--radius-md);
      background: var(--accent-gradient);
      display: grid;
      place-items: center;
      font-size: 20px;
      flex-shrink: 0;
      box-shadow: var(--shadow-glow);
    }

    .logo-text {
      font-size: 17px;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.2;
    }

    .logo-sub {
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.4;
      margin-top: 2px;
    }

    /* ── Stats Grid ── */
    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-glass);
    }

    .stat {
      background: var(--bg-glass);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-glass);
      border-radius: var(--radius-md);
      padding: 10px 12px;
      transition: border-color 0.3s ease, transform 0.2s ease;
    }

    .stat:hover {
      border-color: var(--border-active);
      transform: translateY(-1px);
    }

    .stat-value {
      font-size: 20px;
      font-weight: 800;
      color: var(--text-bright);
    }

    .stat-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-top: 2px;
      font-weight: 600;
    }

    /* ── Section titles ── */
    .section-title {
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      font-weight: 700;
      padding: 16px 20px 8px;
    }

    /* ── Samples ── */
    .samples {
      flex: 1;
      overflow-y: auto;
      padding: 0 20px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .sample-btn {
      width: 100%;
      text-align: left;
      background: var(--bg-glass);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-glass);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
      color: var(--text-secondary);
      font-size: 13px;
      line-height: 1.45;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .sample-btn:hover {
      background: var(--bg-tertiary);
      border-color: var(--border-active);
      color: var(--text-primary);
      transform: translateX(3px);
    }

    /* ── Sidebar Footer ── */
    .sidebar-footer {
      padding: 12px 20px 16px;
      border-top: 1px solid var(--border-glass);
    }

    .batch-btn {
      width: 100%;
      padding: 10px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-glass);
      background: var(--bg-glass);
      color: var(--text-secondary);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .batch-btn:hover {
      background: var(--bg-tertiary);
      border-color: var(--border-active);
      color: var(--text-primary);
    }

    .batch-btn:disabled {
      opacity: 0.5;
      cursor: wait;
    }

    .batch-result {
      margin-top: 8px;
      font-size: 12px;
      color: var(--text-muted);
      background: var(--bg-glass);
      border: 1px solid var(--border-glass);
      border-radius: var(--radius-sm);
      padding: 10px;
      line-height: 1.6;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: var(--radius-full);
      font-size: 11px;
      font-weight: 600;
      border: 1px solid var(--border-glass);
      background: var(--bg-glass);
      backdrop-filter: blur(8px);
    }

    .status-badge .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--text-muted);
    }

    .status-badge.ok .dot { background: var(--success); }
    .status-badge.warn .dot { background: var(--warning); }
    .status-badge.err .dot { background: var(--error); }

    /* ── Chat Area ── */
    .chat-area {
      display: flex;
      flex-direction: column;
      height: 100vh;
      background:
        radial-gradient(ellipse at 20% 0%, rgba(56,189,248,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(129,140,248,0.05) 0%, transparent 50%),
        var(--bg-deep);
      position: relative;
    }

    /* ── Chat Header ── */
    .chat-header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border-glass);
      background: var(--bg-glass);
      backdrop-filter: blur(20px);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      z-index: 10;
    }

    .chat-header h1 {
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.01em;
      color: var(--text-bright);
    }

    .chat-header p {
      font-size: 12.5px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    /* ── Flow diagram ── */
    .flow {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 12px 24px;
      border-bottom: 1px solid var(--border-glass);
      background: rgba(10, 10, 26, 0.5);
      flex-shrink: 0;
      overflow-x: auto;
    }

    .flow-node {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: var(--radius-full);
      border: 1px solid var(--border-glass);
      background: var(--bg-glass);
      font-size: 11.5px;
      font-weight: 600;
      color: var(--text-muted);
      white-space: nowrap;
      transition: all 0.4s ease;
      flex-shrink: 0;
    }

    .flow-node .node-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--text-muted);
      transition: background 0.4s ease, box-shadow 0.4s ease;
    }

    .flow-node.active {
      border-color: rgba(56,189,248,0.35);
      color: var(--accent-1);
      background: rgba(56,189,248,0.08);
    }

    .flow-node.active .node-dot {
      background: var(--accent-1);
      box-shadow: 0 0 8px rgba(56,189,248,0.5);
    }

    .flow-arrow {
      color: var(--text-muted);
      font-size: 14px;
      opacity: 0.4;
      flex-shrink: 0;
    }

    /* ── Messages ── */
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scroll-behavior: smooth;
    }

    .messages::-webkit-scrollbar { width: 5px; }
    .messages::-webkit-scrollbar-track { background: transparent; }
    .messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }

    .msg {
      display: flex;
      gap: 12px;
      max-width: 85%;
      animation: msgIn 0.4s ease both;
    }

    @keyframes msgIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .msg.user { align-self: flex-end; flex-direction: row-reverse; }
    .msg.bot { align-self: flex-start; }

    .msg-avatar {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-size: 16px;
      flex-shrink: 0;
      margin-top: 2px;
    }

    .msg.user .msg-avatar {
      background: var(--user-bubble);
      border: 1px solid rgba(255,255,255,0.1);
    }

    .msg.bot .msg-avatar {
      background: var(--accent-gradient);
      box-shadow: var(--shadow-glow);
    }

    .msg-content {
      border-radius: var(--radius-lg);
      padding: 14px 18px;
      line-height: 1.65;
      font-size: 14.5px;
    }

    .msg.user .msg-content {
      background: var(--user-bubble);
      border: 1px solid rgba(255,255,255,0.08);
      color: var(--text-primary);
      border-bottom-right-radius: 4px;
    }

    .msg.bot .msg-content {
      background: var(--bot-bubble);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-glass);
      color: var(--text-primary);
      border-bottom-left-radius: 4px;
      box-shadow: var(--shadow-sm);
    }

    .msg-text { white-space: pre-wrap; word-break: break-word; }

    .msg-time {
      font-size: 10.5px;
      color: var(--text-muted);
      margin-top: 6px;
    }

    /* ── Evidence toggle ── */
    .evidence-toggle {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      margin-top: 10px;
      padding: 5px 10px;
      border-radius: var(--radius-full);
      border: 1px solid var(--border-glass);
      background: rgba(255,255,255,0.03);
      color: var(--text-muted);
      font-size: 11.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .evidence-toggle:hover {
      border-color: var(--border-active);
      color: var(--accent-1);
    }

    .evidence-toggle .arrow {
      transition: transform 0.2s ease;
      font-size: 10px;
    }

    .evidence-toggle.open .arrow { transform: rotate(90deg); }

    .evidence-panel {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.35s ease;
    }

    .evidence-panel.open { max-height: 800px; }

    .evidence-inner {
      margin-top: 10px;
      border-top: 1px solid var(--border-glass);
      padding-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .ev-section {
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border-glass);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
    }

    .ev-title {
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      font-weight: 700;
      margin-bottom: 4px;
    }

    .ev-body {
      font-size: 12px;
      color: var(--text-secondary);
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }

    /* ── Typing indicator ── */
    .typing {
      display: flex;
      gap: 12px;
      align-self: flex-start;
      max-width: 85%;
      animation: msgIn 0.3s ease both;
    }

    .typing-dots {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 14px 20px;
      background: var(--bot-bubble);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-glass);
      border-radius: var(--radius-lg);
      border-bottom-left-radius: 4px;
    }

    .typing-dots span {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
      animation: bounce 1.4s infinite ease-in-out;
    }

    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.16s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.32s; }

    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
      30% { transform: translateY(-8px); opacity: 1; }
    }

    /* ── Welcome state ── */
    .welcome {
      flex: 1;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 40px;
      animation: fadeIn 0.6s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    .welcome-icon {
      width: 72px;
      height: 72px;
      border-radius: 50%;
      background: var(--accent-gradient);
      display: grid;
      place-items: center;
      font-size: 34px;
      margin: 0 auto 18px;
      box-shadow: 0 0 40px rgba(56,189,248,0.2);
      animation: pulse 3s ease infinite;
    }

    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 20px rgba(56,189,248,0.15); }
      50% { box-shadow: 0 0 40px rgba(56,189,248,0.3); }
    }

    .welcome h2 {
      font-size: 22px;
      font-weight: 800;
      color: var(--text-bright);
      margin-bottom: 8px;
    }

    .welcome p {
      color: var(--text-muted);
      font-size: 14px;
      max-width: 420px;
      line-height: 1.6;
    }

    .welcome-chips {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 8px;
      margin-top: 20px;
      max-width: 520px;
    }

    .welcome-chip {
      padding: 8px 14px;
      border-radius: var(--radius-full);
      border: 1px solid var(--border-glass);
      background: var(--bg-glass);
      color: var(--text-secondary);
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .welcome-chip:hover {
      border-color: var(--border-active);
      color: var(--accent-1);
      background: rgba(56,189,248,0.07);
      transform: translateY(-2px);
    }

    /* ── Composer ── */
    .composer {
      padding: 16px 24px 20px;
      border-top: 1px solid var(--border-glass);
      background: var(--bg-glass);
      backdrop-filter: blur(20px);
      flex-shrink: 0;
    }

    .composer-inner {
      display: flex;
      gap: 10px;
      align-items: center;
      background: var(--bg-secondary);
      border: 1px solid var(--border-glass);
      border-radius: var(--radius-lg);
      padding: 6px 8px 6px 18px;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .composer-inner:focus-within {
      border-color: var(--border-active);
      box-shadow: 0 0 0 3px rgba(56,189,248,0.08);
    }

    .composer input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-primary);
      font-size: 14.5px;
      min-height: 42px;
    }

    .composer input::placeholder { color: var(--text-muted); }

    .composer-actions {
      display: flex;
      gap: 6px;
      flex-shrink: 0;
    }

    .btn-send {
      width: 42px;
      height: 42px;
      border-radius: var(--radius-md);
      border: none;
      background: var(--accent-gradient);
      color: white;
      font-size: 18px;
      cursor: pointer;
      display: grid;
      place-items: center;
      transition: transform 0.15s ease, box-shadow 0.2s ease;
      box-shadow: var(--shadow-glow);
    }

    .btn-send:hover {
      transform: scale(1.06);
      box-shadow: 0 0 24px rgba(56,189,248,0.3);
    }

    .btn-send:disabled {
      opacity: 0.45;
      cursor: wait;
      transform: none;
    }

    .btn-icon {
      width: 42px;
      height: 42px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-glass);
      background: transparent;
      color: var(--text-muted);
      font-size: 16px;
      cursor: pointer;
      display: grid;
      place-items: center;
      transition: all 0.15s ease;
    }

    .btn-icon:hover {
      border-color: var(--border-active);
      color: var(--accent-1);
    }

    .btn-icon.active {
      border-color: var(--accent-1);
      color: var(--accent-1);
      background: rgba(56,189,248,0.08);
    }

    .btn-icon:disabled {
      opacity: 0.45;
      cursor: wait;
    }

    /* ── Quick replies ── */
    .quick-replies {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }

    .quick-reply {
      padding: 5px 12px;
      border-radius: var(--radius-full);
      border: 1px solid var(--border-glass);
      background: rgba(255,255,255,0.03);
      color: var(--text-muted);
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      animation: msgIn 0.4s ease both;
    }

    .quick-reply:hover {
      border-color: var(--border-active);
      color: var(--accent-1);
    }

    /* ── Mobile toggle ── */
    .mobile-toggle {
      display: none;
      position: fixed;
      top: 12px;
      left: 12px;
      z-index: 1000;
      width: 40px;
      height: 40px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-glass);
      background: var(--bg-glass);
      backdrop-filter: blur(12px);
      color: var(--text-primary);
      font-size: 18px;
      cursor: pointer;
    }

    .sidebar-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.5);
      z-index: 99;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
      .app { grid-template-columns: 1fr; }

      .sidebar {
        position: fixed;
        left: -100%;
        top: 0;
        bottom: 0;
        width: 280px;
        z-index: 100;
        transition: left 0.3s ease;
      }

      .sidebar.open { left: 0; }
      .sidebar-overlay.open { display: block; }
      .mobile-toggle { display: grid; place-items: center; }

      .chat-header { padding-left: 60px; }

      .msg { max-width: 95%; }
    }
  </style>
</head>
<body>
  <button class="mobile-toggle" id="mobileToggle" type="button">☰</button>
  <div class="sidebar-overlay" id="sidebarOverlay"></div>

  <div class="app">
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="logo">
          <div class="logo-icon">🛒</div>
          <div>
            <div class="logo-text">Shopping Assistant</div>
            <div class="logo-sub">Multi-Agent · RAG · LLM</div>
          </div>
        </div>
      </div>

      <div class="stats">
        <div class="stat">
          <div class="stat-value" id="sCustomers">—</div>
          <div class="stat-label">Customers</div>
        </div>
        <div class="stat">
          <div class="stat-value" id="sOrders">—</div>
          <div class="stat-label">Orders</div>
        </div>
        <div class="stat">
          <div class="stat-value" id="sVouchers">—</div>
          <div class="stat-label">Vouchers</div>
        </div>
        <div class="stat">
          <div class="stat-value" id="sTopK">—</div>
          <div class="stat-label">RAG top-k</div>
        </div>
      </div>

      <div class="section-title">Câu hỏi mẫu</div>
      <div class="samples" id="sampleList"></div>

      <div class="sidebar-footer">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
          <div class="section-title" style="padding:0">Model</div>
          <div class="status-badge" id="modelBadge"><span class="dot"></span><span id="modelText">Loading…</span></div>
        </div>
        <button class="batch-btn" id="batchBtn" type="button">⚡ Chạy batch test</button>
        <div class="batch-result" id="batchResult" hidden></div>
      </div>
    </aside>

    <main class="chat-area">
      <header class="chat-header">
        <div>
          <h1>💬 Trợ lý mua sắm AI</h1>
          <p>Supervisor routes → Policy/RAG + Data Lookup → Response Agent</p>
        </div>
        <div class="status-badge" id="chatStatus"><span class="dot"></span><span>idle</span></div>
      </header>

      <div class="flow" id="flowBar">
        <div class="flow-node active" data-node="supervisor"><span class="node-dot"></span> Supervisor</div>
        <span class="flow-arrow">→</span>
        <div class="flow-node" data-node="policy"><span class="node-dot"></span> Policy / RAG</div>
        <span class="flow-arrow">→</span>
        <div class="flow-node" data-node="data"><span class="node-dot"></span> Order / Customer</div>
        <span class="flow-arrow">→</span>
        <div class="flow-node active" data-node="response"><span class="node-dot"></span> Response</div>
      </div>

      <div class="messages" id="messages">
        <div class="welcome" id="welcomeScreen">
          <div>
            <div class="welcome-icon">🤖</div>
            <h2>Xin chào!</h2>
            <p>Mình là trợ lý mua sắm AI. Hỏi mình bất cứ điều gì về đơn hàng, voucher, hay chính sách nhé!</p>
            <div class="welcome-chips" id="welcomeChips"></div>
          </div>
        </div>
      </div>

      <div class="composer">
        <form class="composer-inner" id="chatForm" autocomplete="off">
          <input id="chatInput" name="question" placeholder="Nhập câu hỏi về đơn hàng, voucher hoặc policy…" autocomplete="off">
          <div class="composer-actions">
            <button class="btn-icon" id="rebuildBtn" type="button" title="Rebuild RAG index">🔄</button>
            <button class="btn-send" id="sendBtn" type="submit" title="Gửi">➤</button>
          </div>
        </form>
      </div>
    </main>
  </div>

  <script>
    /* ── Config ── */
    const SAMPLES = [
      "Chính sách hoàn trả hàng ra sao?",
      "Đơn hàng 1971 bao giờ được giao?",
      "Đơn hàng 1971 có được hoàn trả không?",
      "Voucher của khách hàng C001 còn những mã nào dùng được?",
      "Voucher của tôi còn dùng được không?",
      "Kiểm tra đơn hàng 9999 giúp tôi"
    ];

    const QUICK_REPLIES = [
      ["Chính sách trả hàng?", "Giao hàng tiêu chuẩn bao lâu?", "Kiểm hàng khi nhận?"],
      ["Xem voucher C001", "Đơn 1971 bao giờ giao?", "Chính sách voucher khi hủy đơn?"],
    ];

    /* ── DOM refs ── */
    const $ = (id) => document.getElementById(id);
    const el = {
      sidebar: $("sidebar"),
      overlay: $("sidebarOverlay"),
      mobileToggle: $("mobileToggle"),
      sampleList: $("sampleList"),
      welcomeChips: $("welcomeChips"),
      welcomeScreen: $("welcomeScreen"),
      messages: $("messages"),
      chatForm: $("chatForm"),
      chatInput: $("chatInput"),
      sendBtn: $("sendBtn"),
      rebuildBtn: $("rebuildBtn"),
      batchBtn: $("batchBtn"),
      batchResult: $("batchResult"),
      modelBadge: $("modelBadge"),
      modelText: $("modelText"),
      chatStatus: $("chatStatus"),
      sCustomers: $("sCustomers"),
      sOrders: $("sOrders"),
      sVouchers: $("sVouchers"),
      sTopK: $("sTopK"),
    };

    let isBusy = false;
    let rebuildNext = false;
    let msgHistory = [];
    let quickReplyIdx = 0;

    /* ── Helpers ── */
    function timeStr() {
      const d = new Date();
      return d.getHours().toString().padStart(2,"0") + ":" + d.getMinutes().toString().padStart(2,"0");
    }

    function escHtml(str) {
      const d = document.createElement("div");
      d.textContent = str;
      return d.innerHTML;
    }

    function setBusy(busy) {
      isBusy = busy;
      el.sendBtn.disabled = busy;
      el.rebuildBtn.disabled = busy;
      el.batchBtn.disabled = busy;
      const st = el.chatStatus;
      if (busy) {
        st.className = "status-badge warn";
        st.innerHTML = '<span class="dot"></span><span>processing…</span>';
      } else {
        st.className = "status-badge ok";
        st.innerHTML = '<span class="dot"></span><span>ready</span>';
      }
    }

    function updateFlow(selectedWorkers) {
      document.querySelectorAll(".flow-node").forEach(n => {
        const name = n.dataset.node;
        const active = name === "supervisor" || name === "response" || (selectedWorkers || []).includes(name);
        n.classList.toggle("active", active);
      });
    }

    /* ── Render messages ── */
    function hideWelcome() {
      if (el.welcomeScreen) el.welcomeScreen.remove();
    }

    function addUserMsg(text) {
      hideWelcome();
      const html = `
        <div class="msg user">
          <div class="msg-avatar">👤</div>
          <div class="msg-content">
            <div class="msg-text">${escHtml(text)}</div>
            <div class="msg-time">${timeStr()}</div>
          </div>
        </div>`;
      el.messages.insertAdjacentHTML("beforeend", html);
      scrollDown();
    }

    function addBotMsg(text, result) {
      const route = result?.route || {};
      const selected = route.selected_workers || [];
      const policy = result?.policy_result || {};
      const data = result?.data_result || {};
      const elapsed = result?.runtime?.elapsed_ms;

      let evidenceHtml = "";

      // Route info
      let routeInfo = `Status: ${route.status || result?.status || "ok"}\n`;
      routeInfo += `Workers: ${selected.length ? selected.join(", ") : "none"}\n`;
      if (route.order_ids?.length) routeInfo += `Order IDs: ${route.order_ids.join(", ")}\n`;
      if (route.customer_ids?.length) routeInfo += `Customer IDs: ${route.customer_ids.join(", ")}\n`;
      if (route.reason) routeInfo += `Reason: ${route.reason}`;

      evidenceHtml += `<div class="ev-section"><div class="ev-title">🔀 Route</div><div class="ev-body">${escHtml(routeInfo.trim())}</div></div>`;

      // Policy evidence
      const citations = policy.citations || [];
      if (citations.length) {
        evidenceHtml += `<div class="ev-section"><div class="ev-title">📜 Policy</div><div class="ev-body">${escHtml(citations.slice(0,5).join("\n"))}</div></div>`;
      }

      // Data evidence
      const facts = data.facts || [];
      if (facts.length) {
        evidenceHtml += `<div class="ev-section"><div class="ev-title">📊 Data</div><div class="ev-body">${escHtml(facts.slice(0,5).join("\n"))}</div></div>`;
      }

      // Trace
      const traceObj = {
        route: result?.route,
        policy_result: compactPolicy(policy),
        data_result: compactData(data),
        runtime: result?.runtime,
      };
      evidenceHtml += `<div class="ev-section"><div class="ev-title">🔍 Trace JSON</div><div class="ev-body" style="font-size:11px;max-height:200px;overflow:auto">${escHtml(JSON.stringify(traceObj, null, 2))}</div></div>`;

      const uid = "ev_" + Date.now();
      const timeInfo = elapsed ? `${timeStr()} · ${elapsed}ms` : timeStr();

      const html = `
        <div class="msg bot">
          <div class="msg-avatar">🤖</div>
          <div class="msg-content">
            <div class="msg-text">${escHtml(text)}</div>
            <div class="msg-time">${timeInfo}</div>
            <button class="evidence-toggle" onclick="toggleEvidence('${uid}', this)">
              <span class="arrow">▶</span> Chi tiết & Evidence
            </button>
            <div class="evidence-panel" id="${uid}">
              <div class="evidence-inner">${evidenceHtml}</div>
            </div>
          </div>
        </div>`;
      el.messages.insertAdjacentHTML("beforeend", html);
      scrollDown();
    }

    function addErrorMsg(text) {
      const html = `
        <div class="msg bot">
          <div class="msg-avatar">⚠️</div>
          <div class="msg-content" style="border-color: rgba(248,113,113,0.3)">
            <div class="msg-text" style="color:var(--error)">${escHtml(text)}</div>
            <div class="msg-time">${timeStr()}</div>
          </div>
        </div>`;
      el.messages.insertAdjacentHTML("beforeend", html);
      scrollDown();
    }

    function showTyping() {
      const html = `
        <div class="typing" id="typingIndicator">
          <div class="msg-avatar" style="background:var(--accent-gradient);width:34px;height:34px;border-radius:50%;display:grid;place-items:center;font-size:16px">🤖</div>
          <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>`;
      el.messages.insertAdjacentHTML("beforeend", html);
      scrollDown();
    }

    function hideTyping() {
      const t = document.getElementById("typingIndicator");
      if (t) t.remove();
    }

    function showQuickReplies() {
      const replies = QUICK_REPLIES[quickReplyIdx % QUICK_REPLIES.length];
      quickReplyIdx++;
      let html = '<div class="quick-replies">';
      for (const r of replies) {
        html += `<button class="quick-reply" onclick="quickAsk(this.textContent)">${escHtml(r)}</button>`;
      }
      html += '</div>';
      el.messages.insertAdjacentHTML("beforeend", html);
      scrollDown();
    }

    function scrollDown() {
      requestAnimationFrame(() => {
        el.messages.scrollTop = el.messages.scrollHeight;
      });
    }

    /* ── Evidence toggle ── */
    window.toggleEvidence = function(id, btn) {
      const panel = document.getElementById(id);
      panel.classList.toggle("open");
      btn.classList.toggle("open");
    };

    /* ── Compact helpers ── */
    function compactPolicy(p) {
      if (!p) return {};
      return { status: p.status, query: p.query, summary: p.summary, citations: p.citations,
        hits: (p.hits||[]).map(h => ({ citation: h.citation, distance: h.distance })) };
    }
    function compactData(d) {
      if (!d) return {};
      return { status: d.status, facts: d.facts, not_found_entities: d.not_found_entities,
        missing_fields: d.missing_fields, tool_calls: d.tool_calls };
    }

    /* ── Quick ask ── */
    window.quickAsk = function(question) {
      // Remove all quick-reply containers
      document.querySelectorAll(".quick-replies").forEach(el => el.remove());
      el.chatInput.value = question;
      doChat(question);
    };

    /* ── Main chat function ── */
    async function doChat(question) {
      const q = question.trim();
      if (!q || isBusy) return;

      addUserMsg(q);
      setBusy(true);
      showTyping();

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, rebuild_index: rebuildNext }),
        });
        rebuildNext = false;
        const payload = await res.json();

        hideTyping();

        if (!res.ok) {
          addErrorMsg(payload.message || payload.error || "Có lỗi xảy ra.");
          updateFlow([]);
          return;
        }

        const answer = payload.natural_answer || payload.final_answer || "";
        addBotMsg(answer, payload);
        updateFlow((payload.route || {}).selected_workers);
        showQuickReplies();

      } catch (err) {
        hideTyping();
        addErrorMsg(String(err));
      } finally {
        setBusy(false);
      }
    }

    /* ── Event listeners ── */
    el.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      doChat(el.chatInput.value);
      el.chatInput.value = "";
    });

    el.rebuildBtn.addEventListener("click", () => {
      rebuildNext = !rebuildNext;
      el.rebuildBtn.classList.toggle("active", rebuildNext);
      el.rebuildBtn.title = rebuildNext ? "Index sẽ rebuild ở câu hỏi tiếp theo" : "Rebuild RAG index";
    });

    el.batchBtn.addEventListener("click", async () => {
      setBusy(true);
      el.batchResult.hidden = false;
      el.batchResult.textContent = "Đang chạy batch test…";
      try {
        const res = await fetch("/api/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rebuild_index: false }),
        });
        const s = await res.json();
        if (!res.ok) {
          el.batchResult.textContent = s.message || s.error || "Batch failed";
          return;
        }
        el.batchResult.innerHTML =
          `<div>Total: <b>${s.total}</b></div>` +
          `<div>Route matches: <b>${s.route_matches}/${s.total}</b></div>` +
          `<div>Status matches: <b>${s.status_matches}/${s.total}</b></div>` +
          `<div style="color:var(--text-muted);margin-top:4px">Elapsed: ${s.runtime?.elapsed_ms}ms</div>`;
      } catch (err) {
        el.batchResult.textContent = String(err);
      } finally {
        setBusy(false);
      }
    });

    /* ── Mobile sidebar ── */
    el.mobileToggle.addEventListener("click", () => {
      el.sidebar.classList.toggle("open");
      el.overlay.classList.toggle("open");
    });
    el.overlay.addEventListener("click", () => {
      el.sidebar.classList.remove("open");
      el.overlay.classList.remove("open");
    });

    /* ── Init ── */
    function initSamples() {
      for (const q of SAMPLES) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sample-btn";
        btn.textContent = q;
        btn.addEventListener("click", () => {
          el.sidebar.classList.remove("open");
          el.overlay.classList.remove("open");
          el.chatInput.value = q;
          doChat(q);
          el.chatInput.value = "";
        });
        el.sampleList.appendChild(btn);
      }

      // Welcome chips
      for (const q of SAMPLES.slice(0, 4)) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "welcome-chip";
        chip.textContent = q;
        chip.addEventListener("click", () => {
          el.chatInput.value = q;
          doChat(q);
          el.chatInput.value = "";
        });
        el.welcomeChips.appendChild(chip);
      }
    }

    async function loadHealth() {
      try {
        const res = await fetch("/api/health");
        const h = await res.json();
        el.sCustomers.textContent = h.counts.customers;
        el.sOrders.textContent = h.counts.orders;
        el.sVouchers.textContent = h.counts.vouchers;
        el.sTopK.textContent = h.top_k;
        el.modelText.textContent = `${h.provider} / ${h.model}`;
        el.modelBadge.className = "status-badge ok";
        setBusy(false);
      } catch (err) {
        el.modelText.textContent = "Error";
        el.modelBadge.className = "status-badge err";
      }
    }

    initSamples();
    loadHealth();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
