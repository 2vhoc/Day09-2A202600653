from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.data_access import ShoppingDataStore, build_data_tools
from app.state import ShoppingState
from provider import get_chat_model
from rag.embeddings import SentenceTransformerEmbeddings
from rag.vector_store import ChromaPolicyStore


_GRAPH_RUNTIME: dict[str, Any] = {}
ORDER_ID_RE = re.compile(r"\b\d{4,}\b")
CUSTOMER_ID_RE = re.compile(r"\bC\d{3,}\b", re.IGNORECASE)


class ShoppingAssistant:
    """Shopping assistant orchestrated as a LangGraph multi-agent workflow."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.llm_error: str | None = None

        try:
            self.llm = get_chat_model(self.settings)
        except Exception as exc:
            self.llm = None
            self.llm_error = str(exc)

        self.data_store = ShoppingDataStore(self.settings.orders_path)
        self.data_tools = build_data_tools(self.data_store)
        self.data_tools_by_name = {data_tool.name: data_tool for data_tool in self.data_tools}

        self.embedding_model = SentenceTransformerEmbeddings(
            self.settings.embedding_model_name,
        )
        self.policy_store = ChromaPolicyStore(
            persist_directory=self.settings.chroma_dir,
            embedding_model=self.embedding_model,
        )

        self.graph = build_graph(
            {
                "llm": self.llm,
                "llm_error": self.llm_error,
                "settings": self.settings,
                "policy_store": self.policy_store,
                "data_store": self.data_store,
                "data_tools_by_name": self.data_tools_by_name,
            }
        )

    def ask(
        self,
        question: str,
        trace_file: Path | None = None,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        if rebuild_index:
            self.policy_store.rebuild(self.settings.policy_path)
        else:
            self.policy_store.ensure_index(self.settings.policy_path)

        final_state = self.graph.invoke({"question": question, "trace": []})
        payload = {
            "question": question,
            "status": _status_from_state(final_state),
            "route": final_state.get("route", {}),
            "policy_result": final_state.get("policy_result", {}),
            "data_result": final_state.get("data_result", {}),
            "final_answer": final_state.get("final_answer", ""),
            "trace": final_state.get("trace", []),
        }

        if trace_file:
            trace_path = Path(trace_file)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return payload

    def generate_natural_response(
        self,
        question: str,
        structured_answer: str,
        policy_summary: str = "",
        data_facts: list[str] | None = None,
    ) -> str:
        """Use the LLM to rewrite a structured answer as a friendly chatbot reply."""
        if self.llm is None:
            return structured_answer

        facts_text = "\n".join(data_facts or [])
        prompt = (
            "Bạn là trợ lý mua sắm thân thiện, chuyên nghiệp. "
            "Dựa trên câu hỏi của khách hàng và thông tin bên dưới, hãy viết một câu trả lời "
            "tự nhiên, lịch sự, dễ hiểu bằng tiếng Việt. Trả lời trực tiếp, không dùng markdown, "
            "không lặp lại câu hỏi, không nói 'Dựa trên dữ liệu'. Giọng văn ấm áp, chuyên nghiệp.\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Thông tin có sẵn:\n{structured_answer}\n\n"
        )
        if policy_summary:
            prompt += f"Tóm tắt chính sách: {policy_summary}\n\n"
        if facts_text:
            prompt += f"Dữ liệu tra cứu:\n{facts_text}\n\n"
        prompt += "Câu trả lời tự nhiên:"

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response)).strip()
            return content if content else structured_answer
        except Exception:
            return structured_answer

    def run_batch(
        self,
        test_file: Path,
        output_dir: Path,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        tests = json.loads(Path(test_file).read_text(encoding="utf-8"))
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        rows = []
        for index, case in enumerate(tests, start=1):
            case_id = case.get("id") or f"case_{index:03d}"
            result = self.ask(
                case["question"],
                trace_file=output_path / f"{case_id}.trace.json",
                rebuild_index=rebuild_index and index == 1,
            )
            actual_route = result["route"].get("selected_workers", [])
            actual_status = result["status"]
            rows.append(
                {
                    "id": case_id,
                    "question": case["question"],
                    "expected_route": case.get("expected_route", []),
                    "actual_route": actual_route,
                    "route_match": sorted(actual_route) == sorted(case.get("expected_route", [])),
                    "expected_status": case.get("expected_status"),
                    "actual_status": actual_status,
                    "status_match": actual_status == case.get("expected_status"),
                    "final_answer": result["final_answer"],
                }
            )

        summary = {
            "total": len(rows),
            "route_matches": sum(1 for row in rows if row["route_match"]),
            "status_matches": sum(1 for row in rows if row["status_match"]),
            "results": rows,
        }
        (output_path / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary


def build_graph(runtime: dict[str, Any] | None = None) -> Any:
    if runtime is not None:
        _GRAPH_RUNTIME.clear()
        _GRAPH_RUNTIME.update(runtime)

    graph = StateGraph(ShoppingState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("worker_1_policy", worker_1_policy_node)
    graph.add_node("worker_2_data", worker_2_data_node)
    graph.add_node("worker_3_response", worker_3_response_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "policy": "worker_1_policy",
            "data": "worker_2_data",
            "response": "worker_3_response",
        },
    )
    graph.add_conditional_edges(
        "worker_1_policy",
        _route_after_policy,
        {
            "data": "worker_2_data",
            "response": "worker_3_response",
        },
    )
    graph.add_edge("worker_2_data", "worker_3_response")
    graph.add_edge("worker_3_response", END)
    return graph.compile()


def supervisor_node(state: ShoppingState) -> ShoppingState:
    question = state["question"]
    route = _classify_question(question)

    llm_route = _call_supervisor_llm(question)
    if llm_route:
        route["llm_route"] = llm_route

    return {
        "route": route,
        "trace": [
            {
                "node": "supervisor",
                "question": question,
                "route": route,
            }
        ],
    }


def worker_1_policy_node(state: ShoppingState) -> ShoppingState:
    settings: Settings = _runtime("settings")
    policy_store: ChromaPolicyStore = _runtime("policy_store")
    question = state["question"]
    query = _enrich_policy_query(question)
    hits = policy_store.search(query, top_k=settings.top_k)
    hits = _merge_policy_hits(hits, _keyword_policy_hits(policy_store, question, settings.top_k))

    facts = _policy_facts_from_question(question)
    if not facts and hits:
        facts = [_shorten_whitespace(hits[0]["content"])[:360]]

    citations = _merge_unique(
        _topic_citations(question),
        [hit["citation"] for hit in hits if hit.get("citation")],
    )
    summary = _summarize_policy(question, facts, citations)

    policy_result = {
        "status": "ok",
        "query": query,
        "summary": summary,
        "facts": facts,
        "citations": citations,
        "hits": hits,
    }
    return {
        "policy_result": policy_result,
        "trace": [
            {
                "node": "worker_1_policy",
                "tool": "search_policy",
                "query": query,
                "hits": [
                    {
                        "citation": hit["citation"],
                        "distance": hit["distance"],
                    }
                    for hit in hits
                ],
                "result": {
                    "summary": summary,
                    "citations": citations,
                },
            }
        ],
    }


def worker_2_data_node(state: ShoppingState) -> ShoppingState:
    tools: dict[str, Any] = _runtime("data_tools_by_name")
    question = state["question"]
    route = state.get("route", {})
    order_ids = route.get("order_ids") or _extract_order_ids(question)
    customer_ids = route.get("customer_ids") or _extract_customer_ids(question)

    tool_calls = []
    tool_results = []
    facts = []
    not_found_entities = []

    for order_id in order_ids:
        result = _invoke_tool(
            tools,
            "get_order_detail_by_order_id",
            {"order_id": order_id},
        )
        tool_calls.append({"tool": "get_order_detail_by_order_id", "args": {"order_id": order_id}})
        tool_results.append(result)
        if result.get("status") == "not_found":
            not_found_entities.append({"entity": "order", "id": order_id})
            continue

        order = result["order"]
        facts.append(_format_order_fact(order))

    for customer_id in customer_ids:
        customer_result = _invoke_tool(
            tools,
            "get_customer_by_id",
            {"customer_id": customer_id},
        )
        tool_calls.append({"tool": "get_customer_by_id", "args": {"customer_id": customer_id}})
        tool_results.append(customer_result)
        if customer_result.get("status") == "not_found":
            not_found_entities.append({"entity": "customer", "id": customer_id})
            continue

        customer = customer_result["customer"]
        facts.append(_format_customer_fact(customer))

        if _asks_for_customer_orders(question):
            orders_result = _invoke_tool(
                tools,
                "get_orders_by_customer_id",
                {"customer_id": customer_id, "limit": 10},
            )
            tool_calls.append(
                {
                    "tool": "get_orders_by_customer_id",
                    "args": {"customer_id": customer_id, "limit": 10},
                }
            )
            tool_results.append(orders_result)
            if orders_result.get("status") == "ok":
                facts.append(_format_orders_list_fact(orders_result["orders"]))

        if _asks_for_vouchers(question):
            only_active = _asks_for_active_vouchers(question)
            vouchers_result = _invoke_tool(
                tools,
                "get_vouchers_by_customer_id",
                {"customer_id": customer_id, "only_active": only_active},
            )
            tool_calls.append(
                {
                    "tool": "get_vouchers_by_customer_id",
                    "args": {"customer_id": customer_id, "only_active": only_active},
                }
            )
            tool_results.append(vouchers_result)
            if vouchers_result.get("status") == "ok":
                facts.append(_format_vouchers_fact(vouchers_result["vouchers"], only_active))

    if not_found_entities:
        data_result = {
            "status": "not_found",
            "facts": facts,
            "not_found_entities": not_found_entities,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }
    elif not order_ids and not customer_ids:
        data_result = {
            "status": "clarification_needed",
            "facts": [],
            "missing_fields": ["order_id_or_customer_id"],
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }
    else:
        data_result = {
            "status": "ok",
            "facts": facts,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    return {
        "data_result": data_result,
        "trace": [
            {
                "node": "worker_2_data",
                "tool_calls": tool_calls,
                "result_status": data_result["status"],
                "facts": facts,
                "not_found_entities": not_found_entities,
            }
        ],
    }


def worker_3_response_node(state: ShoppingState) -> ShoppingState:
    route = state.get("route", {})
    policy_result = state.get("policy_result", {})
    data_result = state.get("data_result", {})

    if route.get("status") == "clarification_needed":
        final_answer = (
            "Status: clarification_needed\n"
            f"Question: {route.get('clarification_question')}"
        )
    elif data_result.get("status") == "clarification_needed":
        final_answer = (
            "Status: clarification_needed\n"
            "Question: Anh/chị vui lòng cung cấp mã đơn hàng hoặc mã khách hàng để em kiểm tra chính xác."
        )
    elif data_result.get("status") == "not_found":
        final_answer = _build_not_found_answer(data_result)
    else:
        answer = _build_success_answer(state)
        policy_evidence = _format_policy_evidence(policy_result)
        data_evidence = _format_data_evidence(data_result)
        final_answer = (
            f"Answer: {answer}\n"
            "Evidence:\n"
            f"- Policy: {policy_evidence}\n"
            f"- Order data: {data_evidence}"
        )

    return {
        "final_answer": final_answer,
        "trace": [
            {
                "node": "worker_3_response",
                "status": _status_from_state(
                    {
                        "route": route,
                        "data_result": data_result,
                    }
                ),
                "final_answer": final_answer,
            }
        ],
    }


def _route_after_supervisor(state: ShoppingState) -> str:
    route = state.get("route", {})
    if route.get("status") == "clarification_needed":
        return "response"
    if route.get("needs_policy"):
        return "policy"
    if route.get("needs_data"):
        return "data"
    return "response"


def _route_after_policy(state: ShoppingState) -> str:
    if state.get("route", {}).get("needs_data"):
        return "data"
    return "response"


def _classify_question(question: str) -> dict[str, Any]:
    text = question.lower()
    order_ids = _extract_order_ids(question)
    customer_ids = _extract_customer_ids(question)

    asks_specific_voucher_without_customer = (
        "voucher" in text
        and not customer_ids
        and any(term in text for term in ["của tôi", "còn dùng", "còn mã", "mã nào"])
    )
    asks_specific_order_without_id = (
        "đơn hàng" in text
        and not order_ids
        and any(term in text for term in ["của tôi", "đơn của tôi", "trạng thái", "bao giờ", "có được"])
    )

    if asks_specific_voucher_without_customer:
        return {
            "status": "clarification_needed",
            "needs_policy": False,
            "needs_data": False,
            "selected_workers": [],
            "order_ids": order_ids,
            "customer_ids": customer_ids,
            "clarification_question": "Anh/chị vui lòng cung cấp mã khách hàng để em kiểm tra voucher chính xác.",
            "reason": "Missing customer_id for account-specific voucher lookup.",
        }

    if asks_specific_order_without_id:
        return {
            "status": "clarification_needed",
            "needs_policy": False,
            "needs_data": False,
            "selected_workers": [],
            "order_ids": order_ids,
            "customer_ids": customer_ids,
            "clarification_question": "Anh/chị vui lòng cung cấp mã đơn hàng để em kiểm tra chính xác.",
            "reason": "Missing order_id for order-specific lookup.",
        }

    needs_data = bool(order_ids or customer_ids)
    needs_policy = _asks_policy(question)

    if order_ids and _asks_order_policy_mix(question):
        needs_policy = True
        needs_data = True

    if customer_ids and not _explicit_policy_request(question):
        needs_policy = False

    if not needs_policy and not needs_data:
        needs_policy = True

    selected_workers = []
    if needs_policy:
        selected_workers.append("policy")
    if needs_data:
        selected_workers.append("data")

    return {
        "status": "ok",
        "needs_policy": needs_policy,
        "needs_data": needs_data,
        "selected_workers": selected_workers,
        "order_ids": order_ids,
        "customer_ids": customer_ids,
        "clarification_question": None,
        "reason": "Rule-based route with optional LLM trace.",
    }


def _call_supervisor_llm(question: str) -> dict[str, Any] | None:
    if os.getenv("LLM_SUPERVISOR_ENABLED", "").lower() not in {"1", "true", "yes"}:
        return None

    llm = _GRAPH_RUNTIME.get("llm")
    if llm is None:
        return None

    prompt = (
        "Bạn là Supervisor Agent cho shopping assistant. "
        "Hãy route câu hỏi vào policy worker, data worker, cả hai, hoặc clarification. "
        "Chỉ trả JSON nhỏ với keys: status, needs_policy, needs_data, clarification_question.\n\n"
        f"Câu hỏi: {question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        return {"error": str(exc)}

    content = getattr(response, "content", response)
    return _extract_json(str(content)) or {"raw": str(content)}


def _build_success_answer(state: ShoppingState) -> str:
    question = state["question"]
    route = state.get("route", {})
    policy_result = state.get("policy_result", {})
    data_result = state.get("data_result", {})
    text = question.lower()

    orders = _orders_from_data_result(data_result)
    customers = _customers_from_data_result(data_result)
    vouchers = _vouchers_from_data_result(data_result)

    if orders and route.get("needs_policy"):
        order = orders[0]
        return _answer_mixed_order_policy(question, order)

    if orders:
        order = orders[0]
        if any(term in text for term in ["bao giờ", "khi nào", "dự kiến", "giao"]):
            return (
                f"Đơn hàng {order['order_id']} đang ở trạng thái {order.get('order_status')}. "
                f"Dự kiến giao ngày {order.get('estimated_delivery')}; ghi chú mới nhất: "
                f"{order.get('latest_status_note')}"
            )
        return (
            f"Đơn hàng {order['order_id']} đang ở trạng thái {order.get('order_status')}. "
            f"Ghi chú mới nhất: {order.get('latest_status_note')}"
        )

    if customers and _asks_for_customer_orders(question):
        customer = customers[0]
        return (
            f"Khách hàng {customer['customer_id']} có các đơn gần đây như trong dữ liệu: "
            f"{_orders_short_text(_orders_list_from_data_result(data_result))}."
        )

    if customers and _asks_for_customer_quota(question):
        customer = customers[0]
        return (
            f"Khách hàng {customer['customer_id']} thuộc hạng {customer.get('tier')}. "
            f"Tối đa dùng {customer.get('max_voucher_per_month')} voucher mỗi tháng, "
            f"đã dùng {customer.get('vouchers_used_this_month')} và còn "
            f"{customer.get('remaining_voucher_quota_this_month')} quota voucher trong tháng này."
        )

    if customers and _asks_for_vouchers(question):
        customer = customers[0]
        active_text = _vouchers_short_text(vouchers)
        return (
            f"Khách hàng {customer['customer_id']} thuộc hạng {customer.get('tier')} và còn quota voucher tháng này là "
            f"{customer.get('remaining_voucher_quota_this_month')}/{customer.get('max_voucher_per_month')}. "
            f"Voucher phù hợp: {active_text}."
        )

    if customers:
        customer = customers[0]
        return (
            f"Khách hàng {customer['customer_id']} thuộc hạng {customer.get('tier')}. "
            f"Hạn mức voucher mỗi tháng là {customer.get('max_voucher_per_month')}, "
            f"đã dùng {customer.get('vouchers_used_this_month')} và còn "
            f"{customer.get('remaining_voucher_quota_this_month')} quota trong tháng này."
        )

    facts = policy_result.get("facts") or []
    if facts:
        return " ".join(facts)
    return policy_result.get("summary") or "Em đã tìm thấy thông tin chính sách liên quan trong policy."


def _answer_mixed_order_policy(question: str, order: dict[str, Any]) -> str:
    text = question.lower()
    order_id = order["order_id"]
    status = order.get("order_status")

    if order.get("can_return_now"):
        return (
            f"Đơn hàng {order_id} có thể gửi yêu cầu trả hàng ở thời điểm hiện tại. "
            f"Đơn đã giao thành công, cửa sổ trả hàng còn đến {order.get('eligible_for_return_until')}; "
            "policy mặc định hỗ trợ đa số ngành hàng trong tối đa 15 ngày kể từ khi giao thành công."
        )

    if status == "in_transit":
        if "từ chối nhận" in text or "đang giao" in text:
            return (
                f"Đơn hàng {order_id} đang trên đường giao nên chưa thể bắt đầu quy trình trả hàng thông thường. "
                "Theo policy, đơn in_transit có thể được hỗ trợ hủy hoặc từ chối nhận trong một số trường hợp, "
                "nhưng chưa được xem là trả hàng sau bán."
            )
        return (
            f"Đơn hàng {order_id} chưa thể hoàn trả theo quy trình trả hàng thông thường vì đơn vẫn đang "
            f"{status} và chưa giao thành công. Policy tính cửa sổ trả hàng từ lúc đơn delivered; "
            "hiện dữ liệu đơn cũng ghi can_return_now=False."
        )

    return (
        f"Đơn hàng {order_id} hiện có trạng thái {status}, can_return_now={order.get('can_return_now')}. "
        f"Cửa sổ trả hàng trong dữ liệu là {order.get('eligible_for_return_until')}; "
        "cần đối chiếu trạng thái giao thành công và điều kiện ngành hàng trước khi tạo yêu cầu."
    )


def _build_not_found_answer(data_result: dict[str, Any]) -> str:
    entities = data_result.get("not_found_entities", [])
    if not entities:
        return "Status: not_found\nMessage: Không tìm thấy dữ liệu phù hợp."

    entity_text = ", ".join(f"{item['entity']} {item['id']}" for item in entities)
    return f"Status: not_found\nMessage: Không tìm thấy dữ liệu cho {entity_text}."


def _format_policy_evidence(policy_result: dict[str, Any]) -> str:
    citations = policy_result.get("citations") or []
    if not citations:
        return "Không dùng policy."
    return "; ".join(citations[:4])


def _format_data_evidence(data_result: dict[str, Any]) -> str:
    facts = data_result.get("facts") or []
    if not facts:
        return "Không dùng dữ liệu đơn hàng/khách hàng."
    return " ".join(facts[:3])


def _summarize_policy(question: str, facts: list[str], citations: list[str]) -> str:
    if facts:
        return " ".join(facts)
    if citations:
        return f"Tìm thấy policy liên quan ở: {'; '.join(citations[:3])}."
    return f"Tìm thấy policy liên quan cho câu hỏi: {question}"


def _policy_facts_from_question(question: str) -> list[str]:
    text = question.lower()
    facts = []

    if any(term in text for term in ["hoàn trả", "trả hàng", "hoàn tiền", "15 ngày", "cửa sổ"]):
        facts.append(
            "Thời hạn mặc định cho đa số ngành hàng là tối đa 15 ngày kể từ khi đơn giao hàng thành công."
        )
        facts.append(
            "Đơn chưa giao thành công thường chưa thể bắt đầu quy trình trả hàng thông thường."
        )

    if "không hỗ trợ" in text:
        facts.append(
            "Một số trường hợp thường không hỗ trợ trả hàng gồm hàng đã mở niêm phong, thực phẩm tươi sống, "
            "đồ cá nhân đã qua sử dụng, mã điện tử, hàng cá nhân hóa hoặc sản phẩm hỏng do dùng sai hướng dẫn."
        )

    if "kiểm hàng" in text:
        facts.append(
            "Khách được kiểm tra ngoại quan gói hàng khi nhận; việc dùng thử sâu, lắp đặt hoặc sử dụng lâu dài "
            "không thuộc phạm vi kiểm hàng tại chỗ."
        )

    if "giao hàng tiêu chuẩn" in text or "giao tiêu chuẩn" in text:
        facts.append(
            "Giao hàng tiêu chuẩn thường mất 1-2 ngày trong nội thành, 2-4 ngày liên tỉnh lân cận, "
            "và 3-7 ngày với tuyến huyện/xã hoặc khu vực xa."
        )

    if "giao nhanh" in text or "giao ưu tiên" in text:
        facts.append(
            "Đơn giao nhanh có thể bị chuyển sang giao tiêu chuẩn nếu shop bàn giao chậm, sản phẩm vượt ngưỡng "
            "kích thước/khối lượng, hoặc có rủi ro vận hành từ đơn vị vận chuyển."
        )

    if "voucher" in text and ("hủy" in text or "huỷ" in text or "hoàn lại" in text):
        facts.append(
            "Voucher có thể được hoàn lại khi đơn bị hủy nếu còn hiệu lực, chưa vượt giới hạn sử dụng, "
            "toàn bộ phần dùng voucher bị hủy và không thuộc nhóm chiến dịch loại trừ."
        )

    return _merge_unique([], facts)


def _topic_citations(question: str) -> list[str]:
    text = question.lower()
    citations = []
    if any(term in text for term in ["hoàn trả", "trả hàng", "hoàn tiền", "15 ngày", "cửa sổ"]):
        citations.extend(
            [
                "5. Chính sách đổi trả và hoàn tiền > 5.1. Điều kiện chung để gửi yêu cầu",
                "5. Chính sách đổi trả và hoàn tiền > 5.10. Quan hệ giữa trạng thái đơn hàng và quyền trả hàng",
            ]
        )
    if "không hỗ trợ" in text:
        citations.append("5. Chính sách đổi trả và hoàn tiền > 5.3. Các trường hợp không hỗ trợ trả hàng")
    if "kiểm hàng" in text:
        citations.append("4. Chính sách giao hàng > 4.6. Kiểm hàng khi nhận")
    if "giao hàng tiêu chuẩn" in text or "giao tiêu chuẩn" in text:
        citations.append("4. Chính sách giao hàng > 4.3. Thời gian giao hàng dự kiến")
    if "giao nhanh" in text or "giao ưu tiên" in text:
        citations.append("4. Chính sách giao hàng > 4.4. Giao hàng nhanh và giao ưu tiên")
    if "voucher" in text and ("hủy" in text or "huỷ" in text or "hoàn lại" in text):
        citations.append("6. Chính sách voucher và khuyến mãi > 6.5. Hoàn lại voucher khi đơn bị hủy")
    return citations


def _enrich_policy_query(question: str) -> str:
    hints = " ".join(_topic_citations(question))
    return f"{question}\n{hints}".strip()


def _keyword_policy_hits(
    policy_store: ChromaPolicyStore,
    question: str,
    top_k: int,
) -> list[dict[str, Any]]:
    try:
        records = policy_store.collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    keywords = _query_keywords(question)
    if not keywords:
        return []

    scored = []
    for chunk_id, document, metadata in zip(
        records.get("ids", []),
        records.get("documents", []),
        records.get("metadatas", []),
        strict=False,
    ):
        haystack = f"{metadata.get('citation', '')} {document}".lower()
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score:
            scored.append((score, chunk_id, document, metadata))

    scored.sort(key=lambda item: item[0], reverse=True)
    hits = []
    for score, chunk_id, document, metadata in scored[:top_k]:
        hits.append(
            {
                "id": chunk_id,
                "citation": metadata.get("citation", ""),
                "section_h2": metadata.get("section_h2", ""),
                "section_h3": metadata.get("section_h3", ""),
                "content": metadata.get("content", document),
                "rendered_text": document,
                "distance": 1.0 / (score + 1),
            }
        )
    return hits


def _query_keywords(question: str) -> list[str]:
    text = question.lower()
    keywords = []
    keyword_groups = {
        "trả hàng": ["trả hàng", "hoàn trả", "đổi trả", "15 ngày", "cửa sổ"],
        "giao": ["giao hàng", "giao tiêu chuẩn", "giao nhanh", "giao ưu tiên"],
        "voucher": ["voucher", "hoàn lại", "hủy đơn", "huỷ đơn"],
        "kiểm hàng": ["kiểm hàng", "ngoại quan"],
    }
    for trigger, group in keyword_groups.items():
        if trigger in text or any(term in text for term in group):
            keywords.extend(group)
    return _merge_unique([], keywords)


def _merge_policy_hits(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for hit in [*primary, *secondary]:
        hit_id = hit.get("id") or hit.get("citation")
        if hit_id in seen:
            continue
        seen.add(hit_id)
        merged.append(hit)
    return merged


def _invoke_tool(tools: dict[str, Any], name: str, args: dict[str, Any]) -> dict[str, Any]:
    return tools[name].invoke(args)


def _format_order_fact(order: dict[str, Any]) -> str:
    return (
        f"Đơn {order.get('order_id')}: trạng thái {order.get('order_status')}, "
        f"dự kiến giao {order.get('estimated_delivery')}, delivered_at={order.get('delivered_at')}, "
        f"eligible_for_return_until={order.get('eligible_for_return_until')}, "
        f"can_return_now={order.get('can_return_now')}."
    )


def _format_customer_fact(customer: dict[str, Any]) -> str:
    return (
        f"Khách hàng {customer.get('customer_id')} ({customer.get('customer_name')}) hạng {customer.get('tier')}, "
        f"max_voucher_per_month={customer.get('max_voucher_per_month')}, "
        f"vouchers_used_this_month={customer.get('vouchers_used_this_month')}, "
        f"remaining_voucher_quota_this_month={customer.get('remaining_voucher_quota_this_month')}."
    )


def _format_orders_list_fact(orders: list[dict[str, Any]]) -> str:
    return f"Danh sách đơn gần đây: {_orders_short_text(orders)}."


def _format_vouchers_fact(vouchers: list[dict[str, Any]], only_active: bool) -> str:
    label = "Voucher còn dùng được" if only_active else "Voucher của khách hàng"
    return f"{label}: {_vouchers_short_text(vouchers)}."


def _orders_short_text(orders: list[dict[str, Any]]) -> str:
    if not orders:
        return "không có đơn hàng"
    return ", ".join(
        f"{order.get('order_id')}({order.get('order_status')})"
        for order in orders[:10]
    )


def _vouchers_short_text(vouchers: list[dict[str, Any]]) -> str:
    if not vouchers:
        return "không có mã phù hợp"
    return ", ".join(
        f"{voucher.get('voucher_code')}({voucher.get('status')}, còn {voucher.get('remaining_uses')} lượt)"
        for voucher in vouchers[:10]
    )


def _orders_from_data_result(data_result: dict[str, Any]) -> list[dict[str, Any]]:
    orders = []
    for result in data_result.get("tool_results", []):
        if result.get("status") == "ok" and "order" in result:
            orders.append(result["order"])
    return orders


def _orders_list_from_data_result(data_result: dict[str, Any]) -> list[dict[str, Any]]:
    for result in data_result.get("tool_results", []):
        if result.get("status") == "ok" and "orders" in result:
            return result["orders"]
    return []


def _customers_from_data_result(data_result: dict[str, Any]) -> list[dict[str, Any]]:
    customers = []
    for result in data_result.get("tool_results", []):
        if result.get("status") == "ok" and "customer" in result:
            customers.append(result["customer"])
    return customers


def _vouchers_from_data_result(data_result: dict[str, Any]) -> list[dict[str, Any]]:
    for result in data_result.get("tool_results", []):
        if result.get("status") == "ok" and "vouchers" in result:
            return result["vouchers"]
    return []


def _asks_policy(question: str) -> bool:
    text = question.lower()
    return any(
        term in text
        for term in [
            "chính sách",
            "policy",
            "quy định",
            "hoàn trả",
            "trả hàng",
            "hoàn tiền",
            "kiểm hàng",
            "giao hàng tiêu chuẩn",
            "giao nhanh",
            "giao ưu tiên",
            "không hỗ trợ",
            "cửa sổ",
            "15 ngày",
            "từ chối nhận",
            "phương thức giao",
        ]
    ) or ("voucher" in text and any(term in text for term in ["hủy", "huỷ", "hoàn lại"]))


def _explicit_policy_request(question: str) -> bool:
    text = question.lower()
    return any(term in text for term in ["policy", "chính sách", "quy định", "cửa sổ", "15 ngày"])


def _asks_order_policy_mix(question: str) -> bool:
    text = question.lower()
    return any(
        term in text
        for term in [
            "hoàn trả",
            "trả hàng",
            "hoàn tiền",
            "đổi ý",
            "từ chối nhận",
            "cửa sổ",
            "15 ngày",
            "policy",
            "chính sách",
        ]
    )


def _asks_for_customer_orders(question: str) -> bool:
    text = question.lower()
    return any(term in text for term in ["danh sách đơn", "những đơn", "đơn nào", "orders"])


def _asks_for_vouchers(question: str) -> bool:
    return "voucher" in question.lower()


def _asks_for_active_vouchers(question: str) -> bool:
    text = question.lower()
    return any(term in text for term in ["còn", "dùng được", "active", "mã nào"])


def _asks_for_customer_quota(question: str) -> bool:
    text = question.lower()
    return any(term in text for term in ["hạng", "quota", "tối đa", "hạn mức"])


def _extract_order_ids(question: str) -> list[str]:
    customer_ids = set(_extract_customer_ids(question))
    candidates = ORDER_ID_RE.findall(question)
    return [
        candidate
        for candidate in _merge_unique([], candidates)
        if candidate.upper() not in customer_ids
    ]


def _extract_customer_ids(question: str) -> list[str]:
    return _merge_unique([], [match.upper() for match in CUSTOMER_ID_RE.findall(question)])


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _status_from_state(state: dict[str, Any]) -> str:
    route = state.get("route", {})
    data_result = state.get("data_result", {})
    if route.get("status") == "clarification_needed":
        return "clarification_needed"
    if data_result.get("status") == "clarification_needed":
        return "clarification_needed"
    if data_result.get("status") == "not_found":
        return "not_found"
    return "ok"


def _shorten_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _merge_unique(prefix: list[str], values: list[str]) -> list[str]:
    merged = []
    seen = set()
    for value in [*prefix, *values]:
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _runtime(name: str) -> Any:
    try:
        return _GRAPH_RUNTIME[name]
    except KeyError as exc:
        raise RuntimeError(f"Graph runtime is missing required value: {name}") from exc
