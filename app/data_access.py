from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


class ShoppingDataStore:
    """Mock-data lookup store for customers, orders, and vouchers."""

    def __init__(self, json_path: Path) -> None:
        with Path(json_path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.metadata: dict[str, Any] = data.get("metadata", {})
        self.customers: list[dict[str, Any]] = data.get("customers", [])
        self.orders: list[dict[str, Any]] = data.get("orders", [])
        self.vouchers: list[dict[str, Any]] = data.get("vouchers", [])

        self.customer_by_id = {
            self._normalize_customer_id(customer["customer_id"]): customer
            for customer in self.customers
            if customer.get("customer_id")
        }
        self.order_by_id = {
            self._normalize_order_id(order["order_id"]): order
            for order in self.orders
            if order.get("order_id")
        }

        self.orders_by_customer_id: dict[str, list[dict[str, Any]]] = {}
        for order in self.orders:
            customer_id = order.get("customer_id")
            if not customer_id:
                continue
            normalized_customer_id = self._normalize_customer_id(customer_id)
            self.orders_by_customer_id.setdefault(normalized_customer_id, []).append(order)

        for customer_orders in self.orders_by_customer_id.values():
            customer_orders.sort(key=lambda order: order.get("created_at") or "", reverse=True)

        self.vouchers_by_customer_id: dict[str, list[dict[str, Any]]] = {}
        for voucher in self.vouchers:
            customer_id = voucher.get("customer_id")
            if not customer_id:
                continue
            normalized_customer_id = self._normalize_customer_id(customer_id)
            self.vouchers_by_customer_id.setdefault(normalized_customer_id, []).append(voucher)

    def get_customer_by_id(self, customer_id: str) -> dict[str, Any]:
        normalized_customer_id = self._normalize_customer_id(customer_id)
        customer = self.customer_by_id.get(normalized_customer_id)
        if customer is None:
            return {
                "status": "not_found",
                "entity": "customer",
                "customer_id": customer_id,
            }
        return {
            "status": "ok",
            "customer": customer,
        }

    def get_orders_by_customer_id(self, customer_id: str, limit: int = 10) -> dict[str, Any]:
        normalized_customer_id = self._normalize_customer_id(customer_id)
        if normalized_customer_id not in self.customer_by_id:
            return {
                "status": "not_found",
                "entity": "customer",
                "customer_id": customer_id,
            }

        safe_limit = max(0, limit)
        orders = self.orders_by_customer_id.get(normalized_customer_id, [])[:safe_limit]
        return {
            "status": "ok",
            "customer_id": normalized_customer_id,
            "count": len(orders),
            "orders": orders,
        }

    def get_order_detail_by_order_id(self, order_id: str) -> dict[str, Any]:
        normalized_order_id = self._normalize_order_id(order_id)
        order = self.order_by_id.get(normalized_order_id)
        if order is None:
            return {
                "status": "not_found",
                "entity": "order",
                "order_id": order_id,
            }
        return {
            "status": "ok",
            "order": order,
        }

    def get_vouchers_by_customer_id(
        self,
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        normalized_customer_id = self._normalize_customer_id(customer_id)
        if normalized_customer_id not in self.customer_by_id:
            return {
                "status": "not_found",
                "entity": "customer",
                "customer_id": customer_id,
            }

        vouchers = self.vouchers_by_customer_id.get(normalized_customer_id, [])
        if only_active:
            vouchers = [
                voucher
                for voucher in vouchers
                if voucher.get("status") == "active"
                and int(voucher.get("remaining_uses") or 0) > 0
            ]

        return {
            "status": "ok",
            "customer_id": normalized_customer_id,
            "only_active": only_active,
            "count": len(vouchers),
            "vouchers": vouchers,
        }

    @staticmethod
    def _normalize_customer_id(customer_id: str) -> str:
        return str(customer_id).strip().upper()

    @staticmethod
    def _normalize_order_id(order_id: str) -> str:
        return str(order_id).strip()


def build_data_tools(store: ShoppingDataStore) -> list:
    @tool("get_customer_by_id")
    def get_customer_by_id(customer_id: str) -> dict[str, Any]:
        """Tra cứu thông tin khách hàng bằng mã customer_id, ví dụ C001."""
        return store.get_customer_by_id(customer_id)

    @tool("get_orders_by_customer_id")
    def get_orders_by_customer_id(customer_id: str, limit: int = 10) -> dict[str, Any]:
        """Lấy danh sách đơn hàng gần nhất của một khách hàng theo customer_id."""
        return store.get_orders_by_customer_id(customer_id, limit=limit)

    @tool("get_order_detail_by_order_id")
    def get_order_detail_by_order_id(order_id: str) -> dict[str, Any]:
        """Tra cứu chi tiết một đơn hàng bằng mã order_id, ví dụ 1971."""
        return store.get_order_detail_by_order_id(order_id)

    @tool("get_vouchers_by_customer_id")
    def get_vouchers_by_customer_id(
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        """Lấy voucher của khách hàng; đặt only_active=True để chỉ lấy mã còn dùng được."""
        return store.get_vouchers_by_customer_id(
            customer_id,
            only_active=only_active,
        )

    return [
        get_customer_by_id,
        get_orders_by_customer_id,
        get_order_detail_by_order_id,
        get_vouchers_by_customer_id,
    ]
