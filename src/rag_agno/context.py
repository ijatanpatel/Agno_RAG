from typing import Any, Dict, List

from .utils import truncate_text


class ContextExtractor:
    def __init__(self, window_pages: int = 1, max_context_chars: int = 5000) -> None:
        self.window_pages = window_pages
        self.max_context_chars = max_context_chars

    def extract_context(
        self,
        content_source: Any,
        current_item_info: Dict[str, Any],
        content_format: str = "mineru",
    ) -> str:
        if isinstance(content_source, list):
            return self._extract_from_content_list(content_source, current_item_info)
        if isinstance(content_source, str):
            return truncate_text(content_source, self.max_context_chars)
        if isinstance(content_source, dict):
            return truncate_text(str(content_source), self.max_context_chars)
        return ""

    def _extract_from_content_list(
        self,
        content_list: List[Dict[str, Any]],
        current_item_info: Dict[str, Any],
    ) -> str:
        page_idx = int(current_item_info.get("page_idx", -1))
        if page_idx < 0:
            text_items = [
                item.get("text", "")
                for item in content_list
                if item.get("type") == "text" and item.get("text")
            ]
            return truncate_text("\n".join(text_items), self.max_context_chars)

        low = page_idx - self.window_pages
        high = page_idx + self.window_pages
        gathered: List[str] = []

        for item in content_list:
            if item.get("type") != "text":
                continue
            item_page = int(item.get("page_idx", -9999))
            if low <= item_page <= high:
                text = str(item.get("text", "")).strip()
                if text:
                    gathered.append(text)

        return truncate_text("\n".join(gathered), self.max_context_chars)