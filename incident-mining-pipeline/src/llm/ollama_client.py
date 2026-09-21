"""Client gọi Qwen chạy local qua Ollama.

Dùng chung cho: bootstrap silver-label NER (Layer 2), relation extraction
(Layer 3), và sinh câu trả lời RAG (Layer 5). generate_structured() ép Ollama
trả JSON đúng theo JSON Schema (Ollama hỗ trợ schema đầy đủ qua tham số
`format`, không chỉ chuỗi "json") và tự retry kèm lỗi khi output không hợp lệ.
"""

import json

import httpx


class LLMExtractionError(Exception):
    """Ollama không trả về JSON hợp lệ theo schema sau khi hết số lần retry."""


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _generate(
        self, prompt: str, format_: dict | str | None = None, options: dict | None = None
    ) -> str:
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **({"format": format_} if format_ is not None else {}),
                **({"options": options} if options is not None else {}),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]

    def generate_structured(
        self, prompt: str, schema: dict, max_retries: int = 3, max_output_tokens: int = 1024
    ) -> dict:
        """Gọi Ollama với JSON Schema bắt buộc, parse JSON, retry kèm lỗi nếu
        output không parse được hoặc request timeout. Raise LLMExtractionError
        khi hết retry.

        max_output_tokens chặn model 3B rơi vào vòng lặp sinh vô hạn khi bị
        ép JSON Schema (lặp entity mãi cho tới khi hết timeout)."""
        current_prompt = prompt
        last_error = ""
        for _ in range(max_retries):
            try:
                raw = self._generate(
                    current_prompt, format_=schema, options={"num_predict": max_output_tokens}
                )
            except httpx.HTTPError as exc:
                last_error = f"Request tới Ollama lỗi ({type(exc).__name__}): {exc}"
                continue
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                last_error = f"Output không phải JSON hợp lệ: {exc}. Raw output: {raw[:500]}"
                current_prompt = (
                    f"{prompt}\n\nLần trả lời trước của bạn bị lỗi: {last_error}\n"
                    "Hãy trả lại đúng JSON theo schema, không thêm text nào khác."
                )
        raise LLMExtractionError(
            f"Hết {max_retries} lần thử, vẫn không nhận được JSON hợp lệ. Lỗi cuối: {last_error}"
        )

    def generate_text(self, prompt: str) -> str:
        return self._generate(prompt)
