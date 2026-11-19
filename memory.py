from app.config import settings

class MemoryService:
    def __init__(self):
        self.client = None
        if settings.mem0_enabled and settings.mem0_api_key:
            try:
                from mem0 import MemoryClient
                self.client = MemoryClient(api_key=settings.mem0_api_key)
            except Exception:
                self.client = None

    def search(self, user_id: str, query: str) -> str:
        if not self.client:
            return ""
        try:
            result = self.client.search(query, user_id=user_id)
            memories = result.get("results", result) if isinstance(result, dict) else result
            return "\n".join(str(m.get("memory", m)) for m in memories[:5])
        except Exception:
            return ""

    def add(self, user_id: str, text: str):
        if not self.client:
            raise RuntimeError("Mem0 is disabled. Configure MEM0_API_KEY and MEM0_ENABLED=true")
        return self.client.add(text, user_id=user_id)

    def get_all(self, user_id: str):
        if not self.client:
            return []
        try:
            result = self.client.get_all(user_id=user_id)
            return result.get("results", result) if isinstance(result, dict) else result
        except Exception:
            return []
