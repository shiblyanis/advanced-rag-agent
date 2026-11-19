import uuid
from qdrant_client import QdrantClient, models
from app.config import settings

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.ready = False

    def ensure_collection(self, vector_size: int):
        names = [c.name for c in self.client.get_collections().collections]
        if settings.qdrant_collection not in names:
            self.client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config={"dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)},
            )
        self.ready = True

    def upsert(self, records: list[dict], vectors: list[list[float]]):
        points = []
        for record, vector in zip(records, vectors):
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, record["id"])),
                vector={"dense": vector},
                payload=record,
            ))
        if points:
            self.client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int = 20) -> list[dict]:
        result = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )
        return [{**p.payload, "score": p.score} for p in result.points]
