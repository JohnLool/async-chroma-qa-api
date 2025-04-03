import asyncio

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


CHROMA_PATH = settings.CHROMA_PATH


class ChromaSearch:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            }
        )
        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

    async def search(self, question: str, k=4, threshold=0.7):
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.db.similarity_search_with_relevance_scores(question, k=k)
            )

            filtered = [res for res in results if res[-1] >= threshold]
            return filtered or None
        except Exception as e:
            print(f"Search error: {str(e)}")
            raise