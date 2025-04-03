import asyncio
from openai import AsyncOpenAI

from app.config import settings
from app.prompt import PROMPT_TEMPLATE
from app.database.chroma_search import ChromaSearch
from app.schemas import QuestionRequest, QuestionResponse


async def get_openrouter_response(client: AsyncOpenAI, prompt: str, max_retries=3) -> str:
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )

            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content

            print(f"Пустой ответ, попытка {attempt + 1}/{max_retries}")

        except Exception as e:
            print(f"Ошибка API: {str(e)}, попытка {attempt + 1}/{max_retries}")

        await asyncio.sleep(2 ** attempt)

    return "Не удалось получить ответ от нейросети"


async def process_question(question: QuestionRequest) -> QuestionResponse:
    try:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )

        chroma = ChromaSearch()
        results = await chroma.search(question.text, question.max_sources, question.similarity_threshold)

        if not results:
            return {"answer": "Информация по вашему запросу не найдена."}

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt = PROMPT_TEMPLATE.format(context=context_text, question=question.text)

        response_text = await get_openrouter_response(client, prompt)

        if "информация не найдена" in response_text.lower() or "не удалось получить ответ от нейросети" in response_text.lower():
            return {"answer": "Информация по вашему запросу не найдена."}

        sources = [f"{doc.metadata['source']} (стр. {doc.metadata.get('page', 'N/A')})"
                   for doc, _ in results]

        response = {
            "answer": response_text,
            "sources": sources
        }

        return QuestionResponse.model_validate(response)

    except Exception as e:
        return {"error": f"Критическая ошибка: {str(e)}"}

