import argparse
import os
import time
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ответь на вопрос строго на русском языке, используя ТОЛЬКО предоставленный контекст.
Если в контексте нет нужной информации, ТАК И ТОЛЬКО ТАК ПИШИ: "Информация не найдена".
В случае, если информация найдена - В ОТВЕТЕ ИЗБЕГАЙ СЛОВОСОЧЕТАНИЯ: "Информация не найдена".

Контекст:
{context}

---

Вопрос: {question}
"""


def get_openrouter_response(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="google/gemini-2.5-pro-exp-03-25:free",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )

            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content

            print(f"Пустой ответ, попытка {attempt + 1}/{max_retries}")

        except Exception as e:
            print(f"Ошибка API: {str(e)}, попытка {attempt + 1}/{max_retries}")

        time.sleep(2 ** attempt)

    return "Не удалось получить ответ от нейросети"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Текст запроса")
    args = parser.parse_args()
    query_text = args.query_text

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            }
        )

        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if not results or results[0][1] < 0.7:
            print("Не удалось найти подходящие результаты.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        response_text = get_openrouter_response(client, prompt)

        if "информация не найдена" in response_text.lower() or "не удалось получить ответ от нейросети" in response_text.lower():
            print("Информация по вашему запросу не найдена.")
            return

        sources = [f"{doc.metadata['source']} (стр. {doc.metadata.get('page', 'N/A')})"
                   for doc, _ in results]

        print(f"Ответ: {response_text}")
        print(f"\nИсточники:\n" + "\n".join(sources))

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")


if __name__ == "__main__":
    main()