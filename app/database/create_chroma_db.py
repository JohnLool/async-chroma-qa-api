from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
import shutil

from app.config import settings


CHROMA_PATH = settings.CHROMA_PATH
DATA_PATH = "data"

#
# Run only if you want to create new ChromaDB with your data!
#

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Downloaded {len(documents)} documents.")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
        add_start_index=True,
        is_separator_regex=True
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["doc_type"] = "academic_book"
        chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    example = chunks[0]
    print("Chunk example:")
    print(example.page_content)
    print(example.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
            "show_progress": True
        }
    )

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:M": 32
        }
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
