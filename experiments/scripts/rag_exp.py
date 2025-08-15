import os
from loguru import logger
import pdfplumber
import ollama
from typing import List
from pathlib import Path


class RagModel:

    def __init__(self, pdf_dir):
        self.EMBEDDING_MODEL: str = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
        self.LANGUAGE_MODEL: str = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
        self.VECTOR_DB: List[tuple] = []
        self.dataset: List = []
        self.pdf_path: str = Path(__file__).parent.parent / "assets" / pdf_dir

    def create_dataset(self):
        pdf_files: List = [f for f in self.pdf_path.glob("*.pdf")]
        for file_path in pdf_files:
            logger.info(f"Processing: {file_path}")
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    self.dataset.append(page.extract_text())
        logger.success(f"Loaded {len(pdf_files)} files contents to the dataset")

    def add_chunk_to_database(self):
        logger.info("Adding chunk to database")
        for i, chunk in enumerate(self.dataset):
            embedding: List[float] = ollama.embed(
                model=self.EMBEDDING_MODEL, input=chunk
            )["embeddings"][0]
            self.VECTOR_DB.append((chunk, embedding))
        logger.success(f"Added chunk {len(self.dataset)} to the database")

    def _cosine_similarity(self, a, b):
        dot_product = sum([x * y for x, y in zip(a, b)])
        norm_a = sum([x**2 for x in a]) ** 0.5
        norm_b = sum([x**2 for x in b]) ** 0.5
        return dot_product / (norm_a * norm_b)

    def retrieve(self, query, top_n=3):
        query_embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=query)[
            "embeddings"
        ][0]
        similarities: List[tuple] = []
        for chunk, embedding in self.VECTOR_DB:
            # calculating the distance
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))

        # sort by similarity in descending order, since higher similarity
        # means more relevant chunks
        similarities.sort(key=lambda x: x[1], reverse=True)

        # return the top N relevant chunks
        return similarities[:top_n]

    def interface(self):
        # Take input from user
        input_query = input("Ask me a question: ")
        self.create_dataset()
        self.add_chunk_to_database()

        # retrieve knowledge and display
        retrieved_knowledge = self.retrieve(query=input_query)
        logger.info("Retrieved knowledge: ")
        for chunk, similarity in retrieved_knowledge:
            logger.info(f"- (similarity: {similarity:.2f}) {chunk}")

        # give context and instruction
        context_text = "\n".join([f"- chunk" for chunk, _ in retrieved_knowledge])

        instruction_prompt = f""""You are a helpful chatbot. Use only the following pieces of context to answer the question. 
        Do not make up any new information. 
        Context: {context_text}
        """

        stream = ollama.chat(
            model=self.LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": input_query},
            ],
            stream=True,
        )

        # print the response from the chatbot in real-time
        logger.info("Chatbot response:")
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)


def main():
    pdf_dir = "pa_slides"
    rag_model = RagModel(pdf_dir=pdf_dir)
    rag_model.interface()


if __name__ == '__main__':
    main()