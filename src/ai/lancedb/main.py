import ollama
import pandas as pd
import lancedb
import time
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
from tqdm import tqdm


registry = EmbeddingFunctionRegistry.get_instance()
embedder = registry.get("ollama").create(name="mxbai-embed-large")


df = pd.read_csv("data/sentences.csv")


class Schema(LanceModel):
    text: str = embedder.SourceField()
    vector: Vector(embedder.ndims()) = embedder.VectorField()
    index: int
    title: str
    url: str


db = lancedb.connect(uri="data/sample-lancedb")
table = db.create_table("sentences", schema=Schema, exist_ok=True)
# tqdm(table.add(df))

# how to read data from lancedb
# print(table.to_pandas().iloc[0])


def extract_context(rows):
    return sorted(
        [{"title": r.title, "text": r.text, "index": r.index} for r in rows],
        key=lambda x: x["index"],
    )


SYSTEM = """
    You will recieve paragraphs of text from news article.
    Answer the subsequent question using that context.
    If you dont know just say you dont know
"""


client = ollama.Client()

question = input("Ask some questions from bbc?")
rows = table.search(question).limit(3).to_pydantic(Schema)
context = extract_context(rows)
stream = ollama.chat(
    model="llava-phi3",
    stream=True,
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Question: {question}"},
    ],
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

# how it looks when feched with our schema
print(rows)
