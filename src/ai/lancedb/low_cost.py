import os
import streamlit as st
import pandas as pd
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableSequence
from groq import Groq
from transformers import AutoTokenizer, AutoModel
import torch
import tempfile
import requests

# Configuration
db_url = "data/low-cost"
# Example context limit for Groq in tokens or characters.
GROQ_CONTEXT_LIMIT = 8192
FIXED_EMBEDDING_SIZE = 384  # Standardized size for embeddings

# Hugging Face Embedding Model
hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModel.from_pretrained(hf_model_name)

# Prompt templates
template1 = """The user provided the following question:
"{question}"

Rewrite this question to be more specific and clear, while preserving its original intent.

Rewritten Question:
"""
prompt1 = ChatPromptTemplate.from_template(template1)

template2 = """Answer the question based only on the following context. Answer it fully and without thinking of length. Make sure you have all the facts:
{context}

Question: {question}
"""
prompt2 = ChatPromptTemplate.from_template(template2)

template = """The following question was asked:
{question}

The following response was given:
{context}

Summarize this answer to include only the key points relevant to answering user queries without omitting any information.

Summary:
"""
prompt = ChatPromptTemplate.from_template(template)


def resize_embedding(embedding, size=FIXED_EMBEDDING_SIZE):
    """Resize or truncate embeddings to a fixed size."""
    embedding = np.array(embedding)
    if len(embedding) > size:
        return embedding[:size]
    elif len(embedding) < size:
        return np.pad(embedding, (0, size - len(embedding)))
    return embedding


def get_hf_embeddings(texts):
    """Get embeddings for a list of texts using Hugging Face."""
    inputs = hf_tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    outputs = hf_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return [resize_embedding(embedding) for embedding in embeddings]


def get_jina_embeddings(texts):
    """Get embeddings for a list of texts using the Jina free API."""
    API_URL = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY', 'your_jina_api_key')}"
    }
    response = requests.post(API_URL, json={"inputs": texts}, headers=headers)
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        return [resize_embedding(embedding) for embedding in embeddings]
    else:
        st.error("Error retrieving embeddings from Jina API.")
        return [np.zeros(FIXED_EMBEDDING_SIZE).tolist()] * len(texts)


def get_ollama_embeddings(texts):
    """Get embeddings for a list of texts using Ollama."""
    embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
    embeddings = embeddings_model.embed_documents(texts)
    return [resize_embedding(embedding) for embedding in embeddings]


class EmbeddingWrapper:
    """A wrapper class to provide embed_documents and embed_query methods."""

    def __init__(self, embed_function):
        self.embed_function = embed_function

    def embed_documents(self, texts):
        return self.embed_function(texts)

    def embed_query(self, text):
        return self.embed_function([text])[0]


class GroqModelWrapper:
    """A wrapper to mimic LangChain's model interface for Groq API with dynamic context selection."""

    def __init__(
        self,
        groq_client,
        retriever,
        model_name="llama3-8b-8192",
        context_limit=GROQ_CONTEXT_LIMIT,
    ):
        self.client = groq_client
        self.retriever = retriever
        self.model_name = model_name
        self.context_limit = context_limit

    def _select_best_context(self, context_docs, question, available_tokens):
        """Select the best context dynamically to fit within the token limit."""
        scored_contexts = []
        for doc in context_docs:
            doc_content = getattr(doc, "page_content", str(doc))
            score = self._score_context(doc_content, question)
            scored_contexts.append((score, doc_content))

        # Sort contexts by relevance score (descending)
        scored_contexts.sort(reverse=True, key=lambda x: x[0])

        # Combine contexts until the token limit is reached
        selected_contexts = []
        total_length = 0
        for _, context in scored_contexts:
            context_length = len(context)
            if total_length + context_length + len(question) <= available_tokens:
                selected_contexts.append(context)
                total_length += context_length
            else:
                break

        return "\n\n".join(selected_contexts)

    def _score_context(self, context, question):
        """Score a context document based on relevance to the question (simple heuristic)."""
        return sum(1 for word in question.split() if word.lower() in context.lower())

    def __call__(self, inputs):
        """Ensure inputs are always converted to a dictionary and processed."""
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary.")
        question = inputs.get("question", "")
        context_docs = inputs.get("context", [])

        best_context = self._select_best_context(
            context_docs, question, self.context_limit
        )

        # Build the final message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{best_context}\n\n{question}"},
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq API Error: {e}")


# Initialize Ollama Model
model_ollama = ChatOllama(model="gemma2:2b")

# Streamlit UI
st.title("Document and CSV Q&A System")

embedding_service = st.radio(
    "Choose the Embedding Service", options=["Hugging Face", "Ollama", "Jina"], index=0
)
model_service = st.radio("Choose the AI Service", options=[
                         "Ollama", "Groq"], index=0)

uploaded_file = st.file_uploader(
    "Choose a text or CSV file", type=["txt", "csv"])

embedding_service_sanitized = embedding_service.replace(" ", "_").lower()

if embedding_service == "Hugging Face":
    embedding_function = EmbeddingWrapper(get_hf_embeddings)
elif embedding_service == "Ollama":
    embedding_function = EmbeddingWrapper(get_ollama_embeddings)
elif embedding_service == "Jina":
    embedding_function = EmbeddingWrapper(get_jina_embeddings)

vector_store = LanceDB(
    uri=db_url,
    table_name=f"documents_{embedding_service_sanitized}",
    embedding=embedding_function,
)
retriever = vector_store.as_retriever()


def process_csv_with_loader(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name

        loader = CSVLoader(file_path=temp_path, csv_args={"delimiter": ","})
        documents = loader.load()
        return [doc.page_content for doc in documents]
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return []


if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "txt":
        string_data = uploaded_file.getvalue().decode("utf-8")
        splitted_data = string_data.split("\n\n")
        vector_store.add_texts(splitted_data)
        st.success("Text file processed and added to the vector store.")

    elif file_type == "csv":
        csv_data = process_csv_with_loader(uploaded_file)
        if csv_data:
            vector_store.add_texts(csv_data)
            st.success("CSV file processed and added to the vector store.")

question = st.text_input("Input your question for the uploaded document")

if question:
    if model_service == "Ollama":
        selected_model = model_ollama
    elif model_service == "Groq":
        selected_model = GroqModelWrapper(
            groq_client=Groq(api_key=os.getenv("GROQ_API_KEY")),
            retriever=retriever,
            model_name="llama3-8b-8192",
            context_limit=GROQ_CONTEXT_LIMIT,
        )

    chain1 = (
        {"question": RunnablePassthrough()}
        | prompt1
        | selected_model
        | StrOutputParser()
    )

    chain2 = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt2
        | selected_model
        | StrOutputParser()
    )

    final_chain = RunnableSequence(
        chain1,
        RunnableMap({"context": chain2, "question": RunnablePassthrough()}),
        prompt,
        selected_model,
        StrOutputParser(),
    )

    try:
        # Convert input to a dictionary for safety
        input_data = {"question": question}
        result = final_chain.invoke(input_data)
        st.write(
            f"Final Result ({model_service} Model, {
                embedding_service} Embeddings):",
            result,
        )
    except Exception as e:
        st.error(f"Error during processing: {e}")
