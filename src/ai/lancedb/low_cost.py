import os
import streamlit as st
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableSequence
from groq import Groq
import tempfile

# Configuration
db_url = "data/low-cost"
# Example context limit for Groq in tokens or characters.
GROQ_CONTEXT_LIMIT = 8192

# Prompt templates
template1 = """The user provided the following question:
"{question}"

Rewrite this question to be more specific and clear, while preserving its original intent. return a question and nothing more. dont add ur thoughts on how or why you did it. Just the Question

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

# Initialize models
model_ollama = ChatOllama(model="gemma2:2b")
embeddings_ollama = OllamaEmbeddings(model="mxbai-embed-large")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


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
            # Extract page content from the document
            doc_content = doc.page_content if hasattr(
                doc, "page_content") else str(doc)
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
        """Mimic callable behavior to process prompts."""
        # Extract the question
        if hasattr(inputs, "to_string"):
            question = inputs.to_string()
        elif isinstance(inputs, str):
            question = inputs
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

        # Retrieve and rank contexts dynamically
        if hasattr(inputs, "get"):
            context_docs = inputs.get("context", [])
        else:
            # If inputs is a ChatPromptValue, retrieve context manually
            context_docs = self.retriever.get_relevant_documents(question)

        best_context = self._select_best_context(
            context_docs, question, self.context_limit
        )

        print("<----------------------------------------->")
        print(f"--------------->: {best_context}\n\n{question}")
        print("<----------------------------------------->")

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


# Streamlit UI
st.title("Document and CSV Q&A System")
service = st.radio("Choose the AI Service", options=[
                   "Ollama", "Groq"], index=0)

uploaded_file = st.file_uploader(
    "Choose a text or CSV file", type=["txt", "csv"])

vector_store = LanceDB(
    uri=db_url,
    table_name="some_documents",
    embedding=embeddings_ollama,  # Default embeddings from Ollama
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


if uploaded_file is not None:
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

# Select model dynamically
if service == "Ollama":
    selected_model = model_ollama
elif service == "Groq":
    selected_model = GroqModelWrapper(groq_client, retriever)

if question:
    # Chain 1: Rewrite Question
    chain1 = (
        {"question": RunnablePassthrough()}
        | prompt1
        | selected_model
        | StrOutputParser()
    )

    # Chain 2: Retrieve Context and Generate Answer
    chain2 = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt2
        | selected_model
        | StrOutputParser()
    )

    # Combined Chain: Rewrite -> Retrieve & Answer -> Summarize
    final_chain = RunnableSequence(
        chain1,
        RunnableMap({"context": chain2, "question": RunnablePassthrough()}),
        prompt,
        selected_model,
        StrOutputParser(),
    )

    # Invoke the combined chain
    result = final_chain.invoke({"question": question})
    st.write(f"Final Result ({service}):", result)
