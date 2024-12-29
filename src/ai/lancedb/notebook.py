import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    # Import required libraries
    import os
    import pandas as pd
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_community.vectorstores import LanceDB
    from langchain_community.document_loaders import CSVLoader
    from langchain_core.runnables import (
        RunnablePassthrough,
        RunnableMap,
        RunnableSequence,
    )
    from groq import Groq
    import tempfile
    from dotenv import load_dotenv
    import marimo

    load_dotenv(
        "/home/valiantlynx/Desktop/bible/src/ai/lancedb/.env"
    )  # take environment variables from .env

    # Configuration
    db_url = "data/low-cost"
    # Example context limit for Groq in tokens or characters.
    GROQ_CONTEXT_LIMIT = 8192
    return (
        CSVLoader,
        ChatOllama,
        ChatPromptTemplate,
        GROQ_CONTEXT_LIMIT,
        Groq,
        LanceDB,
        OllamaEmbeddings,
        RunnableMap,
        RunnablePassthrough,
        RunnableSequence,
        StrOutputParser,
        db_url,
        load_dotenv,
        marimo,
        os,
        pd,
        tempfile,
    )


@app.cell
def _(ChatPromptTemplate):
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
    return prompt, prompt1, prompt2, template, template1, template2


@app.cell
def _(ChatOllama, Groq, OllamaEmbeddings, os):
    # Initialize models for multiple services
    model_ollama = ChatOllama(model="gemma2:2b")
    embeddings_ollama = OllamaEmbeddings(model="mxbai-embed-large")
    groq_client = Groq(api_key=os.environ.get("groq api key"))
    return embeddings_ollama, groq_client, model_ollama


@app.cell
def _(GROQ_CONTEXT_LIMIT):
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
                doc_content = (
                    doc.page_content if hasattr(
                        doc, "page_content") else str(doc)
                )
                score = self._score_context(doc_content, question)
                scored_contexts.append((score, doc_content))

            scored_contexts.sort(reverse=True, key=lambda x: x[0])

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
            return sum(
                1 for word in question.split() if word.lower() in context.lower()
            )

        def __call__(self, inputs):
            """Process the question and context."""
            if hasattr(inputs, "to_string"):
                question = inputs.to_string()
            elif isinstance(inputs, str):
                question = inputs
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")

            if hasattr(inputs, "get"):
                context_docs = inputs.get("context", [])
            else:
                context_docs = self.retriever.get_relevant_documents(question)

            best_context = self._select_best_context(
                context_docs, question, self.context_limit
            )

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

    return (GroqModelWrapper,)


@app.cell
def _(LanceDB, db_url, embeddings_ollama, marimo):
    # Marimo UI
    marimo.Markdown("# Document and CSV Q&A System")

    service = marimo.Select(
        "Choose the AI Service", options=["Ollama", "Groq"], index=0
    )
    uploaded_file = marimo.FileUpload(
        "Choose a text or CSV file", type=["txt", "csv"])

    vector_store = LanceDB(
        uri=db_url,
        table_name="some_documents",
        embedding=embeddings_ollama,  # Default embeddings from Ollama
    )
    retriever = vector_store.as_retriever()
    return retriever, service, uploaded_file, vector_store


@app.cell
def _(CSVLoader, marimo, tempfile):
    def process_csv_with_loader(file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_file.write(file.getvalue())
                temp_path = temp_file.name

            loader = CSVLoader(file_path=temp_path,
                               csv_args={"delimiter": ","})
            documents = loader.load()
            return [doc.page_content for doc in documents]
        except Exception as e:
            marimo.Error(f"Error processing CSV file: {e}")
            return []

    return (process_csv_with_loader,)


@app.cell
def _(marimo, process_csv_with_loader, uploaded_file, vector_store):
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]

        if file_type == "txt":
            string_data = uploaded_file.getvalue().decode("utf-8")
            splitted_data = string_data.split("\n\n")
            vector_store.add_texts(splitted_data)
            marimo.Success(
                "Text file processed and added to the vector store.")

        elif file_type == "csv":
            csv_data = process_csv_with_loader(uploaded_file)
            if csv_data:
                vector_store.add_texts(csv_data)
                marimo.Success(
                    "CSV file processed and added to the vector store.")
    return csv_data, file_type, splitted_data, string_data


@app.cell
def _(
    GroqModelWrapper,
    groq_client,
    marimo,
    model_ollama,
    retriever,
    service,
):
    question = marimo.TextInput(
        "Input your question for the uploaded document")

    # Select model dynamically
    if service == "Ollama":
        selected_model = model_ollama
    elif service == "Groq":
        selected_model = GroqModelWrapper(groq_client, retriever)
    return question, selected_model


@app.cell
def _(
    RunnableMap,
    RunnablePassthrough,
    RunnableSequence,
    StrOutputParser,
    marimo,
    prompt,
    prompt1,
    prompt2,
    question,
    retriever,
    selected_model,
    service,
):
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
            RunnableMap(
                {"context": chain2, "question": RunnablePassthrough()}),
            prompt,
            selected_model,
            StrOutputParser(),
        )

        # Invoke the combined chain
        result = final_chain.invoke({"question": question})
        marimo.Markdown(f"Final Result ({service}): {result}")
    return chain1, chain2, final_chain, result


if __name__ == "__main__":
    app.run()
