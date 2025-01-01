import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""# Bible Q&A""")
    return


@app.cell
def _(mo):
    mo.md("""This app lets you talk to a bible and ask questions about it.""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "How is this app implemented?": """
        - Your Bible is tokenized into chunks, which are embedded using
        jina embeddings
        - Your question is embedded using the same model.
        - We use an approximate k-nearest neighbor search on the PDF embeddings to
        retrieve relevant chunks.
        - The most relevant chunks are added to the context of your prompt, which
        is processed by a GPT model.
        """
    })
    return


@app.cell
def _():
    import os
    from groq import Groq
    from dotenv import load_dotenv
    import marimo as mo

    load_dotenv(
        "/Users/gormery/Desktop/projects/bible/src/ai/lancedb/.env"
    )  # take environment variables from .env

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY") if os.environ.get("GROQ_API_KEY") else mo.ui.text(label="🤖 Groq Key", kind="password")
    JINA_API_KEY = os.environ.get("JINA_API_KEY") if os.environ.get("JINA_API_KEY") else mo.ui.text(label="🦾 Jina Key", kind="password")

    config = mo.hstack([GROQ_API_KEY,JINA_API_KEY])
    mo.accordion({"⚙️ Config -  here is all thats needed for the notebook to run and function correctly. both of these are FREE!! baby!!": config})
    return GROQ_API_KEY, Groq, JINA_API_KEY, config, load_dotenv, mo, os


@app.cell
def _(Groq):
    client = Groq() # uses the default api key in the environment
    return (client,)


@app.cell
def _(mo):
    SYSTEM = """
        You will recieve verses of the bible.
        Answer the subsequent question using that context.
        If you dont know just say you dont know
    """
    mo.md(
        f"""
        We set a system message to determine how our agent Model will behave.
        
        This is the system message:
        **{SYSTEM}**
        """
    )
    return (SYSTEM,)


@app.cell
def _(mo):
    mo.md(
        f"""
        ⚡️✨Test this chat below to see ✨⚡️
        """
    )
    return


@app.cell
def _(BibleSchema, SYSTEM, bible_table, client, mo):
    # Define the function to query the Bible table and Groq model
    def bible_query_model(messages, config):
        """
        Custom RAG (Retrieval-Augmented Generation) model for querying the Bible.

        Args:
            messages (List[ChatMessage]): The chat history, including the user question.
            config (ChatModelConfig): The configuration for the LLM.

        Returns:
            str: The LLM-generated response.
        """
        print(messages[0])
        # Extract the latest user message
        user_message = messages[-1].content

        # Helper function to extract and sort context from LanceDB rows
        def extract_context(rows):
            return sorted(
                [{"full_text": r.full_text, "verse_id": r.verse_id} for r in rows],
                key=lambda x: x["verse_id"],
            )

        # Query the Bible table for context
        rows = bible_table.search(user_message).limit(100).to_pydantic(BibleSchema)
        context = extract_context(rows)

        if not context:
            return "No relevant context found in the database."

        # Prepare the context and question for the Groq model
        groq_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {user_message}"},
        ]

        # Query the Groq model
        response = client.chat.completions.create(
            messages=groq_messages,
            model="llama3-8b-8192",
            stream=False,
        )

        # Stream and collect the response chunk by chunk
        # print("**Groq Model Response:**\n")
        # response_text = ""
        # for chunk in stream:
        #    if hasattr(chunk, "choices") and chunk.choices:
        #        for choice in chunk.choices:
        #            if choice.delta and choice.delta.content:
        #                print(choice.delta.content, end="", flush=True)
        #                response_text += choice.delta.content

        return response.choices[0].message.content

    # Configure the Marimo chat UI with the Bible query model
    chat = mo.ui.chat(bible_query_model)

    # Display the chat UI
    chat

    return bible_query_model, chat


@app.cell
def _(BibleSchema, SYSTEM, bible_table, client, mo, user_question):
    mo.md(
        f"""
        We can also stream but i was not able to put it in marimo chat ui
        """
    )
    if not user_question.value or not bible_table:
        mo.md("No question provided or bible_table is empty.")

    # Helper function to extract and sort context
    def extract_context(rows):
        """Extract and sort context from the database rows."""
        return sorted(
            [{"full_text": r.full_text, "verse_id": r.verse_id} for r in rows],
            key=lambda x: x["verse_id"],
        )

    # Query the table for the context
    rows = bible_table.search(user_question.value).limit(100).to_pydantic(BibleSchema)
    context = extract_context(rows)

    if not context:
        mo.md("No relevant context found in the database.")

    # Create a chat completion request with streaming enabled
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {user_question.value}"},
        ],
        model="llama3-8b-8192",
        stream=True,  # Enable streaming
    )

    # Stream and collect the response chunk by chunk
    print("**Groq Model Response:**\n")
    response_text = ""
    for chunk in stream:
        if hasattr(chunk, "choices") and chunk.choices:
            for choice in chunk.choices:
                if choice.delta and choice.delta.content:
                    print(choice.delta.content, end="", flush=True)
                    response_text += choice.delta.content

    print("\n\n**Response streaming complete.**")
    mo.md(response_text)
    return (
        choice,
        chunk,
        context,
        extract_context,
        response_text,
        rows,
        stream,
    )


@app.cell
def _(mo):
    # Integration with Marimo UI
    user_question = mo.ui.text_area(
        placeholder="💬 Ask your question about the Bible..."
    ).form()
    user_question
    return (user_question,)


@app.cell
def _(os):
    import pandas as pd
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.embeddings import EmbeddingFunctionRegistry
    from tqdm import tqdm
    import requests


    # Initialize the Jina embedder through LanceDB
    registry = EmbeddingFunctionRegistry.get_instance()
    jina_embedder = registry.get("jina").create() # uses the api key in the environment

    # Confirm embedding dimensions from Jina
    EMBEDDING_DIM = jina_embedder.ndims()
    print(f"Using embedding dimension: {EMBEDDING_DIM}")

    # Define schema for Bible embeddings
    class BibleSchema(LanceModel):
        text: str = jina_embedder.SourceField()
        embedding: Vector(EMBEDDING_DIM) = jina_embedder.VectorField()
        verse_id: int
        book_name: str
        chapter: int
        verse: int
        full_text: str

    # LanceDB setup
    db_path = "bible_lancedb_bob"
    db = lancedb.connect(db_path)
    bible_table = db.create_table("bible_embeddings", schema=BibleSchema, exist_ok=True)

    # Load Bible data
    bible_df = pd.read_csv(
        "web.csv",
        names=["Verse ID", "Book Name", "Book Number", "Chapter", "Verse", "Text"],
        skiprows=1,  # Skip the header row since column names are explicitly defined
    )

    # Ensure correct data types and add concatenated metadata text
    bible_df["Verse ID"] = bible_df["Verse ID"].astype(int)
    bible_df["Book Number"] = bible_df["Book Number"].astype(int)
    bible_df["Chapter"] = bible_df["Chapter"].astype(int)
    bible_df["Verse"] = bible_df["Verse"].astype(int)
    bible_df["full_text"] = bible_df.apply(
        lambda row: f"{row['Book Name']} {row['Chapter']}:{row['Verse']} - {row['Text']}",
        axis=1,
    )

    # Log current database status
    existing_rows = bible_table.to_pandas()
    print(f"Total rows in the database: {len(existing_rows)}")
    existing_ids = set(existing_rows["verse_id"])
    print(f"Already embedded rows: {len(existing_ids)}")

    # Determine unembedded rows
    unembedded_bible_data = []
    for _, row in tqdm(bible_df.iterrows(), total=len(bible_df), desc="Checking embedding status"):
        if row["Verse ID"] not in existing_ids:
            unembedded_bible_data.append(
                {
                    "text": str(row["Text"]),
                    "verse_id": int(row["Verse ID"]),
                    "book_name": str(row["Book Name"]),
                    "chapter": int(row["Chapter"]),
                    "verse": int(row["Verse"]),
                    "full_text": str(row["full_text"]),
                }
            )

    print(f"Unembedded rows to process: {len(unembedded_bible_data)}")

    # Embed unembedded rows if there are any
    if unembedded_bible_data:
        print(f"Embedding {len(unembedded_bible_data)} new rows...")
        for batch_start in tqdm(range(0, len(unembedded_bible_data), 100), desc="Embedding batches"):
            batch = unembedded_bible_data[batch_start:batch_start + 100]

            # Prepare texts for embedding
            texts = [row["full_text"] for row in batch]
            try:
                # Get embeddings from Jina
                data = {"model": "jina-embeddings-v3", "task": "text-matching", "input": texts}
                response = requests.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {os.environ['JINA_API_KEY']}"},
                    json=data,
                )
                if response.status_code != 200:
                    print(f"Error: {response.status_code}, Response: {response.text}")
                    continue

                embeddings = response.json().get("data", [])
                if len(embeddings) != len(batch):
                    print(f"Embedding count mismatch. Expected {len(batch)}, got {len(embeddings)}")
                    continue

                # Add embeddings to LanceDB
                for idx, row in enumerate(batch):
                    # Validate embedding dimensions
                    embedding = embeddings[idx]["embedding"]
                    if len(embedding) != EMBEDDING_DIM:
                        print(f"Error embedding batch starting at index {batch_start}: Length of item not correct.")
                        continue

                    row["embedding"] = [float(e) for e in embedding]
                bible_table.add(batch)

            except Exception as e:
                print(f"Error embedding batch starting at index {batch_start}: {e}")
    else:
        print("No new rows to embed.")

    # Final database status
    final_rows = bible_table.to_pandas()
    print(f"Final rows in the database: {len(final_rows)}")
    print(f"Newly embedded rows: {len(final_rows) - len(existing_rows)}")
    return (
        BibleSchema,
        EMBEDDING_DIM,
        EmbeddingFunctionRegistry,
        LanceModel,
        Vector,
        batch,
        batch_start,
        bible_df,
        bible_table,
        data,
        db,
        db_path,
        embedding,
        embeddings,
        existing_ids,
        existing_rows,
        final_rows,
        idx,
        jina_embedder,
        lancedb,
        pd,
        registry,
        requests,
        response,
        row,
        texts,
        tqdm,
        unembedded_bible_data,
    )


@app.cell
def _(bible_table, pd):
    import numpy as np
    from sklearn.decomposition import PCA

    # Query LanceDB for embeddings
    all_embeddings = np.array([row.embedding for row in bible_table.to_pandas().itertuples()])
    print(f"Embeddings shape: {all_embeddings.shape}")

    # PCA for dimensionality reduction
    pca = PCA(n_components=2, whiten=True)
    pca_result = pca.fit_transform(all_embeddings)

    # Visualization data
    embedding_plot = pd.DataFrame(
        {
            "x": pca_result[:, 0],
            "y": pca_result[:, 1],
            "full_text": [row.full_text for row in bible_table.to_pandas().itertuples()],
        }
    )
    embedding_plot
    return PCA, all_embeddings, embedding_plot, np, pca, pca_result


@app.cell
def _(bible_table, mo):
    # Fetch all data from LanceDB and convert to a DataFrame
    db_data = bible_table.to_pandas()
    mo.md(f"**Total Rows in LanceDB: {len(db_data)}**")
    mo.ui.table(db_data)
    return (db_data,)


@app.cell
def _(mo):
    mo.md(
        f"""
        Here's a PCA **embedding of bible verses**: each point represents a
        verse, with similar verses close to each other. The data is from the csv dataset. im thinking of indexing the entire bible.

        This notebook will automatically drill down into points you **select with
        your mouse**; try it!
        """
    )
    return


@app.cell
def _(chart, mo):
    table_ui = mo.ui.table(chart.value)
    return (table_ui,)


@app.cell
def _(alt):
    def scatter(df):
        print(df)
        return (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X("x:Q").scale(),
                y=alt.Y("y:Q").scale(),
                color=alt.Color("full_text:N"),
            )
            .properties(width=500, height=500)
        )
    return (scatter,)


@app.cell
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install("altair")

    import altair as alt
    return alt, micropip, sys


@app.cell
def _(embedding_plot, mo, scatter):
    chart = mo.ui.altair_chart(scatter(embedding_plot))
    chart
    return (chart,)


@app.cell
def _(mo):
    mo.md("""# Bible Q&A""")
    return


@app.cell
def _(mo):
    mo.md("""This app lets you talk to the bible.""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "How is this app implemented?": """
        - The bible is tokenized into chunks, which are embedded using
        Jina's `jina-embeddings-v3`.
        - Your question is embedded using the same model.
        - We use an approximate k-nearest neighbor search on the PDF embeddings to
        retrieve relevant chunks.
        - The most relevant chunks are added to the context of your prompt, which
        is processed by a GPT model.
        """
    })
    return


@app.cell
def _():
    '''
    pdf = mo.ui.file(
        label="Upload PDF", filetypes=[".pdf"], multiple=False, kind="area"
    )
    pdf


    def parse_pdf():
        if not pdf.value:
            print("No PDF")
            return None
        if not pdf.value[0]:
            print("No PDF")
            return None

        contents = pdf.value[0].contents
        file = io.BytesIO(contents)
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openaikey.value)
        return FAISS.from_texts(chunks, embeddings)


    knowledge_base = parse_pdf()
    '''
    return


if __name__ == "__main__":
    app.run()
