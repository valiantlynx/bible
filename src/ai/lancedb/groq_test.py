import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Embedding Visualizer""")
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
    return Groq, load_dotenv, mo, os


@app.cell
def _(Groq, os):
    client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    return (client,)


@app.cell
def _(client):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of low latency LLMs",
            }
        ],
        model="llama3-8b-8192",
    )
    llm_response = chat_completion.choices[0].message.content
    llm_response
    return chat_completion, llm_response


@app.cell
def _(os):
    import pandas as pd
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    import requests
    import numpy as np
    from sklearn.decomposition import PCA
    import json

    # Jina API setup
    url = "https://api.jina.ai/v1/embeddings"
    JINA_API_KEY = os.environ.get('JINA_API_KEY')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }

    # Load Bible data with explicit column names, skipping the header row
    bible_df = pd.read_csv(
        "web.csv",
        names=["Verse ID", "Book Name", "Book Number", "Chapter", "Verse", "Text"],
        skiprows=1  # Skip the header row since column names are explicitly defined
    )

    # Ensure correct data types
    bible_df["Verse ID"] = bible_df["Verse ID"].astype(int)
    bible_df["Book Number"] = bible_df["Book Number"].astype(int)
    bible_df["Chapter"] = bible_df["Chapter"].astype(int)
    bible_df["Verse"] = bible_df["Verse"].astype(int)

    # Create concatenated metadata text
    bible_df["full_text"] = bible_df.apply(
        lambda row: f"{row['Book Name']} {row['Chapter']}:{row['Verse']} - {row['Text']}",
        axis=1
    )

    # LanceDB setup
    db_path = "bible_lancedb"
    db = lancedb.connect(db_path)

    # Define LanceDB schema for embeddings
    class BibleSchema(LanceModel):
        verse_id: int
        book_name: str
        chapter: int
        verse: int
        text: str
        full_text: str
        embedding: Vector(1024)  # Assuming 1024 dimensions from Jina model

    # Create or open the LanceDB table
    table = db.create_table("bible_embeddings", schema=BibleSchema, exist_ok=True)

    # Verify LanceDB contents
    print(f"Initial rows in LanceDB: {len(table)}")

    # Function to get unembedded rows
    def get_unembedded_rows(df, table):
        # Extract existing IDs from LanceDB
        existing_ids = {row.verse_id for row in table.to_pandas().itertuples()}
        # Filter the DataFrame to exclude already embedded rows
        return df[~df["Verse ID"].isin(existing_ids)]

    # Dynamically fetch unembedded rows
    unembedded_df = get_unembedded_rows(bible_df, table)
    print(f"Unembedded rows to process: {len(unembedded_df)}")

    # Optional limit for testing or controlled processing
    limit = None  # Set to None to process all rows
    if limit:
        unembedded_df = unembedded_df[:limit]

    # Embedding loop with tracking
    batch_size = 5
    for i in range(0, len(unembedded_df), batch_size):
        # Dynamically fetch unembedded rows for each batch
        batch = unembedded_df.iloc[i : i + batch_size]
        print(f"Processing batch {i} - {i + len(batch)} of {len(unembedded_df)}")

        texts = batch["full_text"].tolist()

        # Prepare API request payload
        data = {
            "model": "jina-embeddings-v3",
            "task": "text-matching",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": texts,
        }

        # Send request to Jina API
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, Response: {response.text}")
            break
        print(response.status_code, "-", json.loads(response.text)["usage"]["total_tokens"])

        # Parse embeddings
        output = response.json().get("data", [])
        embeddings = [entry["embedding"] for entry in output]

        # Check for response consistency
        if len(embeddings) != len(batch):
            print(f"Warning: Mismatched response. Expected {len(batch)}, got {len(embeddings)}")
            continue  # Skip the batch

        # Prepare rows for LanceDB
        rows = [
            {
                "verse_id": int(row["Verse ID"]),
                "book_name": str(row["Book Name"]),
                "chapter": int(row["Chapter"]),
                "verse": int(row["Verse"]),
                "text": str(row["Text"]),
                "full_text": str(row["full_text"]),
                "embedding": [float(e) for e in embeddings[idx]],
            }
            for idx, (_, row) in enumerate(batch.iterrows())
        ]

        # Insert embeddings into LanceDB if rows are valid
        try:
            table.add(rows)
            print(f"Inserted {len(rows)} rows into LanceDB.")
        except Exception as e:
            print(f"Error inserting rows: {e}")
            print("Skipping this batch...")

    # Verify LanceDB contents
    print(f"Total rows in LanceDB: {len(table)}")


    # Query LanceDB for embeddings
    all_embeddings = np.array([row.embedding for row in table.to_pandas().itertuples()])
    print(f"Embeddings shape: {all_embeddings.shape}")

    # PCA for dimensionality reduction
    pca = PCA(n_components=2, whiten=True)
    pca_result = pca.fit_transform(all_embeddings)

    # Visualization data
    embedding_plot = pd.DataFrame(
        {
            "x": pca_result[:, 0],
            "y": pca_result[:, 1],
            "full_text": [row.full_text for row in table.to_pandas().itertuples()],
        }
    )
    print(embedding_plot)
    return (
        BibleSchema,
        JINA_API_KEY,
        LanceModel,
        PCA,
        Vector,
        all_embeddings,
        batch,
        batch_size,
        bible_df,
        data,
        db,
        db_path,
        embedding_plot,
        embeddings,
        get_unembedded_rows,
        headers,
        i,
        json,
        lancedb,
        limit,
        np,
        output,
        pca,
        pca_result,
        pd,
        requests,
        response,
        rows,
        table,
        texts,
        unembedded_df,
        url,
    )


@app.cell
def _(mo, table):
    # Fetch all data from LanceDB and convert to a DataFrame
    db_data = table.to_pandas()
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
def _(chart, mo, raw_digits, table_ui):
    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table_ui
    mo.stop(not len(chart.value))

    def show_images(indices, max_images=10):
        import matplotlib.pyplot as plt

        indices = indices[:max_images]
        images = raw_digits.reshape((-1, 8, 8))[indices]
        fig, axes = plt.subplots(1, len(indices))
        fig.set_size_inches(12.5, 1.5)
        if len(indices) > 1:
            for im, ax in zip(images, axes.flat):
                ax.imshow(im, cmap="gray")
                ax.set_yticks([])
                ax.set_xticks([])
        else:
            axes.imshow(images[0], cmap="gray")
            axes.set_yticks([])
            axes.set_xticks([])
        plt.tight_layout()
        return fig

    selected_images = (
        show_images(list(chart.value["index"]))
        if not len(table_ui.value)
        else show_images(list(table_ui.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table_ui}
        """
    )
    return selected_images, show_images


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
                x=alt.X("x:Q").scale(domain=(-2.5, 2.5)),
                y=alt.Y("y:Q").scale(domain=(-2.5, 2.5)),
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


if __name__ == "__main__":
    app.run()
