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
    os.environ.get("GROQ_API_KEY")
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
    print(llm_response)
    return chat_completion, llm_response


@app.cell
def _(os):
    import requests
    import json
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    # Load the Bible data
    bible_df = pd.read_csv("web.csv")
    print(bible_df.head())

    # Ensure all values in the Text column are strings
    bible_df['Text'] = bible_df['Text'].fillna("").astype(str)

    # Prepare concatenated text for chapters and books
    def prepare_hierarchical_text(df):
        # Chapter-level concatenation
        chapter_texts = (
            df.groupby(["Book Name", "Chapter"])["Text"]
            .apply(lambda texts: " ".join(texts))
            .reset_index()
            .rename(columns={"Text": "Chapter Text"})
        )

        # Book-level concatenation
        book_texts = (
            chapter_texts.groupby(["Book Name"])["Chapter Text"]
            .apply(lambda texts: " ".join(texts))
            .reset_index()
            .rename(columns={"Chapter Text": "Book Text"})
        )

        return chapter_texts, book_texts

    chapter_texts, book_texts = prepare_hierarchical_text(bible_df)

    # Chunking function with token validation
    def chunk_text(text, max_length=1000):
        words = text.split()
        chunks = [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
        return chunks

    # Validate and adjust chunks
    def validate_chunks(chunks, max_tokens=8194):
        validated_chunks = []
        for chunk in chunks:
            if len(chunk.split()) > max_tokens:
                validated_chunks.extend(chunk_text(chunk, max_length=max_tokens - 200))
            else:
                validated_chunks.append(chunk)
        return validated_chunks

    # Prepare chunked data with validation
    def prepare_chunked_data(data, metadata=None, max_length=1000):
        chunked_data = []
        chunked_metadata = []

        for idx, text in enumerate(data):
            chunks = chunk_text(text, max_length)
            validated_chunks = validate_chunks(chunks)
            chunked_data.extend(validated_chunks)

            # Add metadata for each chunk
            if metadata:
                chunked_metadata.extend(
                    [f"{metadata[idx]} - Part {i + 1}" for i in range(len(validated_chunks))]
                )
            else:
                chunked_metadata.extend([f"Chunk {idx + 1}" for _ in range(len(validated_chunks))])

        return chunked_data, chunked_metadata

    # Prepare data for embedding with chunking and validation
    verse_data_chunks, verse_metadata = prepare_chunked_data(
        bible_df.apply(
            lambda row: f"{row['Book Name']} {row['Chapter']}:{row['Verse']} - {row['Text']}",
            axis=1
        ).tolist()[:20]
    )

    chapter_data_chunks, chapter_metadata = prepare_chunked_data(
        chapter_texts.apply(
            lambda row: f"{row['Book Name']} Chapter {row['Chapter']} - {row['Chapter Text']}",
            axis=1
        ).tolist()[:20]
    )

    book_data_chunks, book_metadata = prepare_chunked_data(
        book_texts.apply(
            lambda row: f"{row['Book Name']} Overview - {row['Book Text']}",
            axis=1
        ).tolist()[:2],
        metadata=book_texts["Book Name"].tolist()[:2]
    )

    # Combine all data into a single dictionary for processing
    hierarchical_data = {
        "verse": verse_data_chunks,
        "chapter": chapter_data_chunks,
        "book": book_data_chunks
    }

    hierarchical_metadata = {
        "verse": verse_metadata,
        "chapter": chapter_metadata,
        "book": book_metadata
    }

    # URL and headers for API request
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}",
    }

    # Function to save progress
    def save_progress(level, processed_indices, embeddings, progress_file="progress.json"):
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress = json.load(f)
        else:
            progress = {}

        progress[level] = {
            "processed_indices": list(processed_indices),
            "embeddings": embeddings
        }

        with open(progress_file, "w") as f:
            json.dump(progress, f)

    # Function to load progress
    def load_progress(level, progress_file="progress.json"):
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress = json.load(f)
            if level in progress:
                return set(progress[level]["processed_indices"]), progress[level]["embeddings"]
        return set(), []

    # Process each hierarchy level
    all_embeddings = {}

    for level, data in hierarchical_data.items():
        print(f"Processing level: {level}")

        # Load progress for the current level
        processed_indices, embeddings = load_progress(level)

        # Embedding loop with tracking
        batch_size = 5  # Adjust to fit context limits
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            # Skip already processed indices
            if all(idx in processed_indices for idx in range(i, i + len(batch))):
                continue

            # Input data with text samples for embedding
            request_data = {
                "model": "jina-embeddings-v3",
                "task": "text-matching",
                "late_chunking": False,
                "dimensions": 1024,
                "embedding_type": "float",
                "input": batch,
            }

            # Send request
            response = requests.post(url, headers=headers, json=request_data)
            if response.status_code != 200:
                print(f"Error: {response.status_code}, Response: {response.text}")
                continue  # Skip to the next batch instead of breaking

            # Parse and store embeddings
            output = json.loads(response.text)["data"]
            embeddings += [entry["embedding"] for entry in output]

            # Update processed indices
            processed_indices.update(range(i, i + len(batch)))

            # Save progress
            save_progress(level, processed_indices, embeddings)

            print(f"Processed {level} batch {i} to {i + len(batch)}")

        # Store embeddings for this level
        all_embeddings[level] = {"embeddings": embeddings, "texts": hierarchical_metadata[level]}

    # Convert embeddings to numpy arrays for visualization
    for level, embedding_data in all_embeddings.items():
        print(embedding_data)
        embeddings = embedding_data["embeddings"]
        texts = embedding_data["texts"]

        # Ensure alignment of embeddings and texts
        assert len(embeddings) == len(texts), f"Mismatch in {level}: {len(embeddings)} embeddings vs {len(texts)} texts."

        embeddings_array = np.array(embeddings)
        print(f"{level.capitalize()} embeddings shape: {embeddings_array.shape}")

        # Apply PCA for dimensionality reduction
        try:
            pca = PCA(n_components=2, whiten=True)
            pca_result = pca.fit_transform(embeddings_array)

            # Create a DataFrame for visualization
            embedding_plot = pd.DataFrame(
                {
                    "x": pca_result[:, 0],
                    "y": pca_result[:, 1],
                    "text": texts,
                }
            ).reset_index()

            print(f"{level.capitalize()} Embedding Plot:")
            print(embedding_plot[:5])

        except ValueError as e:
            print(f"Error processing PCA for level {level}: {e}")

    return (
        PCA,
        all_embeddings,
        batch,
        batch_size,
        bible_df,
        book_data_chunks,
        book_metadata,
        book_texts,
        chapter_data_chunks,
        chapter_metadata,
        chapter_texts,
        chunk_text,
        data,
        embedding_data,
        embedding_plot,
        embeddings,
        embeddings_array,
        headers,
        hierarchical_data,
        hierarchical_metadata,
        i,
        json,
        level,
        load_progress,
        np,
        output,
        pca,
        pca_result,
        pd,
        prepare_chunked_data,
        prepare_hierarchical_text,
        processed_indices,
        request_data,
        requests,
        response,
        save_progress,
        texts,
        url,
        validate_chunks,
        verse_data_chunks,
        verse_metadata,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        Here's a PCA **embedding of some texts**: each point represents a
        text, with similar texts close to each other. The data is from the hard coded data. im thinking of indexing the entire bible.

        This notebook will automatically drill down into points you **select with
        your mouse**; try it!
        """
    )
    return


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell(hide_code=True)
def _(chart, mo, raw_digits, table):
    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
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
        if not len(table.value)
        else show_images(list(table.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    )
    return selected_images, show_images


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
                color=alt.Color("text:N"),
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
