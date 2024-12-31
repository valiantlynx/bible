import marimo

__generated_with = "0.10.8"
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
    import pandas as pd

    load_dotenv(
        "/Users/gormery/Desktop/projects/bible/src/ai/lancedb/.env"
    )  # take environment variables from .env
    return Groq, load_dotenv, mo, os, pd


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
    print(llm_response)
    return chat_completion, llm_response


@app.cell
def _(pd):
    bible_df = pd.read_csv("web.csv")
    bible_df.head(15000)
    return (bible_df,)


@app.cell
def _(bible_df):
    # Create a concatenated string for each verse
    bible_text_with_metadata = bible_df.apply(
        lambda row: f"{row['Book Name']} {row['Chapter']}:{row['Verse']} - {row['Text']}",
        axis=1
    )

    # Convert to a normal array of strings
    bible_array = bible_text_with_metadata.tolist()[:1000]
    bible_array
    return bible_array, bible_text_with_metadata


@app.cell
def _(bible_array, os, pd):

    import requests
    import json
    import numpy as np
    from sklearn.decomposition import PCA

    # URL and headers for API request
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}",  # Fixed quoting issue
    }

    # Function to save progress
    def save_progress(processed_indices, embeddings, progress_file="progress.json"):
        with open(progress_file, "w") as f:
            json.dump({
                "processed_indices": list(processed_indices),  # Convert set to list for JSON
                "embeddings": embeddings
            }, f)

    # Function to load progress
    def load_progress(progress_file="progress.json"):
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress = json.load(f)
            return set(progress["processed_indices"]), progress["embeddings"]  # Convert back to set
        return set(), []

    # Load previous progress
    processed_indices, embeddings = load_progress()

    # Embedding loop with tracking
    batch_size = 5  # Adjust to fit context limits
    for i in range(0, len(bible_array), batch_size):
        batch = bible_array[i : i + batch_size]

        # Skip already processed indices
        if all(idx in processed_indices for idx in range(i, i + len(batch))):
            continue

        # Input data with text samples for embedding
        data = {
            "model": "jina-embeddings-v3",
            "task": "text-matching",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": batch,
        }

        # Send request
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, Response: {response.text}")
            break  # Exit loop on error (e.g., hitting API limit)

        # Parse and store embeddings
        output = json.loads(response.text)["data"]
        embeddings += [entry["embedding"] for entry in output]

        # Update processed indices
        processed_indices.update(range(i, i + len(batch)))

        # Save progress
        save_progress(processed_indices, embeddings)

        print(f"Processed batch {i} to {i + len(batch)}")

    # Convert the list of embeddings into a numpy array (2D)
    embeddings_array = np.array(embeddings)

    # Print the shape of embeddings_array to verify it's 2D (samples x features)
    print(f"Shape of embeddings_array: {embeddings_array.shape}")

    # Apply PCA for dimensionality reduction (reduce to 2D)
    pca = PCA(n_components=2, whiten=True)
    pca_result = pca.fit_transform(embeddings_array)

    # Print the PCA results
    print(f"PCA Result: {pca_result}")

    embedding_plot = pd.DataFrame(
        {
            "x": pca_result[:, 0],
            "y": pca_result[:, 1],
            "text": bible_array,
        }
    ).reset_index()

    print(embedding_plot)

    return (
        PCA,
        batch,
        batch_size,
        data,
        embedding_plot,
        embeddings,
        embeddings_array,
        headers,
        i,
        json,
        load_progress,
        np,
        output,
        pca,
        pca_result,
        processed_indices,
        requests,
        response,
        save_progress,
        url,
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


@app.cell
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
