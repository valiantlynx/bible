import marimo

__generated_with = "0.10.8"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from groq import Groq
    from dotenv import load_dotenv
    import marimo as mo
    import pandas as pd

    load_dotenv(
        "/home/valiantlynx/Desktop/bible/src/ai/lancedb/.env"
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
def _(pd, sklearn):
    import requests
    import json
    import numpy as np

    # URL and headers for API request
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer some api key",
    }

    # Input data with text samples for embedding
    data = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "late_chunking": False,
        "dimensions": 1024,
        "embedding_type": "float",
        "input": [
            "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin. Infused with the calming properties of aloe vera and chamomile, each product provides gentle nourishment and protection. Say goodbye to irritation and hello to a glowing, healthy complexion.",
            "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille: Erleben Sie die wohltuende Wirkung unserer Bio-Hautpflege, speziell für empfindliche Haut entwickelt. Mit den beruhigenden Eigenschaften von Aloe Vera und Kamille pflegen und schützen unsere Produkte Ihre Haut auf natürliche Weise. Verabschieden Sie sich von Hautirritationen und genießen Sie einen strahlenden Teint.",
            "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla: Descubre el poder de la naturaleza con nuestra línea de cuidado de la piel orgánico, diseñada especialmente para pieles sensibles. Enriquecidos con aloe vera y manzanilla, estos productos ofrecen una hidratación y protección suave. Despídete de las irritaciones y saluda a una piel radiante y saludable.",
            "针对敏感肌专门设计的天然有机护肤产品：体验由芦荟和洋甘菊提取物带来的自然呵护。我们的护肤产品特别为敏感肌设计，温和滋润，保护您的肌肤不受刺激。让您的肌肤告别不适，迎来健康光彩。",
            "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています: 今シーズンのメイクアップトレンドは、大胆な色彩と革新的な技術に注目しています。ネオンアイライナーからホログラフィックハイライターまで、クリエイティビティを解き放ち、毎回ユニークなルックを演出しましょう。",
            "llm_response",  # Add the actual response text here
        ],
    }

    # Make the API request
    response = requests.post(url, headers=headers, json=data)

    # Check the response status
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
    else:
        # Parse the JSON response and extract embeddings
        output = json.loads(response.text)["data"]

        # Extract the embedding data into a list of lists
        embeddings_list = [entry["embedding"] for entry in output]
        print(f"Embeddings extracted: {embeddings_list}")

        # Convert the list of embeddings into a numpy array (2D)
        data_array = np.array(embeddings_list)

        # Print the shape of data_array to verify it's 2D (samples x features)
        print(f"Shape of data_array: {data_array.shape}")

        # Apply PCA for dimensionality reduction (reduce to 2D)
        pca = sklearn.decomposition.PCA(n_components=2, whiten=True)
        pca_result = pca.fit_transform(data_array)

        # Print the PCA results
        print(f"PCA Result: {pca_result}")

        embedding_plot = pd.DataFrame(
            {
                "x": pca_result[:, 0],
                "y": pca_result[:, 1],
                "text": [entry for entry in data["input"]],
            }
        ).reset_index()

    embedding_plot

    return (
        data,
        data_array,
        embedding_plot,
        embeddings_list,
        headers,
        json,
        np,
        output,
        pca,
        pca_result,
        requests,
        response,
        url,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Embedding Visualizer""")
    return


@app.cell
def _():
    import sklearn
    import sklearn.datasets
    import sklearn.manifold

    raw_digits, raw_labels = sklearn.datasets.load_digits(return_X_y=True)
    return raw_digits, raw_labels, sklearn


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        Here's a PCA **embedding of numerical digits**: each point represents a
        digit, with similar digits close to each other. The data is from the UCI
        ML handwritten digits dataset.

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
def _(pd, raw_digits, raw_labels, sklearn):
    X_embedded = sklearn.decomposition.PCA(n_components=2, whiten=True).fit_transform(
        raw_digits
    )

    embedding = pd.DataFrame(
        {"x": X_embedded[:, 0], "y": X_embedded[:, 1], "digit": raw_labels}
    ).reset_index()
    return X_embedded, embedding


@app.cell
def _(alt):
    def scatter(df):
        return (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X("x:Q").scale(domain=(-2.5, 2.5)),
                y=alt.Y("y:Q").scale(domain=(-2.5, 2.5)),
                color=alt.Color("digit:N"),
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
