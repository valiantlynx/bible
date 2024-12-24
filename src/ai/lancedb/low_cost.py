import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableSequence

db_url = "data/low-cost"

template1 = """The user provided the following question:
"{question}"

Rewrite this question to be more specific and clear, while preserving its original intent.

Rewritten Question:
"""

prompt1 = ChatPromptTemplate.from_template(template1)

template2 = """Answer the question based only on the following context. Anwser it fully and without thinking of length make sure you have all the facts:
{context}

Question: {question}
"""

prompt2 = ChatPromptTemplate.from_template(template2)

template = """The following question was asked:
{question}

The following response was given:
{context}

Summarize this answer to include only the key points relevant to answering user queries. without ommiting any information.

Summary:
"""

prompt = ChatPromptTemplate.from_template(template)


model = ChatOllama(model="llava-phi3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

st.write("Hello, Fuckers!")
uploaded_file = st.file_uploader("Choose a text file", type="txt")

vector_store = LanceDB(
    uri=db_url,
    table_name="some_documents",
    embedding=embeddings,
)
retriever = vector_store.as_retriever()


if uploaded_file is not None:
    string_data = uploaded_file.getvalue().decode("utf-8")
    splitted_data = string_data.split("\n\n")

    vector_store.add_texts(splitted_data)

question = st.text_input("Input your question for the uploaded document")

# chain 1: Rewrite Question
chain1 = {"question": RunnablePassthrough(
)} | prompt1 | model | StrOutputParser()

# Chain 2: Retrieve Context and Generate Answer
chain2 = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt2
    | model
    | StrOutputParser()
)

# Combined Chain: Rewrite -> Retrieve & Answer -> Summarize
final_chain = RunnableSequence(
    # Step 1: Rewrite Question
    chain1,
    # Step 2: Retrieve Context and Answer
    RunnableMap({"context": chain2, "question": RunnablePassthrough()}),
    # Step 3: Summarize Answer
    prompt,
    model,
    StrOutputParser(),
)

# Input Handling
if question:
    # Invoke the combined chain
    result = final_chain.invoke({"question": question})
    st.write("Final Result:", result)
