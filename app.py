# bring in deps
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


# Customize the layout
st.set_page_config(page_title="Doc_Q/A_ai", page_icon="📑🐛", layout="wide", )
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://img.freepik.com/free-vector/note-paper-background-with-hole-punches_78370-2344.jpg?w=1380&t=st=1693819536~exp=1693820136~hmac=24ee26fcb782a1bc31b36ee722ccc440f94bdc34274bc032eb4d039e781a750d");
                    background-attachment: fixed;
                    background-size: cover}}
            </style>
""", unsafe_allow_html=True)

# function for writing uploaded file in temp
def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occured while writing the file: {e}")
        return False

# set prompt template
Prompt_Template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know with polite tone, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(template=Prompt_Template, input_variables=["context", "question"])


# Initialize the LLM and Embeddings
llm = LlamaCpp(model_path="./models/llama-7b.ggmlv3.q8_0.bin")
embeddings = LlamaCppEmbeddings(model_path="models/llama-7b.ggmlv3.q8_0.bin")
llm_chain = LLMChain(llm=llm, prompt=prompt)

st.title("📑🐛 DocuWise")
uploaded_file = st.file_uploader("Upload an article", type="txt")

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    # st.write(content)
    file_path = "temp/file.txt"
    write_text_file(content, file_path)

    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    db = Chroma.from_documents(texts, embeddings)
    st.success("File loaded Successfully 👍🙂")

    # Query through LLM
    question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the Text?", disabled=not uploaded_file,)
    if question:
        similar_doc = db.similarity_search(question, k = 1)
        context = similar_doc[0].page_content
        query_llm = LLMChain(llm=llm, prompt=prompt)
        response = query_llm.run({"context":context, "question": question})
        st.write(response)