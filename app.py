import os
import streamlit as st
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import platform

# Configuración de la página
st.set_page_config(page_title="Generación Aumentada por Recuperación (RAG)", layout="wide")

# Título y descripción de la aplicación
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Imagen de encabezado
image = Image.open('Chat_pdf.png')
st.image(image, width=350)

# Sidebar con instrucciones
with st.sidebar:
    st.subheader("Instrucciones")
    st.write("Este agente te ayudará a realizar análisis sobre el PDF cargado.")
    ke = st.text_input('🔑 Ingresa tu Clave de API de OpenAI', type="password")

# Carga del archivo PDF
pdf = st.file_uploader("📥 Carga el archivo PDF", type="pdf")

# Extracción y análisis del texto del PDF
if pdf is not None:
    # Leer el PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Dividir el texto en fragmentos
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)

    # Crear embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Entrada de la pregunta del usuario
    st.subheader("📝 Escribe qué quieres saber sobre el documento:")
    user_question = st.text_area("")

    # Procesar la pregunta
    if user_question:
        if ke:  # Verificar si la clave de API está presente
            with st.spinner("🔄 Procesando tu consulta..."):
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(model_name="gpt-4o-mini")
                chain = load_qa_chain(llm, chain_type="stuff")

                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
                st.write("### Respuesta:")
                st.success(response)  # Mostrar la respuesta
        else:
            st.warning("⚠️ Por favor, ingresa tu clave de API para continuar.")
else:
    st.warning("⚠️ Carga un archivo PDF para comenzar.")
