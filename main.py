import streamlit
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

import os

streamlit.title('Profesor AI')

load_dotenv()

if __name__ == '__main__':
    try:
        llm = AzureChatOpenAI(
            temperature=0.2,
            deployment_name="kobra_chat35",
            model_name="gpt-35-turbo"
        )
        embeddings = OpenAIEmbeddings(
            deployment="kobra_emb",
            model="text-embedding-ada-002",
        )
        documents_pdf = []
        for file in os.listdir("docs"):
            if file.endswith(".pdf"):
                pdf_path = "./docs/" + file
                pdf_loader = PyPDFLoader(pdf_path)
                documents_pdf.extend(pdf_loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100000,
            chunk_overlap=0
        )
        documents = text_splitter.split_documents(documents_pdf)

        #db = Chroma(persist_directory="data\\",embedding_function=embeddings)
        db = Chroma.from_documents(documents, embeddings, persist_directory='data\\')
        db.persist()
        retriever = db.as_retriever()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        qa = RetrievalQA.from_chain_type(llm=llm, memory=ConversationBufferMemory(), chain_type="stuff",
                                         retriever=retriever, input_key="question", verbose=True)

        chat_history = []
        while True:
            query = streamlit.text_input("Que deseas preguntar? : ","Hola")
            if query == "salir" or query == "quit" or query == "q":
                print('Ha salido del sistema!')
                exit()
            result = qa({'question': query})
            streamlit.write(result['result'])

    except Exception as e:
        print("No fue posible ejecutar el proceso: {}".format(e))
