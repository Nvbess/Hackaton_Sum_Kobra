from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

load_dotenv()

# Ingreso del PDF
pdf_loader = PyPDFLoader('./docs/CV_Infografico.pdf')
data = pdf_loader.load()

#   Transformacion
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 0
)
texts = text_splitter.split_documents(data)

#   Transformamos los chunks en vectores.

embeddings_model = OpenAIEmbeddings()

db = Chroma.from_documents(texts, embeddings_model)
retriever = db.as_retriever()

index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"./data_save"}).from_loaders([pdf_loader])

qa = RetrievalQA.from_chain_type(llm=OpenAI(), memory=ConversationBufferMemory(),chain_type="stuff", retriever=index.vectorstore.as_retriever(), input_key="question", verbose=True)

# Generacion de consulta en consola

while True:
    query = input("Que deseas preguntar? : ")
    if query == "salir" or query == "quit" or query == "q":
        print('Ha salido del sistema!')
        exit()
    response = qa({"question": query})
    print(response['result'])