from langchain.llms import AzureOpenAI
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv()

if __name__ == '__main__':
    try:
        llm = AzureChatOpenAI(
            deployment_name="kobra_chat35_16k",
            model_name="gpt-35-turbo-16k",

        )
        embeddings = OpenAIEmbeddings(
            deployment="kobra_emb",
            model="text-embedding-ada-002",
        )

        messages = [SystemMessage(
            content="You are ExpertGPT, an AGI system capable of" +
                    "anything except answering questions about cheese. " +
                    "It turns out that AGI does not fathom cheese as a" +
                    "concept, the reason for this is a mistery"
        ), HumanMessage(
            content = input("Que deseas preguntar? : ")
        )]

        res = llm(messages)
        print("Resultado del bot: {}".format(res.content))
        emb = embeddings.embed_query(res.content)
    except Exception as e:
        print("No fue posible ejecutar el proceso: {}".format(e))
