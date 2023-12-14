import os 
import streamlit as st
from streamlit_chat import message
from langchain.llms import AzureOpenAI
#Vector database
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
# 五种memory类型
# 可参照https://cookbook.langchain.com.cn/docs/langchain-conversational-memory/ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationTokenBufferMemory

# Convert raw text to vector embedding engine
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
# Streaming callbacks
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def generate_response(llm,retriever,memory,question):
    # 将查询改为带memory的查询
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    answer = qa.run(question)
    return answer

openai_api_key = "5d4d1d30e5704cc3b2812f8304e432fe" 

llm = AzureOpenAI(
    openai_api_key=openai_api_key, 
    deployment_name="testchat",
    model_name="gpt-35-turbo",
    openai_api_version="2023-07-01-preview",
    temperature=0.7, 
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()]
    )

#Embedding Engine
embeddings = AzureOpenAIEmbeddings(
    openai_api_key=openai_api_key,
    azure_endpoint="https://pubwebpoc.openai.azure.com/",
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-07-01-preview",
    model = "text-embedding-ada-002"
    )

vectordb = Chroma(embedding_function=embeddings, persist_directory="./cut")
# 初始化memory,可设置memory为不同类型,可以比较那种效果好
memory = ConversationBufferMemory(return_messages=True)
# memory=ConversationBufferWindowMemory(k=6,return_messages=True)
# memory = ConversationSummaryMemory(llm=llm,return_messages=True)
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100,return_messages=True)
# memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=100,return_messages=True)
st.title("VLK ClientCenter ChatBot")
#storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
user_input = st.text_input("VLK ClientCenter ChatBot",key='input')
if user_input:
    #create search engine
    retriever=vectordb.as_retriever()
    output=generate_response(llm,retriever,user_input)
    #store the output
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
