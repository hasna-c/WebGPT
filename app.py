import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_vectorstore(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="context"),
      ("user", "{input}"),
      ("user", "Generate relevant search query.")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    
def get_conversational_rag(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="context"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever(st.session_state.doc_store)
    conversation_rag_chain = get_conversational_rag(retriever_chain)
    response = conversation_rag_chain.invoke({
        "context": st.session_state.context,
        "input": user_input
    })
    if not response['answer']:
        return "I don't know."
    return response['answer']

# App configuration
st.set_page_config(page_title="Website GPT", page_icon="🔗🦜")
st.title("Interactive Website GPT🔗🦜")

# Sidebar
with st.sidebar:
    st.header("Customize")
    website_url = st.text_input("Enter Website URL")

if website_url is None or website_url == "":
    st.info("Please copy your website URL")

else:
    # Session state initialization
    if "context" not in st.session_state:
        st.session_state.context = [
            AIMessage(content="Hello there! How can I assist you today?"),
        ]
    if "doc_store" not in st.session_state:
        st.session_state.doc_store = get_vectorstore(website_url)    

    # User input
    user_query = st.text_input("Enter a prompt here")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.context.append(HumanMessage(content=user_query))
        st.session_state.context.append(AIMessage(content=response))
        
    # Message
    for message in st.session_state.context:
        if isinstance(message, AIMessage):
            with st.container():
                st.write("Bot:", message.content)
        elif isinstance(message, HumanMessage):
            with st.container():
                st.write("You:", message.content)