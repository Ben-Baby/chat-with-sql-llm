import streamlit as st
from pathlib import Path
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler

# ---------------------- Streamlit UI Setup ----------------------
st.set_page_config(page_title="Chat with SQL DB", page_icon='💻')
st.title('💬 Chat with Your SQL DB | Powered by LangChain + Groq')
st.markdown("#### Created by **Ben Baby**")

# ---------------------- Sidebar Inputs ----------------------
st.sidebar.header("🔐 Connect to Your MySQL Database")
mysql_host = st.sidebar.text_input('MySQL Host')
mysql_user = st.sidebar.text_input('MySQL Username')
mysql_pass = st.sidebar.text_input('MySQL Password', type='password')
mysql_db = st.sidebar.text_input('MySQL Database Name')

api_key = st.sidebar.text_input('Groq API Key', type='password')

# ---------------------- Validation ----------------------
if not (mysql_host and mysql_user and mysql_pass and mysql_db):
    st.warning("🛑 Please fill all MySQL database details to proceed.")
    st.stop()

if not api_key:
    st.warning("🔑 Please enter your Groq API Key.")
    st.stop()

# ---------------------- LLM Setup (Non-Streaming for Speed) ----------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name='Llama3-8b-8192',
    streaming=False  # Streaming off = faster full response
)

# ---------------------- DB Setup ----------------------
@st.cache_resource(ttl='2h')
def configure_db():
    return SQLDatabase(create_engine(
        f'mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}'
    ))

db = configure_db()

# ---------------------- LangChain Agent Setup ----------------------
@st.cache_resource(ttl='2h')
def get_agent():
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        handle_parsing_errors=True,
        verbose=False,  # Turn off verbose to boost speed
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

agent = get_agent()

# ---------------------- Session State for Chat ----------------------
if 'messages' not in st.session_state or st.sidebar.button('🧹 Clear Chat'):
    st.session_state['messages'] = [{'role': 'assistant', 'content': "👋 How can I help you with your MySQL data today?"}]

# ---------------------- Chat Display ----------------------
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

# ---------------------- User Input and Response ----------------------
user_query = st.chat_input("Ask your question in natural language...")

if user_query:
    st.session_state.messages.append({'role': 'user', 'content': user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
        except Exception as e:
            response = f"❌ Error: {str(e)}"
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)
