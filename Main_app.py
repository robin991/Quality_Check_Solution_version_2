import streamlit as st
import pandas as pd 
#from langchain.prompts import PromptTemplate
#from langchain.llms import CTransformers 
#import pandas as pd 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain 

# for pandas ai
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import PandasAI

def read_file(file_name : str) -> pd.DataFrame: 
    # Function to read file and return and dataframe and Fileuploader object
    df_uploader = st.sidebar.file_uploader("✳️Upload your file here", type = ['csv'])

    if df_uploader is not None : 
        df = pd.read_csv(df_uploader)

        st.session_state['df'] = df
        return st.session_state['df'], df_uploader
    else:
        return pd.DataFrame(), None

def clear_chat_history():
    #st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]   
    st.session_state['past'] = ['Hey! ']
    st.session_state['generated'] = ['Hello, Ask me anything about uploaded file']
    st.session_state['history'] = []

# function for streamlit chat
def conversational_chat(query, chain):
    result = chain({'question':query, "chat_history" : st.session_state['history']}) # passing question along with chat history
    st.session_state['history'].append([query, result['answer']])
    return result['answer']
    


#initializing the session state

def initialize_session_state() -> None:
    if "df" not in st.session_state : 
        st.session_state['df'] = pd.DataFrame()
    if "File_uploader_object" not in st.session_state : 
        st.session_state['File_uploader_object'] = None

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    if "outlier_check_trigger" not in st.session_state : 
        st.session_state['outlier_check_trigger'] = False
    if "chat_bot_llama_trigger" not in st.session_state : 
        st.session_state['chat_bot_llama_trigger'] = False
    if "chat_bot_Pandasai_trigger" not in st.session_state : 
        st.session_state['chat_bot_Pandasai_trigger'] = False

    if "button_outlier_check_trigger" not in st.session_state : 
        st.session_state['button_outlier_check_trigger'] = False

    #session state for chat bot
    if 'history' not in st.session_state : 
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['Hello, Ask me anything about uploaded file']
    
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hey! ']

def display_uploaded_data() -> None:

    # write uploaded file
    st.header('Uploaded Data is :')
    st.dataframe(st.session_state['df'], width = 600)

    return

def performed_task_checklist(uploaded_file)-> None:
    '''
    Provid a check button on sidebar reagrding option for the task to be performed. User can choose from these.
    '''
    st.session_state['outlier_check_trigger'] = st.sidebar.checkbox(f"Perform outlier check on {uploaded_file.name}")
    st.session_state['chat_bot_llama_trigger'] = st.sidebar.checkbox(f"Perform chat with {uploaded_file.name} (local llama)")
    st.session_state['chat_bot_Pandasai_trigger'] = st.sidebar.checkbox(f"Perform chat with {uploaded_file.name} (Pandasai)")

    return

def outlier_check(df : pd.DataFrame) -> None:
    st.write("---")
    st.header("Outlier Check")

    group_col = st.text_input("Goup By Column", "day, time")
    outlier_col = st.text_input("Outlier check column", "tip")

    st.session_state['button_outlier_check_trigger'] = st.button("Run")

    if st.session_state['button_outlier_check_trigger'] :
        
        #Execute the solution if button is clicked
        
        group_col_list = [i.strip() for i in group_col.split(",")]

        #if st.button("Check Outlier"):
        # run if button is clicked
        df_out = df.groupby(by = group_col_list)[outlier_col].sum()

        df_out = df_out.reset_index()
        st.dataframe(df_out)

        st.write(f"This is the mean :{round(df_out.loc[:,outlier_col].mean(),3)}")
        st.write(f"This is the std :{round(df_out.loc[:,outlier_col].std(),3)}")

        # threshold calculation
        thresh = 2*round(df_out.loc[:,outlier_col].std(),3)
        st.write(f"This is the outlier thershold(+/-) :{thresh}")

        st.write("Outlier values:")

        # Final Outlier dataframe
        df_out = df_out[(df_out[outlier_col] <-thresh) | (df_out[outlier_col] >thresh) ]

        # Display the outlier dataframe
        st.dataframe(df_out)

        return None

def load_llm():
        llm = CTransformers(
            model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type = "llama",
            max_new_tokens = 512,
            temperature = 0.9
        )
        return llm


def chat_bot_local_llama() -> None:
    st.write("---")

    # create a clear chat botton on sidebar
    st.sidebar.button("Clear Chat", on_click=clear_chat_history)

    # store embeddings
    DB_FAISS_PATH = "vectorestore/db_faiss"

    # loading the model
    

    st.title("Chat with CSV (LLAMAv2 quantized)")

    # build sidebar to upload the file on streamlit
    #uploaded_file = st.sidebar.file_uploader("Upload CSV File", type = "csv")

    #if uploaded_file :

        
        
    # create a temporary file object
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','}) # csv loader needs  a filepath hence we created a temporary file path

    # data has the csv
    data = loader.load()
    #st.json(data)

    # word embedding model ( vector creation)
    embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs = {'device' : 'cpu'})
    
    db = FAISS.from_documents(data,embeddings)
    # save the db to the path
    db.save_local(DB_FAISS_PATH)

    # load llm model . it will be passed in conversation retrieval chain
    llm = load_llm()

    #chain call
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())            

    # assigning containers for the chat history

    response_container = st.container()
    
    container = st.container()

    with container:
        with st.form(key = "my_form", clear_on_submit = True):
            user_input = st.text_input("Query:", placeholder = "Talk to your CSV Data here", key = 'input')

            submit_buttom = st.form_submit_button(label ='Send')

        if submit_buttom and user_input:
            output = conversational_chat(user_input, chain)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user = True,
                        key = str(i) + '_user',
                        avatar_style = "big-smile")
                message(st.session_state["generated"][i],
                        key = str(i) ,
                        avatar_style = "thumbs")

def chat_bot_Pandasai_api() -> None:
    '''
    This function uses Pandasai library using Open AI to converse with uploaded file data.
    The key is available in .env file against OPENAI_API_KEY
    '''
    st.write("---")
    
    # checking is the .env file exits ( only incase of running file locally) else extract API from streamlit interface
    check_file = os.path.isfile('.env')

    if check_file:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

    else:
        openai_api_key = st.secrets["API_KEY"]

    st.title("Chat with CSV (Pandasai Openai)")
        
    def chat_with_csv(df,prompt):
        llm = OpenAI(api_token=openai_api_key)
        pandas_ai = PandasAI(llm)
        result = pandas_ai.run(df, prompt=prompt)
        print(result)
        return result

      

    st.info("Chat Below")
    
    input_text = st.text_area("Enter your query")

    if input_text is not None:
        if st.button("Chat"):
            st.info("Your Query: "+input_text)
            result = chat_with_csv(st.session_state['df'], input_text)
            st.success(result)



    return None
##---------Begin--------------##
# set initial configuration
st.set_page_config(
    page_title = "DAX solution",
    layout="wide"
)
# initialize session state
initialize_session_state() 


##-------Homepage------##
st.title("AMEX CoPilot Solution")

#Read file from side bar and  File uploader object
st.session_state['df'],st.session_state['File_uploader_object'] = read_file(file_name = "File1")


st.sidebar.write("[Sample file download link](https://github.com/robin991/Quality_Check_Solution/blob/main/data/tips.csv)")

st.sidebar.write("🧑‍💻🧑‍🤝‍🧑Solution presented by : Nikhil's team (Utkarsh, Amit, Hemlata, Robin)")

if not st.session_state['df'].empty:

    # assign to session state
    #df = st.session_state['df']
    uploaded_file = st.session_state['File_uploader_object']


    # Function call : display uploaded file on application
    display_uploaded_data()
    
    # creating check button for Outlier check and chat bot
    performed_task_checklist(uploaded_file)

    ##----Outlier Check-----##
    if st.session_state['outlier_check_trigger']:
        
        #Perform outier check analysis on uploaded file if check button is clicked
        outlier_check(st.session_state['df'])
        

    ## ------chat bot -----##
    if (st.session_state['chat_bot_llama_trigger'] ==  True) and (st.session_state['File_uploader_object'] != None) :
        # chat with uploaded data if the check box is clicked and File_uploader not None ( same as dataframe not empty)
        
        chat_bot_local_llama()
    
    if (st.session_state['chat_bot_Pandasai_trigger'] ==  True) and (st.session_state['File_uploader_object'] != None) :
        # chat with uploaded data if the check box is clicked and File_uploader not None ( same as dataframe not empty)
        
        chat_bot_Pandasai_api()

    

else:
    # if file is not uploaded
    st.error("Kindly upload the file !")