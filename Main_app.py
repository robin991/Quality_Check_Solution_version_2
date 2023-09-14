import streamlit as st
import base64




def add_bg_from_local(image_file,main_bg_ext):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)
    
    
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{main_bg_ext};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
 

import pandas as pd 
import io

# for pandas ai
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import PandasAI




def page_configuration() -> None:
    st.set_page_config(
    page_title = "RIO-CoPilot",
    layout="wide")

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

def read_file(file_name : str) -> pd.DataFrame: 
    # Function to read file and return and dataframe and Fileuploader object
    df_uploader = st.file_uploader("✳️Upload your file here", type = ['csv'])

    if df_uploader is not None : 
        df = pd.read_csv(df_uploader)

        st.session_state['df'] = df
        return st.session_state['df'], df_uploader
    else:
        return pd.DataFrame(), None

def get_df_info(df):
    '''
    Function to display df.info() details in dataframe format
    '''
    buffer = io.StringIO ()
    df.info (buf=buffer)
    lines = buffer.getvalue ().split ('\n')
    # lines to print directly
    lines_to_print = [0, 1, 2, -2, -3]
    for i in lines_to_print:
        st.write (lines [i])
    # lines to arrange in a df
    list_of_list = []
    for x in lines [5:-3]:
        list = x.split ()
        list_of_list.append (list)
    info_df = pd.DataFrame (list_of_list, columns=['index', 'Column', 'Non-null-Count', 'null', 'Dtype'])
    info_df.drop (columns=['index', 'null'], axis=1, inplace=True)
    st.dataframe(info_df)

def get_df_info_text(df) -> None:
    '''
    Function to display df.info() details in text format
    '''
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

def display_uploaded_data() -> None:

    # display uploaded file
    st.dataframe(st.session_state['df'], width = 1100)

def display_uploaded_data_info() -> None:
    data = st.session_state['df']

    # display info():
    get_df_info_text(data)

    # display describe:
    st.dataframe(data.describe().T, width = 1100)
    
def chat_bot_Pandasai_api() -> None:
    '''
    This function uses Pandasai library using Open AI to converse with uploaded file data.
    The key is available in .env file against OPENAI_API_KEY
    '''    
    # checking is the .env file exits ( only incase of running file locally) else extract API from streamlit interface
    check_file = os.path.isfile('.env')

    if check_file:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

    else:
        openai_api_key = st.secrets["API_KEY"]

        
    def chat_with_csv(df,prompt):
        llm = OpenAI(api_token=openai_api_key)
        pandas_ai = PandasAI(llm)
        result = pandas_ai.run(df, prompt=prompt)
        print(result)
        return result

    
    input_text = st.text_area("Enter your query")

    if input_text is not None:
        if st.button("Chat"):
            st.info("Your Query: "+input_text)
            result = chat_with_csv(st.session_state['df'], input_text)
            st.success(result)



    return None 

import streamlit as st
import base64





# page setting configuration
page_configuration()

# setting backgorund
main_bg = "Amex background6.jpg"
main_bg_ext = "jpg"
add_bg_from_local(main_bg,main_bg_ext) 

#initializing session state variables
initialize_session_state()

# adding page title
st.markdown("<h1 style='text-align: center; color: Black;'>RIO-CoPilot</h1>", unsafe_allow_html=True)


# subheader
st.subheader( 'Our webpage-integrated Chat Bot Solution, part of a proof-of-concept for the Reporting and Insights Team, offers a user-friendly chat interface for streamlined data access.It employs Natural Language Processing (NLP) for easy communication, allows customization to fit team-specific needs, and delivers insights while integrating seamlessly with existing tools. This scalable solution enhances productivity and data-driven decision-making for your team.',
            )
st.divider()
col1, col2 = st.columns(2)

with col1 :
    #Read file from column  and  File uploader object
    st.session_state['df'],st.session_state['File_uploader_object'] = read_file(file_name = "File1")
    st.write("[Sample file download link](https://github.com/robin991/Quality_Check_Solution/blob/main/data/tips.csv)")

    if not st.session_state['df'].empty:
        
        # assign to session state
        uploaded_file = st.session_state['File_uploader_object']

        with st.expander('Uploaded Data Display'):
            # Function call : display uploaded file on application
            display_uploaded_data()

        with st.expander("Uploaded Data Info"):
            # Function call : display basic details regarding the dataset
            display_uploaded_data_info()

        with st.expander("Regular BAU"):
            # Function call : dispalay regular BAU analysis
            st.write('Analyse')

with col2 :
    st.title("Chat Template")
    if not st.session_state['df'].empty:
        chat_bot_Pandasai_api()
    else:

        st.error("Kindly Upload your file!")