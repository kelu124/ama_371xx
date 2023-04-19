import openai 
import streamlit as st
import numpy as np
import pandas as pd 
import tiktoken
from streamlit_chat import message
import datetime
import cryptpandas as crp

import pymongo
from pymongo import MongoClient
from libs.commons import *
import glob
st.set_page_config(
    page_title="Questions about ISO371xx",
    page_icon="🫡",
)


DEFQUESTION = "What are green neighbourhoods?"
appName = "iso317xx"

@st.cache(hash_funcs={pd.core.frame.DataFrame: id},\
    allow_output_mutation=True,suppress_st_warning=True)
def loadLargeFilesOnce371xxA():

    st.write("## Please wait, initial embedding loading can take up to a minute")
    AllDF = []
    print("--- loading files")
    for file in glob.glob("data/*csv.gzip"):
        DF = pd.read_csv(file, header=0,compression="gzip")
        AllDF.append(DF)
        print('Loading',file,", ",len(DF))
    print("Loading done")
    AllDF = pd.concat(AllDF)
    cols = list(AllDF.columns)
    cols[0] = "hash"
    AllDF.columns = cols
    df = AllDF
    max_dim = max([int(c) for c in df.columns if c != "hash"])
    print("maxdim done")
    document_embeddings =  { (r.hash): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows() }
    print("embed")
    return document_embeddings

@st.cache(hash_funcs={pd.core.frame.DataFrame: id},\
    allow_output_mutation=True,suppress_st_warning=True)
def loadLargeFilesOnce371xxB():
    st.write("## Please wait, initial data loading can take up to a minute")
    #df = pd.read_parquet("data/srcStrings.parquet.gzip").reset_index()
    df = crp.read_encrypted(path='data/srcStrings.parquet', password=PWD).reset_index()
    df = df[~df.duplicated(subset=["hash"])]
    df = df[list(df.columns)[1:]]
    df["section"] = "" # because no section
    df["IDX"] = df["hash"]
    df["dlv"] = df["article"] # for the dois
    df = df.set_index('IDX')
    return df


document_embeddings = loadLargeFilesOnce371xxA()
df = loadLargeFilesOnce371xxB()

#pwd = st.sidebar.text_input("Password", type="password")
if 1:# pwd == PWD:
    st.title("ISO371xx 'Ask Me Anything' Assistant ")
    st.write("Use this to ask it questions about ISO371xx. Sources will appear in the bottow left side of the screen, or the sidebar if you are on a small screen.")
    st.write('## Chat interface')
    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

        
    st.sidebar.write("### __Parameters__ (Experimental)")    

    if "visibility" not in st.session_state:
        st.session_state.temperature = 0.0
        st.session_state.style = False

    temp = st.sidebar.slider(
        'Creativity',
        0.0, 1.0,value= st.session_state.temperature ,step=0.01)
    st.session_state.temperature = temp
    COMPLETIONS_API_PARAMS["temperature"] = temp


    style = st.sidebar.selectbox(
            "What style ?",
            ("normal","simple", "advanced"), 
        )
    st.session_state.style = style

    st.sidebar.write("* _Crea.:_",COMPLETIONS_API_PARAMS["temperature"])
    st.sidebar.write("* _Style:_",style)

    if style == "simple":
        STYLE = " in the style of a seven-year old, "
    elif style == "advanced":
        STYLE = " in the elaborate style of a management consultant, "
    else:
        STYLE = ""
    #header = "Answer the question as truthfully as possible using the provided context," +STYLE+" and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    header = """Answer the question as truthfully as possible using the provided context,""" +STYLE+""" and add then a new line with the main five keywords, each starting with a hashtag, separated by commas. If the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""


    user_input = get_text(DEFQUESTION)

    if user_input:
        output, dlvs = generate_response(appName, DEFQUESTION,header,user_input+ " Please provide an answer as detailed as possible.",df, document_embeddings,st.session_state.style)
        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


