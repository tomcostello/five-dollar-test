# Import from standard library
import logging

import argparse

# Import from 3rd party libraries
import streamlit as st
import streamlit.components.v1 as components

parser = argparse.ArgumentParser(description='The only argument is --merged_model, whoch should point at the merged model.  Add -- before it as streamlit is like that.')

parser.add_argument('--merged_model', action='store', default="merged-model",
                    help="The name of the merged model")
args = parser.parse_args()
# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)

# Configure Streamlit page and state
st.set_page_config(page_title="Chat Bot", page_icon="🍩")


# Store the initial value of widgets in session state
if "chat" not in st.session_state:
    st.session_state.chat = ""
if "text_error" not in st.session_state:
    st.session_state.text_error = ""

if "n_requests" not in st.session_state:
    st.session_state.n_requests = 0

if "top_k" not in st.session_state:
    st.session_state.top_k = 8

if "alpha" not in st.session_state:
    st.session_state.alpha = 0.6

    
    
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@st.cache_resource
def  get_tokenizer():
    return  AutoTokenizer.from_pretrained("google/flan-ul2")
    
@st.cache_resource
def get_model():
    model_name = args.merged_model
    return AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.bfloat16, device_map="auto")     

tokenizer = get_tokenizer()

model = get_model()

@st.cache_data
def get_response(sample, top_k, alpha):
    input_ids = tokenizer(sample, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, 
                            max_length=500,
                            penalty_alpha=alpha,
                            top_k = top_k,
                            repetition_penalty=2.5,
                            no_repeat_ngram_size=2)
    print(f"input sentence: {sample}\n{'---'* 10}")
    return f"Answer:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}"




# Force responsive layout for columns also on mobile
st.write(
    """
    <style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Render Streamlit page
st.title("A Chatbot")


st.markdown(
    "This mini-app displays our $5 chatbot."
)






def chatbot(text, help):
    """
    Chatbot Text.
    """

    if st.session_state.n_requests >= 5:
        st.session_state.text_error = "Too many requests. Please wait a few seconds."
        logging.info(f"Session request limit reached: {st.session_state.n_requests}")
        st.session_state.n_requests = 1
        return

    st.session_state.chat = ""
    st.session_state.text_error = ""
    st.session_state.n_requests = 0

    if not text:
        st.session_state.text_error = "Please enter a question."
        return

    
    with st.spinner("Please wait ..."):
        
        
        chat_text = get_response(text,
                      st.session_state.top_k,
                      st.session_state.alpha )
        
        st.session_state.text_error = ""
        st.session_state.n_requests += 1
        st.session_state.chat = chat_text
        
        logging.info(
            f"""
            Question: {text}
            Answer: {chat_text}
            """
        )

# text
text = st.text_area(label="Enter text", placeholder="Example: how do you do a blowout")

# chat button
st.button(
    label="Ask Me",
    key="generate",
    help="Press to Ask", 
    type="primary", 
    on_click=chatbot,
    args= ( text , "help" ),
    )

st.markdown(st.session_state.chat)
st.session_state.top_k = st.number_input('Set top_k. Currently: ' + str(st.session_state.top_k ), value=8)
st.session_state.alpha = st.number_input('Set alpha_penalty.  Currently: ' + str(st.session_state.alpha), value=0.6)
