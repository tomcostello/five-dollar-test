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

if "temperature" not in st.session_state:
    st.session_state.temperature = 1.5

if "top_k" not in st.session_state:
    st.session_state.top_k = 50

if "top_p" not in st.session_state:
    st.session_state.top_p = 0.9

    
if "num_beams" not in st.session_state:
    st.session_state.num_beams= 2

    
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
def get_response(sample, temperature, top_k, top_p, num_beams):
    input_ids = tokenizer(sample, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, 
                             max_new_tokens=500, 
                             do_sample=True, 
                             top_p=top_p,
                             num_beams=num_beams,
                             repetition_penalty=2.5,
                             early_stopping=True,
                             no_repeat_ngram_size=2,
                             use_cache=True,
                             temperature=temperature,
                             top_k = top_k)
    print(f"input sentence: {sample}\n{'---'* 10}")
    return f"Answer:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}"


def get_one_word(sample, temperature, top_k, top_p, num_beams):
    input_ids = tokenizer(sample, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids,
                             max_new_tokens=500,
                             do_sample=True,
                             top_p=top_p,
                             num_beams=num_beams,
                             repetition_penalty=2.5,
                             early_stopping=True,
                             no_repeat_ngram_size=2,
                             use_cache=True,
                             temperature=temperature,
                             top_k = top_k)
    print(f"input sentence: {sample}\n{'---'* 10}")
    return f"Answer:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}"


def generate_stream(
    model, tokenizer, params, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    
    max_src_len = context_len
        
    input_ids = input_ids[-max_src_len:]

    encoder_output = model.encoder(
        input_ids=torch.as_tensor([input_ids], device="auto")
    )[0]
    start_ids = torch.as_tensor(
        [[model.generation_config.decoder_start_token_id]],
        dtype=torch.int64,
        device="auto",
    )
    past_key_values = out = None
    for i in range(max_new_tokens):
        if i == 0:
            out = model.decoder(
                input_ids=start_ids,
                encoder_hidden_states=encoder_output,
                use_cache=True,
            )
            logits = model.lm_head(out[0])
            past_key_values = out.past_key_values
        else:
            out = model.decoder(
                input_ids=torch.as_tensor([[token]], device="auto"),
                encoder_hidden_states=encoder_output,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            logits = model.lm_head(out[0])
            past_key_values = out.past_key_values

            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                    last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            else:
                last_token_logits = logits[0, -1, :]

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))
                
            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0
                    
                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                partially_stopped = False
                if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None
        
        yield {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }
        
        # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    
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

    
    #with st.spinner("Please wait ..."):
    while True:
        resp = generate_stream(model, tokenizer, {"prompt": text,
                                           "temperature": st.session_state.temperature,
                                           "repetition_penalty": 2.0
                                           "top_p": st.session_state.top_p,
                                           "top_k": st.session_state.top_k,
                                           "max_new_tokens": 2048,
                                           "stop": "?",
                                           "echo": True,
                                           "stop_token_ids" : [],
                                           }
                                           , context_len=2048, stream_interval=2)
        
        if resp.get("finish_reason") and resp["finish_reason"] == "stop":
            break

        if resp.get("finish_reason") and resp["finish_reason"] == "length":
            break


        
        st.session_state.text_error = ""
        st.session_state.n_requests += 1
        st.session_state.chat = resp["text"]
        
        logging.info(
            f"""
            Question: {text}
            Answer: {chat_text}
            """
        )

# text
text = st.text_area(label="Enter text", placeholder="Example: does the whitehouse have a pool")

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
st.session_state.temperature = st.number_input('Set temperature. Currently: ' + str(st.session_state.temperature  ), value=1.5)
st.session_state.top_k = st.number_input('Set top_k. Currently: ' + str(st.session_state.top_k ), value=50)
st.session_state.top_p = st.number_input('Set top_p.  Currently: ' + str(st.session_state.top_p), value=0.9)
st.session_state.num_beams = st.number_input('Set num_beams.  Currently: ' + str(st.session_state.num_beams), value=2)
