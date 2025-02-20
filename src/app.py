import streamlit as st

from transformers import AutoTokenizer, AutoModelWithLMHead

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import time
from src.const import OPENAI_API_KEY



# Global ground truth value
GROUND_TRUTH = "love"

# streamlit run --server.fileWatcherType none run.py

def get_emotion(text):
    # Load the tokenizer and model without using the fast tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion", use_fast=False)
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

def model_output(input_text):
    # Use the get_emotion function to obtain the model's output
    return get_emotion(input_text)

def human_output(input_text):
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant"),
        ("user", 
        """
        Read the following text and determine if it conveys any of these emotions:{emotions}.
        If one of the emotions is clearly conveyed, simply output that emotion. Only output a single word and nothing else, that output must be the emotion.
        Text:{text}  
    """)
    ])

    prompt=prompt_template.invoke({"emotions": "anger, disgust, fear, joy, sadness, surprise", "text": input_text})


    model= ChatOpenAI(api_key=OPENAI_API_KEY)
    output=model.invoke(prompt)
    return output.content

def compare_outputs(model_out, human_out):
    # Convert outputs to lowercase and strip whitespace before comparing
    norm_model = model_out.lower().strip()
    norm_human = human_out.lower().strip()
    return (GROUND_TRUTH == norm_human and GROUND_TRUTH != norm_model)


def app():
    # Create an empty container that will hold the entire UI
    container = st.empty()
    
    with container.container():
        st.title("Input and Output Comparison")
        
        # Ensure the text area resets by providing a key and default value.
        input_text = st.text_area("Enter your sentence (max 100 words):", key="input_text", value="")
        
        if input_text:
            if len(input_text.split()) > 100:
                st.error("Your input exceeds 100 words. Please try again.")
            else:
                # Get outputs from the model and human functions.
                human_out = model_output(input_text)
                model_out = human_output(input_text)
                norm_ground = GROUND_TRUTH.lower().strip()
                norm_human = human_out.lower().strip()
                norm_model = model_out.lower().strip()
                
                st.write("**Model Output:**", norm_ground)
                st.write("**Human Output:**", norm_human)
                
                # Normalize outputs and ground truth for comparison.
                
                
                # Output success only if human output equals ground truth and model output does not.
                if compare_outputs(norm_model, norm_human):
                    st.success("Success! Human understands your emotions and the AI Machine is confused!!!.")
                else:
                    if GROUND_TRUTH==norm_model:
                        st.error('AI caught your escape!!!')
                    if GROUND_TRUTH!=norm_human:
                        st.error('Human failed to understand your emotions (Incorrect emotion)!!!')
                
                # Retry button: clear the container and session state, then restart the app.
                if st.button("Retry", key="retry_button"):
                    container.empty()  # Clear the UI container
                    st.session_state.clear()  # Clear all session state variables
                    st.rerun()
                
                # Auto-reset mechanism: initialize the timer if not present.
                if "start_time" not in st.session_state:
                    st.session_state.start_time = time.time()
                
                elapsed = time.time() - st.session_state.start_time
                remaining = max(0, int(30 - elapsed))
                st.info(f"Auto-reset in {remaining} seconds...")
                
                # Auto-reset after 30 seconds by clearing the container and session state.
                if elapsed > 30:
                    container.empty()
                    st.session_state.clear()
                    st.rerun()

if __name__ == "__main__":
    text = "I am happy"
    print(human_output(text))