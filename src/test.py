import streamlit as st
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch 

# Global ground truth value
GROUND_TRUTH = "EXPECTED OUTPUT"


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
    return "HUMAN OUTPUT"


def main():
    input_text = "I am happy"
    print ("Input: ", input_text)
    print ("Model Output: ", model_output(input_text))
    print ("Human Output: ", human_output(input_text))

if __name__ == "__main__":
    main()
"""
def app():
    st.title("Input and Output Comparison")
    
    # Input text box with a 100-word limit
    input_text = st.text_area("Enter your sentence (max 100 words):")
    
    if input_text:
        words = input_text.split()
        if len(words) > 100:
            st.error("Your input exceeds 100 words. Please limit your sentence to 100 words.")
        else:
            # Get outputs from the functions based on the input
            ml_out = model_output(input_text)
            human_out = human_output(input_text)
            
            # (Optional) Display the computed outputs for transparency
            st.write("**Computed Model Output:**", ml_out)
            st.write("**Computed Human Output:**", human_out)
            
            # Compare outputs when the button is pressed
            if st.button("Compare Outputs"):
                # Check if human output matches ground truth and model output does NOT match ground truth
                if human_out.strip() == GROUND_TRUTH and ml_out.strip() != GROUND_TRUTH:
                    st.success("You have passed")
                else:
                    st.warning("You have failed, try again")

"""