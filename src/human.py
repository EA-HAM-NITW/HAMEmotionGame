from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models.openai import ChatOpenAI 
from src.const import OPENAI_API_KEY


input_text='I am happy.'

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", 
     """
    Read the following text and determine if it conveys any of these emotions:{emotions}.
    If one of the emotions is clearly conveyed, simply output that emotion.
    Text:{text}  
""")
])

prompt=prompt_template.invoke({"emotions": "anger, disgust, fear, joy, sadness, surprise", "text": input_text})


model= ChatOpenAI(api_key=OPENAI_API_KEY)
output=model.invoke(prompt)
print(output.content)
