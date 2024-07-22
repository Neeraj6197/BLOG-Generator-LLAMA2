import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

#Function to get the response from the LLAMA model:
def get_llama_response(input_text,no_words,blog_style):
    #laoding the LLAMA model:
    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        config={"max_new_tokens":256,
                                "temperature":0.01})
    
    #Creating a Prompt template:
    template = """
                Write a blog for {blog_style} job profile for a topic {input_text} in
                {no_words} words.
                """
    prompt_template = PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                                     template=template)
    

    #Generating the response:
    formated_prompt = prompt_template.format(blog_style=blog_style,
                                             input_text=input_text,
                                             no_words=no_words)
    response = llm(formated_prompt)
    return response




#creating streamlit UI:
st.set_page_config(page_title="Generate Blogs",
                   layout="centered",
                   initial_sidebar_state="collapsed")

#define a heading:
st.header("Blog Generator")

#input text from user:
input_text = st.text_input("Enter the topic to generate a blog.")

#creating two more fields for number of words and blog style:
col1,col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("Enter the number of words you want.")

with col2:
    blog_style = st.selectbox("Writing the blog for",
                              ("Researchers","Data Scientists","Common People"),
                              index=0)
    
#creating a generate button:
submit = st.button("Generate")

#displaying the response:
if submit:
    output = get_llama_response(input_text,no_words,blog_style)
    st.write(output)
    



