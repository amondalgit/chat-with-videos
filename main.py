# LLMs
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredHTMLLoader

from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# Streamlit
import streamlit as st

# Scraping
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# YouTube
from langchain.document_loaders import YoutubeLoader
# !pip install youtube-transcript-api

import json
# Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()

with open('api_keys.json', 'r') as json_file:
    OPENAI_API_KEY = json.load(json_file)["openai"]


# Load up your LLM
def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.1, model_name='gpt-3.5-turbo')
    return llm

# A function that will be called only if the environment's openai_api_key isn't set
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

def pull_from_website(url):
    st.write("Getting webpages...")
    # Doing a try in case it doesn't work
    try:
        response = requests.get(url)
    except:
        # In case it doesn't work
        print ("Whoops, error")
        return
    
    # Put your response in a beautiful soup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get your text
    text = soup.get_text()

    # Convert your html to markdown. This reduces tokens and noise
    text = md(text)
     
    return text

# Pulling data from YouTube in text form
def get_video_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript

def get_transcript(path):
    speech = ''
    for file in os.listdir(path):
        filepath = os.path.join(path,file)
        with open(filepath,'r') as f:
            speech += f"\n{f.read()}"
    return speech

# Function to change our long text about a person into documents
def split_text(user_information):
    # First we make our text splitter
    text_splitter = RecursiveCharacterTextSplitter(separators = ['##','\n','. '],chunk_size = 10000, chunk_overlap=2000)

    # Then we split our user information into different documents
    docs = text_splitter.create_documents([user_information])

    return docs

# Prompts - We'll do a dynamic prompt based on the option the users selects
# We'll hold different instructions in this dictionary below
response_types = {
    'Interview Questions' : """
        Your goal is to generate interview questions that we can ask to the interviewer
        Please respond with list of a few interview questions based on the contexts below
    """,
    '1-Page Bullet Summary' : """
        Your goal is to generate summary in bullet points about the interview
        Please respond with paragraphs that would prepare someone to talk to context below
    """
}

# map_prompt = """You are a helpful AI bot that aids a user in research.
# Below is information about an institute named {institute_name}

# {response_type}

# % START OF INFORMATION ABOUT {institute_name}:
# {text}
# % END OF INFORMATION ABOUT {institute_name}:

# YOUR RESPONSE:"""
# map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text","institute_name", "response_type"])

combine_prompt = """
You are a helpful AI bot that aids a user in research.
Below is information about an organization named {institute_name}
The information consists of interview transcripts, which were done with the persons in the organization administration.
Also scrapped data from the website of that organization might be attacthed in that information.
Do not make anything up, only use information which is in the institute's context

{response_type}

%  CONTEXT
{text}

% YOUR RESPONSE:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text","institute_name", "response_type"])

# Start Of Streamlit page
st.set_page_config(page_title="LLM for Interview Assistance", page_icon=":robot:")

# Start Top Information
st.header("LLM for Interview Assistance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Have an interview coming up? This tool is meant to help you generate \
                interview questions based off of topics they've talked about Or summary of the interview(s).")
with col2:
    st.image(image='Researcher.png', width=300, caption='Mid Journey: A researcher who is really good at their job and utilizes twitter to do research about the person they are interviewing. playful, pastels. --ar 4:7')
# End Top Information

st.markdown("## 👱🏽‍♂️ The LLM Researcher")

# Output type selection by the user
output_type = st.radio(
    "Output Type:",
    ('Interview Questions', '1-Page Bullet Summary'))

# Collect information about the person you want to research
institute_name = st.text_input(label="Organization Name",  placeholder="CMI, Coriolis Technologies", key="institute_name")
#twitter_handle = st.text_input(label="Twitter Username",  placeholder="@eladgil", key="twitter_user_input")
#youtube_videos = st.text_input(label="YouTube URLs (Use , to seperate videos)",  placeholder="Ex: https://www.youtube.com/watch?v=c_hO_fjmMnk, https://www.youtube.com/watch?v=c_hO_fjmMnk", key="youtube_user_input")
text_path = st.text_input(label="Text file directory",  placeholder="Absolute or relative path to the interview transcripts", key="path")
webpages = st.text_input(label="Web Page URLs (Use , to seperate urls. Must include https://)",  placeholder="https://eladgil.com/", key="webpage_user_input")

# Output
st.markdown(f"### {output_type}:")

# Get URLs from a string
def parse_urls(urls_string):
    """Split the string by comma and strip leading/trailing whitespaces from each URL."""
    return [url.strip() for url in urls_string.split(',')]

# Get information from those URLs
def get_content_from_urls(urls, content_extractor):
    """Get contents from multiple urls using the provided content extractor function."""
    return "\n".join(content_extractor(url) for url in urls)

button_ind = st.button("*Generate Output*", type='secondary', help="Click to generate output based on information")

# Checking to see if the button_ind is true. If so, this means the button was clicked and we should process the links
if button_ind:
    if not (webpages):
        st.warning('Please provide links to parse', icon="⚠️")
        st.stop()

    if not OPENAI_API_KEY:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

    if OPENAI_API_KEY == 'YourAPIKeyIfNotSet':
        # If the openai key isn't set in the env, put a text box out there
        OPENAI_API_KEY = get_openai_api_key()

    # Go get your data
    # video_text = get_content_from_urls(parse_urls(youtube_videos), get_video_transcripts) if youtube_videos else ""
    website_data = get_content_from_urls(parse_urls(webpages), pull_from_website) if (webpages) else ""
    video_text = get_transcript(text_path)
    user_information = "\n".join([video_text, website_data])

    user_information_docs = split_text(user_information)

    # Calls the function above
    llm = load_LLM(openai_api_key=OPENAI_API_KEY)

    chain = load_summarize_chain(llm,
                                    chain_type="map_reduce",
                                    # map_prompt=map_prompt_template,
                                    combine_prompt=combine_prompt_template,
                                    verbose=True
                                    )

    st.write("Sending to LLM...")

    # Here we will pass our user information we gathered, the persons name and the response type from the radio button
    output = chain({"input_documents": user_information_docs, # The seven docs that were created before
                    "institute_name": institute_name,
                    "response_type" : response_types[output_type]
                    })

    st.markdown(f"#### Output:")
    st.write(output['output_text'])