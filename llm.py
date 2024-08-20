from openai import AzureOpenAI
import os
from langchain_core.prompts import PromptTemplate
import time
from dotenv import load_dotenv
load_dotenv() 


deployment = os.getenv("OPENAI_DEPLOYMENT_NAME")
key = os.getenv("OPENAI_API_KEY")
version = os.getenv("OPENAI_VERSION")
endpoint = os.getenv("OPENAI_ENDPOINT")
llm = AzureOpenAI(azure_endpoint=endpoint, api_key= key, api_version=version)


def query_endpoint(query):
    completion = llm.chat.completions.create(model = deployment, messages=[{"role": "user",
    "content": query}], max_tokens=5000,temperature=0,top_p=0.95,frequency_penalty=0,presence_penalty=0,stop=None,stream=False)

    return completion.choices[0].message.content



def chain(initial_prompt, refine_prompt, documents, time_delay_seconds):

    first_doc = documents[0]
    remaining = documents[1:]

    ## starting with initial response
    prev_response = query_endpoint(initial_prompt.format(context = first_doc.page_content))


    ## sequentially adding onto previous query with new context
    for i in remaining:
        time.sleep(time_delay_seconds)
        prev_response = query_endpoint(refine_prompt.format(prev_response= prev_response, context = i.page_content))

    return prev_response








