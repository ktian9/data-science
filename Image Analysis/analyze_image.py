import base64
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import traceback

load_dotenv()


class ImageProcessingException(Exception):
    pass

class LlmQueryException(Exception):
    pass


def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    except Exception:
        raise ImageProcessingException("Something went wrong when trying to load the image. Please check if the image exists ", traceback.print_exc())
    

def query_llm(client, model_name, text_prompt, img_64, max_tokens):

    try:
        completion = client.chat.completions.create(model=model_name, messages=[
        
        {
            "role": "user",
            "content": [
                {
                    "type":"text",
                    "text": text_prompt
                
                
                },


                {
                    
                    "type":"image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{img_64}"
                    }
                }
                
                
                ]
        }
        ],
        max_tokens = max_tokens)
        return completion.choices[0].message.content
    except Exception:
        raise LlmQueryException("Something went wrong when trying to query the LLM ", traceback.print_exc())



