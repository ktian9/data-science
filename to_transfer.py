from langchain_core.prompts import PromptTemplate
import time
from palantir_models.transforms import OpenAiGptChatLanguageModelInput
from palantir_models.models import OpenAiGptChatLanguageModel
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    GptChatCompletionRequest,
)
from language_model_service_api.languagemodelservice_api import (
    ChatMessage,
    ChatMessageRole,
)
from langchain.docstore.document import Document


class custom_chain:
    def __init__(
        self,
        text_id,
        time_delay,
        model: OpenAiGptChatLanguageModel,
    ):
        self.model = model
        self.time_delay = time_delay
        self.text_id = text_id

    def process_documents(self, text):
        print("Handling documents")
        try:
            text = text.split(" ### ")
            docs = []
            for i in text:
                docs.append(Document(page_content=i))

            self.docs = docs

        except Exception as e:
            print("Exeception occurred in process_documents ", e)

    async def create_and_query_chain_dict(self):
        first_document = self.docs[0]
        remaining_documents = self.docs[1:]
        
        prompt = PromptTemplate.from_template(
            """Given a detailed service event report about an engine or vehicle failure, extract the following information from the text and format the result as a python dictionary with JSON formatting standards. This means surround all properties in double quotes instead of single:

        With the key of "model_identified_complaint", extract the primary customer complaint. The complaint is defined as the main engine or vehicle issue that the customer or technician is claiming is causing the repair, and summarize the complaint in less than 4 words, including the exact part or system affected, using technical terms.
If no such information is found, mention "Not available". in the response, only keep the relevant information. Avoid narriative explainations. Only include engine related terms and complaints. Do not count P codes as a complaint. Prioritize the first mentioned complaint, and if there is something mentioned by the customer prioritize that. If the text contains the phrase "maint" or similar to "maintenance", output the complaint as maintenance. For the outputted complaint, fix any spelling mistakes. If the identified complaint is in another language, translate it to English. Remove any quotations and extra punctuation, just keep the alphanumeric characters. Output the result in lowercase.
With the key of "identified_cause", extract the assignable cause of repair if any are mentioned in the text. For example: Belt noise, no communication, no start, sensor malfunctioning. This cause should identify the component or the cause that was fixed that able to fix the issue during the repair.
only keep the relevant information and avoid narrative explanations.
With the key of "identified_repairs", extract and present the repair actions in a short summary, For example: replaced nox sensor, replaced egr cooler, fixed coolant leak, in the response and only keep the relevant information while avoiding narrative explanations.
With the key of "identified_keywords", extract important service, repair or symptom related keywords in a comma separated list. Focus on any objective nouns that are mentioned. Examples: loose connector, burning smell, P codes or U codes.
in the response, only keep the relevant information and avoid any narrative explanations.
With the key of "identified_codes", given a detailed service event report about an engine or vehicle failure, extract ONLY P codes or U codes that start with a P or U and have 4 digits and truncate values before hyphens if any are present. Codes with hyphens like U0100-00 or P0100-00 should be truncated to U0100, and P0100 respectively. For example text might mention U0100, P2580 etc. If there are no such codes mentioned in Combined Narrative: , return "Not Found"
Do not include the prefix of "P Code and U Code:" in the response, only keep the relevant information and avoid narrative explanations. If none are found, return "Not found".
output the final combined response in the format of a python dictionary, with the keys of the dictionary being the following: "complaint", "cause", "repairs", "keywords", "codes", and the values being the answers to the corresponding information above.

     {context}

"""
            
        )



        prompt_refine = PromptTemplate.from_template(
            """Here's your first extracted dictionary from the service text: {prev_response}. """
            
            """Now utilize the information in this dictionary for information extraction in the next service text and the instructions below:
            Given a detailed service event report about an engine or vehicle failure, extract the following information from the text and format the result as a python dictionary with JSON formatting standards. This means surround all properties in double quotes instead of single:

        With the key of "model_identified_complaint", extract the primary customer complaint. The complaint is defined as the main engine or vehicle issue that the customer or technician is claiming is causing the repair, and summarize the complaint in less than 4 words, including the exact part or system affected, using technical terms.
If no such information is found, mention "Not available". in the response, only keep the relevant information. Avoid narriative explainations. Only include engine related terms and complaints. Do not count P codes as a complaint. Prioritize the first mentioned complaint, and if there is something mentioned by the customer prioritize that. If the text contains the phrase "maint" or similar to "maintenance", output the complaint as maintenance. For the outputted complaint, fix any spelling mistakes. If the identified complaint is in another language, translate it to English. Remove any quotations and extra punctuation, just keep the alphanumeric characters. Output the result in lowercase.
With the key of "identified_cause", extract the assignable cause of repair if any are mentioned in the text. For example: Belt noise, no communication, no start, sensor malfunctioning. This cause should identify the component or the cause that was fixed that able to fix the issue during the repair.
only keep the relevant information and avoid narrative explanations.
With the key of "identified_repairs", extract and present the repair actions in a short summary, For example: replaced nox sensor, replaced egr cooler, fixed coolant leak, in the response and only keep the relevant information while avoiding narrative explanations.
With the key of "identified_keywords", extract important service, repair or symptom related keywords in a comma separated list. Focus on any objective nouns that are mentioned. Examples: loose connector, burning smell, P codes or U codes.
in the response, only keep the relevant information and avoid any narrative explanations.
With the key of "identified_codes", given a detailed service event report about an engine or vehicle failure, extract ONLY P codes or U codes that start with a P or U and have 4 digits and truncate values before hyphens if any are present. Codes with hyphens like U0100-00 or P0100-00 should be truncated to U0100, and P0100 respectively. For example text might mention U0100, P2580 etc. If there are no such codes mentioned in Combined Narrative: , return "Not Found"
Do not include the prefix of "P Code and U Code:" in the response, only keep the relevant information and avoid narrative explanations. If none are found, return "Not found".
output the final combined response in the format of a python dictionary, with the keys of the dictionary being the following: "complaint", "cause", "repairs", "keywords", "codes", and the values being the answers to the corresponding information above.
             {context}"""
        )

        prev_response = self.query_endpoint(prompt.format(context=first_document))

        for i in remaining_documents:
            time.sleep(self.time_delay)
            prev_response = self.query_endpoint(prompt_refine.format(prev_response=prev_response, context = i.page_content))
        return prev_response

    async def query_endpoint_dict(self, query):
        request = GptChatCompletionRequest(
            [
                    ChatMessage(ChatMessageRole.USER, query),
            ]
        )
        resp = self.model.create_chat_completion(request)
        content = resp.choices[0].message.content

        content = content[content.find("{") : content.rfind("}") + 1]
        return content
    

    async def create_and_query_chain_summary(self):
        first_document = self.docs[0]
        remaining_documents = self.docs[1:]
        
        prompt = PromptTemplate.from_template(
            """"Briefly summarize the service information provided within the brackets at the end of these instructions.  Focus on identifying the primary complaint, all fault codes present, what issues were found, what actions were taken to resolve the issues, any parts that were replaced, and any other relevant information.  The given service information may be missing some of these details; in which case, they should not be mentioned in your summary.

            The goal is to provide a factually accurate summary, that is as brief as possible, while still providing a full understanding of the service event.  The service information to be summarized is within these brackets: [ " & PF_Notes & " ]      {context}
"""
            
        )



        prompt_refine = PromptTemplate.from_template(
            """Here's your first summary: {prev_response}. """
            
            """Now add to it based on the following context and instructions

            Briefly summarize the service information provided within the brackets at the end of these instructions.  Focus on identifying the primary complaint, all fault codes present, what issues were found, what actions were taken to resolve the issues, any parts that were replaced, and any other relevant information.  The given service information may be missing some of these details; in which case, they should not be mentioned in your summary.

            The goal is to provide a factually accurate summary, that is as brief as possible, while still providing a full understanding of the service event.  The service information to be summarized is within these brackets: [ " & PF_Notes & " ]      {context}

             {context}"""
        )

        prev_response = self.query_endpoint(prompt.format(context=first_document))

        for i in remaining_documents:
            time.sleep(self.time_delay)
            prev_response = self.query_endpoint(prompt_refine.format(prev_response=prev_response, context = i.page_content))

        return prev_response


    async def query_endpoint_summary(self, query):
        request = GptChatCompletionRequest(
            [
                    ChatMessage(ChatMessageRole.USER, query),
            ]
        )
        resp = self.model.create_chat_completion(request)
        cont
        ent = resp.choices[0].message.content
        return content
    
    
