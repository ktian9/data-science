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


### Defining Prompts
# document_prompt = PromptTemplate(
#     input_variables=["page_content"],
#      template="{page_content}"
# )
# document_variable_name = "context"
# prompt = PromptTemplate.from_template(
#     "Summarize this content: {context}"
# )


# prompt_refine = PromptTemplate.from_template(
#     "Here's your first summary: {prev_response}. "
#     "Now add to it based on the following context: {context}"
# )


class custom_chain:
    def __init__(
        self,
        text_id,
        summary_prompt,
        refine_prompt,
        time_delay,
        model: OpenAiGptChatLanguageModel,
    ):
        self.model = model
        self.summary_prompt = summary_prompt
        self.time_delay = time_delay
        self.refine_prompt = refine_prompt
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

    async def create_and_query_chain_summary(self):
        first_document = self.docs[0]
        remaining_documents = self.docs[1:]

        prev_response = self.query_endpoint(self.summary_prompt, first_document)

        for i in remaining_documents:
            time.sleep(self.time_delay)
            prev_response = self.query_endpoint(self.refine_prompt, prev_response)

    async def query_endpoint(self, prompt, query):
        request = GptChatCompletionRequest(
            [
                ChatMessage(ChatMessageRole.SYSTEM, prompt),
                ChatMessage(ChatMessageRole.USER, query),
            ]
        )
        resp = self.model.create_chat_completion(request)
        content = resp.choices[0].message.content

        content = content[content.find("{") : content.rfind("}") + 1]
        return content
