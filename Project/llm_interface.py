import os
import requests
from transformers import pipeline
import openai
import google.generativeai as genai



class LLMInterface:
    """
    An abstract interface for any Large Language Model wrapper.
    """
    def get_name(self) -> str:
        """
        Return a short, unique name for this LLM (e.g. "ChatGPT" or "HF-FlanT5").
        """
        raise NotImplementedError("Subclasses must implement get_name().")

    def get_response(self, query: str, context: str) -> str:
        """
        Return the model's generated response given user query and context.
        """
        raise NotImplementedError("Subclasses must implement get_response().")


class ChatGPTLLM(LLMInterface):
    """
    Wrapper for OpenAI ChatGPT (model='gpt-3.5-turbo' or 'gpt-4').
    """
    def __init__(self, openai_api_key=None, model_name="gpt-3.5-turbo"):
        self._name = f"ChatGPT-{model_name}"
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        openai.api_key = self.openai_api_key

    def get_name(self) -> str:
        return self._name

    def get_response(self, query: str, context: str) -> str:
        """
        Calls OpenAI's ChatCompletion API, passing context as a system prompt.
        """
        system_prompt = (
            "You are an AI assistant specialized in explaining UK Open Government Data. "
            "Use the following context to answer the user's query as accurately as possible.\n\n"
            f"Context:\n{context}\n\n"
            "Please provide your answer below. If the context does not have the answer, say so."
        )
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2  # Lower temperature = more focused & deterministic
        )
        answer = response['choices'][0]['message']['content']
        return answer.strip()


class HuggingFaceLLM(LLMInterface):
    """
    Wrapper for a Hugging Face model using transformers pipelines.
    Best possible model for summarization or QA (as of now) 
    might be something like 'google/flan-t5-xxl' or 'facebook/bart-large-mnli' for classification.
    For general Q&A or summarization, let's pick 'google/flan-t5-xxl' or 'tiiuae/falcon-7b-instruct' if you have GPU.
    """
    def __init__(self, 
                 model_name="google/flan-t5-xxl",  # or 'tiiuae/falcon-7b-instruct'
                 task="text2text-generation", 
                 device=0):
        """
        device=-1 means CPU
        If you have a GPU, set device=0 (or another GPU index).
        """
        self.model_name = model_name
        self.task = task
        self.device = device
        self._name = f"HF-{model_name}"
        try:  
            self.pipe = pipeline(task, model=model_name,  use_auth_token= os.environ.get('hf_token'))
        except Exception as e:
            raise RuntimeError(f"Error loading Hugging Face model '{model_name}': {e}")

    def get_name(self) -> str:
        return self._name

    def get_response(self, query: str, context: str) -> str:
        """
        Creates a prompt with query + context, then calls the pipeline.
        """
        prompt = (
            f"You are an AI assistant specialized in explaining UK Open Government Data.\n\n"
            f"Context:\n{context}\n\n"
            f"User question: {query}\n"
            "Answer as best you can based on the context above. If the context doesn't contain the answer, say so."
        )
        try:
            result = self.pipe(prompt, max_length=512, do_sample=False)
            if isinstance(result, list) and len(result) > 0:
                # For text2text-generation or similar
                text = result[0].get('generated_text', '') or result[0].get('text', '')
                return text.strip()
            return "No output from Hugging Face pipeline."
        except Exception as e:
            return f"Error generating response with Hugging Face model: {e}"


class GeminiLLM(LLMInterface):
    """
    A mock/hypothetical wrapper for Google Gemini (not publicly released as of 2024).
    This is just a placeholder to illustrate how you'd structure calls to a different service.
    """
    def __init__(self, api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash", api_key=None):
        self.api_url = api_url
        self.api_key = api_key
        self._name = "Gemini-Alpha"

    def get_name(self) -> str:
        return self._name

    def get_response(self, query: str, context: str) -> str:
        genai.configure(api_key=self.api_key)
        # headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        # payload = {
        #     "model": "gemini-alpha",
        #     "prompt": (
        #         ".\n\n"
        #         f"Context:\n{context}\n\n"
        #         f"Query: {query}\n"
        #         "Answer as accurately as you can."
        #     )
        # }
        try:
                
            model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                system_instruction="You are an AI assistant specialized in explaining UK Open Government Data")
            response = model.generate_content(f"Based on the following information, {context}, answer the question: {query}")
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {e}"


class MultiLLM:
    """
    This class can manage multiple LLMInterface instances and 
    query all of them, returning a dictionary of {model_name: answer}.
    """
    def __init__(self, llms=None):
        """
        :param llms: A list of LLMInterface objects (e.g. [ChatGPTLLM(...), HuggingFaceLLM(...), GeminiLLM(...)])
        """
        self.llms = llms if llms is not None else []

    def add_llm(self, llm: LLMInterface):
        self.llms.append(llm)

    def get_all_responses(self, query: str, context: str) -> dict:
        """
        Call each LLM in self.llms, collecting their answers in a dict:
        {
          LLMName: answer_str,
          ...
        }
        """
        answers = {}
        for llm in self.llms:
            name = llm.get_name()
            ans = llm.get_response(query, context)
            answers[name] = ans
        return answers