import os
import yaml
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain_azure_ai.chat_models import AzureChatOpenAI

from dotenv import load_dotenv

load_dotenv()


class Loader:


    def __init__(self, config_path='src/configs/config.yaml'):

        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file: {e}")

        openai_config = config.get('openai', {})
        self.model_name = openai_config.get('completion_model', '')
        self.model_emb_name = openai_config.get('embedding_model', '')
        del openai_config

    def load_model(self):

        if os.getenv("AZURE_OPENAI_API_INSTANCE_NAME") is None or os.getenv("AZURE_OPENAI_API_KEY") is None or os.getenv("AZURE_OPENAI_API_COMPLETIONS_DEPLOYMENT_NAME") is None or os.getenv("AZURE_OPENAI_COMPLETIONS_API_VERSION") is None:
            raise ValueError("Azure OpenAI API environment variables are not properly set.")

        try:
            return AzureChatOpenAI(
                azure_endpoint=f"https://{os.getenv('AZURE_OPENAI_API_INSTANCE_NAME')}.cognitiveservices.azure.com",
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                model_name=os.getenv("AZURE_OPENAI_API_COMPLETIONS_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_COMPLETIONS_API_VERSION"),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load the language model '{self.model_name}': {e}")

    def load_model_emb(self):

        # if not self.model_emb_name:
        #     raise ValueError("No embedding model name specified in the configuration.")

        if os.getenv("AZURE_OPENAI_API_INSTANCE_NAME") is None or os.getenv("AZURE_OPENAI_API_KEY") is None or os.getenv("AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME") is None or os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION") is None:
            raise ValueError("Azure OpenAI API environment variables are not properly set.")

        try:
            # return OpenAIEmbeddings(model=self.model_emb_name)
            return AzureOpenAIEmbeddings(
                    azure_endpoint=f"https://{os.getenv('AZURE_OPENAI_API_INSTANCE_NAME')}.cognitiveservices.azure.com",
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    deployment=os.getenv("AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME"),
                    api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load the embedding model '{self.model_emb_name}': {e}")
