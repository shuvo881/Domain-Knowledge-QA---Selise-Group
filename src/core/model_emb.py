import yaml
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

        if not self.model_name:
            raise ValueError("No completion model name specified in the configuration.")

        try:
            return ChatOpenAI(model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load the language model '{self.model_name}': {e}")

    def load_model_emb(self):

        if not self.model_emb_name:
            raise ValueError("No embedding model name specified in the configuration.")

        try:
            return OpenAIEmbeddings(model=self.model_emb_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load the embedding model '{self.model_emb_name}': {e}")
