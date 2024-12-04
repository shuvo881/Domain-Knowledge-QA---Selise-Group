import yaml
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, config_path="./src/configs/config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def load_docs(self):
        
        loader_config = self.config["document_loader"]
        loader = JSONLoader(
            file_path=loader_config["file_path"],
            jq_schema=loader_config["jq_schema"],
            text_content=loader_config["text_content"],
            json_lines=loader_config["json_lines"],
        )
        docs = loader.load()
        return docs

    def split_docs(self, docs):

        splitter_config = self.config["processing"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=splitter_config["chunk_size"],
            chunk_overlap=splitter_config["chunk_overlap"],
            add_start_index=splitter_config["add_start_index"],
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Split blog post into {len(all_splits)} sub-documents.")
        return all_splits

    def process(self):
        docs = self.load_docs()
        return self.split_docs(docs)



