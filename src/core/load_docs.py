import yaml
import os
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, config_path="./src/configs/config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def load_docs(self):

        loader_config = self.config["document_loader"]
        docs = []

        file_path = loader_config["file_path"]
        if os.path.isfile(file_path):  # Single file
            docs.extend(self._load_single_file(file_path, loader_config))
        elif os.path.isdir(file_path):  # Directory of files
            for filename in os.listdir(file_path):
                full_path = os.path.join(file_path, filename)
                if filename.endswith(".jsonl") and os.path.isfile(full_path):
                    docs.extend(self._load_single_file(full_path, loader_config))
        else:
            raise ValueError(f"Invalid path: {file_path}. It must be a file or directory.")

        return docs

    def _load_single_file(self, file_path, loader_config):

        loader = JSONLoader(
            file_path=file_path,
            jq_schema=loader_config["jq_schema"],
            text_content=loader_config["text_content"],
            json_lines=loader_config["json_lines"],
        )
        return loader.load()

    def split_docs(self, docs):

        splitter_config = self.config["processing"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=splitter_config["chunk_size"],
            chunk_overlap=splitter_config["chunk_overlap"],
            add_start_index=splitter_config.get("add_start_index", False),
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Split documents into {len(all_splits)} sub-documents.")
        return all_splits

    def process(self):

        docs = self.load_docs()
        return self.split_docs(docs)
