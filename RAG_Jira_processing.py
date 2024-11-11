import os
import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from httpx import Client
from uuid import uuid4

class JiraKBProcessor:
    def __init__(self, jira_folder_path, api_key, base_url, tiktoken_cache_dir=None):
        self.jira_folder_path = jira_folder_path
        self.api_key = api_key
        self.base_url = base_url
        self.http_client = Client()
        self.embed_model = OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=self.http_client,
            model="text-embedding-ada-002"
        )
        self.vector_stores = {}

        # Set up TikToken cache if provided
        if tiktoken_cache_dir:
            os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

    def load_jira_files(self):
        """Load all Jira JSON files from the specified folder."""
        jira_data = []
        for file_name in os.listdir(self.jira_folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.jira_folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    jira_data.append(json.load(f))
        return jira_data

    def group_jiras_by_kb(self, jira_data):
        """Group Jiras by KB article."""
        kb_groups = {}
        for jira in jira_data:
            kb_articles = jira.get("kb", [])
            status = jira.get("status", "").lower()
            comments = jira.get("comments", [])
            # Only add comments for Jiras with status "complete"
            if status == "complete":
                for kb_article in kb_articles:
                    if kb_article not in kb_groups:
                        kb_groups[kb_article] = []
                    kb_groups[kb_article].extend(comments)
        return kb_groups

    def load_or_create_vector_store(self, kb_article):
        """Load or create the vector store for a KB article."""
        if kb_article not in self.vector_stores:
            collection_name = kb_article
            persist_directory = f"hygiene_advisor/rag/chroma_db/chroma_{collection_name}"
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embed_model,
                persist_directory=persist_directory
            )
            self.vector_stores[kb_article] = vector_store
        return self.vector_stores[kb_article]

    def add_comments_to_vector_store(self, kb_groups):
        """Add comments to the corresponding vector store for each KB article."""
        for kb_article, comments in kb_groups.items():
            # Load or create the vector store for this KB article
            vector_store = self.load_or_create_vector_store(kb_article)

            # Add comments to the vector store
            for comment in comments:
                embedding = self.embed_model.embed_query(comment)
                vector_store.add_texts([comment], embeddings=[embedding], ids=[str(uuid4())])

    def process_jira_kb_comments(self):
        """Main function to process and add comments from Jiras to their respective KB vector stores."""
        # Step 1: Load all Jira files
        jira_data = self.load_jira_files()
        
        # Step 2: Group Jiras by KB article with "complete" status
        kb_groups = self.group_jiras_by_kb(jira_data)
        
        # Step 3: Add comments to the corresponding vector store for each KB article
        self.add_comments_to_vector_store(kb_groups)


# Example usage
api_key = "your_api_key_here"
base_url = "https://aigateway-dev.ms.com/openai/v1"
jira_folder_path = "path_to_your_jira_folder"  # Replace with your actual Jira folder path
tiktoken_cache_dir = "hygiene_advisor/tokenizer_cache"

processor = JiraKBProcessor(
    jira_folder_path=jira_folder_path,
    api_key=api_key,
    base_url=base_url,
    tiktoken_cache_dir=tiktoken_cache_dir
)

# Process Jiras and add comments to KB vector stores
processor.process_jira_kb_comments()
