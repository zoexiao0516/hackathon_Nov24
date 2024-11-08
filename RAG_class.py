from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

class KnowledgeBaseRAG:
    def __init__(self, api_key, base_url, model_name="gpt-4-turbo", embedding_model_name="text-embedding-ada-002"):
        # Initialize the embedding model and LLM
        self.token = api_key
        self.base_url = base_url
        self.http_client = Client(verify="/path/to/certificate")  # Adjust path as needed

        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.token,
            http_client=self.http_client,
            model_name=model_name
        )

        # Initialize OpenAI embeddings model
        self.embed_model = OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.token,
            http_client=self.http_client,
            model=embedding_model_name
        )

        # Store loaded vector stores to avoid reloading
        self.loaded_vector_stores = {}

    def load_vector_store(self, kb_article):
        """Load the vector store for a specific KB article only if it hasn't been loaded yet."""
        if kb_article not in self.loaded_vector_stores:
            collection_name = f"KB{kb_article}"
            persist_directory = f"hygiene_advisor\\rag\\chroma_db\\chroma_{collection_name}"
            
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embed_model,
                persist_directory=persist_directory
            )
            self.loaded_vector_stores[kb_article] = vector_store
        return self.loaded_vector_stores[kb_article]

    def run_rag(self, kb_article, question):
        """Run RAG for a given question using the specified KB article's vector store."""
        # Load the vector store (loads only once per KB article)
        vector_store = self.load_vector_store(kb_article)

        # Set up retriever and QA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 5}, chain_type_kwargs={"verbose": True})
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)

        # Run the RAG chain with the question
        result = qa_chain.invoke(question)
        return result

# Example usage
api_key = "your_api_key_here"
base_url = "https://aigateway-dev.ms.com/openai/v1"

kb_rag = KnowledgeBaseRAG(api_key=api_key, base_url=base_url)

# Load KB article KB1848406 once and ask multiple questions
kb_article = "1848406"
questions = [
    "Who is the Sponsor and SME?",
    "What is the primary purpose of this KB article?",
    "What are the key recommendations?"
]

for question in questions:
    answer = kb_rag.run_rag(kb_article, question)
    print(f"Question: {question}\nAnswer: {answer}\n")
