import os
import logging
import uuid
from pymilvus import MilvusClient, DataType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
# Make sure to install python-dotenv: pip install python-dotenv
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Basic Configuration ---
# Set up logging to display informational messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Environment Variables ---
# The OpenAI API key is no longer needed for embeddings.
# The Groq key is still needed for the language model.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- MODIFIED: Configuration for Sentence Transformer model ---
# Using a popular, efficient Sentence Transformer model.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# The dimension for "all-MiniLM-L6-v2" is 384. This is a critical change.
VECTOR_DIMENSION = 384
MILVUS_COLLECTION_NAME = "email_rag_collection_st" # Using a new name to avoid conflicts
MILVUS_DB_FILE = "email_rag_st.db"
BATCH_SIZE = 128 # A smaller batch size can be more manageable for memory

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

class EmbedData:
    """Handles the creation of text embeddings using a specified model."""
    def __init__(self, embed_model_name=EMBED_MODEL_NAME, batch_size=BATCH_SIZE):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size

    def _load_embed_model(self):
        """
        --- MODIFIED: Initializes a local Sentence Transformer model. ---
        This will download the model from Hugging Face the first time it runs.
        """
        logger.info(f"Loading local embedding model: {self.embed_model_name}")
        return HuggingFaceEmbedding(model_name=self.embed_model_name)

    def embed(self, contexts):
        """Generates embeddings for a list of text contexts."""
        logger.info(f"Generating float embeddings for {len(contexts)} contexts...")
        all_embeddings = []
        for batch_context in batch_iterate(contexts, self.batch_size):
            try:
                # The method call remains the same due to LlamaIndex's consistent interface
                batch_embeddings = self.embed_model.get_text_embedding_batch(batch_context)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                all_embeddings.extend([[0.0] * VECTOR_DIMENSION] * len(batch_context))

        logger.info(f"Generated {len(all_embeddings)} float embeddings.")
        return all_embeddings


class MilvusVDB:
    """Manages the Milvus vector database, including collection creation and data ingestion."""
    def __init__(self, collection_name, vector_dim, db_file, batch_size=BATCH_SIZE):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.db_file = db_file
        self.client = MilvusClient(self.db_file)

    def create_collection(self):
        """Creates a new Milvus collection with a schema tailored for email data."""
        if self.client.has_collection(collection_name=self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists. Dropping it.")
            self.client.drop_collection(collection_name=self.collection_name)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_fields=True)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=255)
        schema.add_field(field_name="sender", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=2000)
        schema.add_field(field_name="context", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.vector_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2")
        self.client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)
        logger.info(f"Created email collection '{self.collection_name}' with FLAT index.")

    def ingest_data(self, emails, embeddings):
        """Ingests a list of email dictionaries and their corresponding float embeddings."""
        logger.info(f"Ingesting {len(emails)} emails...")
        data_to_insert = [{
            "id": email['id'],
            "sender": email['sender'],
            "subject": email['subject'],
            "context": f"From: {email['sender']}\nSubject: {email['subject']}\n\n{email['body']}",
            "vector": emb
        } for email, emb in zip(emails, embeddings)]

        for batch in batch_iterate(data_to_insert, self.batch_size):
            self.client.insert(collection_name=self.collection_name, data=batch)
        logger.info(f"Successfully ingested {len(emails)} emails.")


class Retriever:
    """Handles searching the vector database to find relevant documents for a query."""
    def __init__(self, vector_db, embed_data, top_k=3):
        self.vector_db = vector_db
        self.embed_data = embed_data
        self.top_k = top_k

    def search(self, query, top_k=None):
        """Embeds a query and searches for the most similar documents in Milvus."""
        top_k = top_k or self.top_k
        query_embedding = self.embed_data.embed_model.get_query_embedding(query)

        search_results = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            data=[query_embedding],
            anns_field="vector",
            limit=top_k,
            output_fields=["id", "context", "sender", "subject"]
        )
        return search_results[0] if search_results else []


class RAG:
    """The main RAG class that orchestrates retrieval and generation."""
    def __init__(self, retriever, llm_model="llama3-8b-8192"):
        if not GROQ_API_KEY:
            logger.warning("Groq API key not found in environment variables. LLM calls may fail.")
        self.llm = Groq(model=llm_model, api_key=GROQ_API_KEY, temperature=0.2, max_tokens=1000)
        self.retriever = retriever
        self.prompt_template = (
            "CONTEXT: {context}\n"
            "---------------------\n"
            "Given the context from the user's emails above, answer the user's query crisply.\n"
            "Only answer based on the facts and information provided in the context.\n"
            "If the information is not in the context, say 'I could not find an answer in your emails.'\n"
            "QUERY: {query}\n"
            "ANSWER: "
        )

    def answer(self, query, top_k=3):
        """Generates an answer by first retrieving context and then calling the LLM."""
        logger.info(f"Generating answer for query: '{query}'")
        results = self.retriever.search(query, top_k=top_k)
        if not results:
            return "I could not find any relevant information in your emails.", []

        context = "\n\n---\n\n".join([entry["entity"]["context"] for entry in results])
        formatted_prompt = self.prompt_template.format(context=context, query=query)

        try:
            response = self.llm.complete(formatted_prompt)
            return response.text, results
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            return "There was an error generating a response from the language model.", results

        #This is the end of the code
