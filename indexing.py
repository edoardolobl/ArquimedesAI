from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS

class Indexing:
    """
    The Indexing class is responsible for loading documents from a web source,
    splitting them into smaller segments, and storing these segments in a
    vector database for efficient retrieval.
    """

    def __init__(self, url, model_name, cache_path):
        """
        Initializes the Indexing class with required parameters.

        :param url: URL of the web source from which to load documents.
        :param model_name: Name of the model used for embeddings.
        :param cache_path: Path to store cached embeddings.
        """
        self.url = url
        self.model_name = model_name
        self.cache_path = cache_path

    def load_documents(self):
        """
        Loads documents from the specified URL using WebBaseLoader.

        :return: Loaded documents.
        """
        loader = WebBaseLoader(self.url)
        return loader.load()

    def split_documents(self, docs):
        """
        Splits the loaded documents into smaller segments using RecursiveCharacterTextSplitter.

        :param docs: Documents to split.
        :return: List of split documents.
        """
        splitter = RecursiveCharacterTextSplitter()
        return splitter.split_documents(docs)

    def store_documents(self, documents):
        """
        Stores and indexes the split documents in a FAISS vector database.

        :param documents: List of documents to be stored.
        :return: FAISS database object.
        """
        underlying_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        store = LocalFileStore(self.cache_path)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store)
        db = FAISS.from_documents(documents, cached_embedder)
        return db
