from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

class RetrievalGeneration:
    """
    The RetrievalGeneration class is responsible for retrieving relevant document
    segments based on a user query and generating a response using a Large Language Model.
    """

    def __init__(self, llm_model, prompt_template):
        """
        Initializes the RetrievalGeneration class with an LLM model and a prompt template.

        :param llm_model: The model name of the Large Language Model.
        :param prompt_template: Template used for generating prompts for the LLM.
        """
        self.llm = Ollama(model=llm_model)
        self.prompt = ChatPromptTemplate.from_template(prompt_template)

    def create_document_chain(self):
        """
        Creates a document chain for combining multiple document segments into a single response.

        :return: Document chain object.
        """
        return create_stuff_documents_chain(self.llm, self.prompt)

    def create_retrieval_chain(self, retriever, document_chain):
        """
        Creates a retrieval chain using the provided retriever and document chain.

        :param retriever: The retriever object used for searching the database.
        :param document_chain: The document chain used for combining documents.
        :return: Retrieval chain object.
        """
        return create_retrieval_chain(retriever, document_chain)

    def generate_response(self, retrieval_chain, input_data):
        """
        Generates a response to the user's query using the retrieval chain.

        :param retrieval_chain: The retrieval chain used for generating the response.
        :param input_data: User's query input.
        :return: Generated response from the LLM.
        """
        return retrieval_chain.invoke(input_data)
