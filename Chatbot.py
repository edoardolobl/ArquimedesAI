# Import necessary libraries
from transformers import BertTokenizer, BertModel
from DBHandler import DBHandler
import torch
import json
import faiss
import numpy as np

class Chatbot:
    def __init__(self):
        """
        Initializes the Chatbot with necessary models and data.
        
        Parameters:
        - training_data_path (str): Path to the training data in JSON format.
        """
        # Load BERT model and tokenizer for Portuguese
        self.bert_tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.bert_model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

        # Load training data and precompute BERT embeddings for questions
        self.training_data = self._load_training_data()
        self.question_embeddings = [self.get_bert_embedding(question["text"]) for question in self.training_data["questions"]]
        self.embeddings_array = np.array(self.question_embeddings)
        
        # Initialize the FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings_array.shape[1])
        self.index.add(self.embeddings_array)

    def _load_training_data(self):
        db_handler = DBHandler("ArquimedesDB.sqlite3")
        training_data = {"questions": []}
        for row in db_handler.cursor.execute("SELECT query_text, r.response_text FROM training_data t JOIN responses r ON t.response_id = r.response_id"):
            training_data["questions"].append({
                "text": row[0],
                "response": row[1]
            })
        return training_data


    def get_bert_embedding(self, text):
        """
        Computes the BERT embedding for a given text.
        
        Parameters:
        - text (str): Input text to embed.
        
        Returns:
        - np.array: Combined BERT embedding for the input text.
        """
        encoded_input = self.bert_tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = self.bert_model(**encoded_input)
        mean_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
        cls_embedding = output.last_hidden_state[0, 0, :].numpy()
        combined_embedding = (mean_embedding + cls_embedding) / 2
        return combined_embedding

    def get_response(self, user_query):
        """
        Retrieves the best response for a user's query.
        
        Parameters:
        - user_query (str): User's input query.
        
        Returns:
        - str: Best matching response from the training data.
        """
        user_embedding = self.get_bert_embedding(user_query)
        
        # Search the FAISS index for the most similar embedding
        D, I = self.index.search(user_embedding.reshape(1, -1), 1)
        best_match_index = I[0][0]

        return self.training_data["questions"][best_match_index]["response"]

    def save_faiss_index(self, filename="faiss_index.bin"):
        """
        Saves the current FAISS index to a file.
        
        Parameters:
        - filename (str): Path to save the index.
        """
        faiss.write_index(self.index, filename)

    def load_faiss_index(self, filename="faiss_index.bin"):
        """
        Loads a FAISS index from a file.
        
        Parameters:
        - filename (str): Path to load the index from.
        """
        self.index = faiss.read_index(filename)
