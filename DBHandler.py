
import sqlite3

class DBHandler:
    def __init__(self, db_path="ArquimedesDB.sqlite3"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
    def fetch_response(self, query_text):
        """Fetch the corresponding response for a given query."""
        self.cursor.execute("SELECT r.response_text FROM training_data t JOIN responses r ON t.response_id = r.response_id WHERE t.query_text = ?", (query_text,))
        response = self.cursor.fetchone()
        return response[0] if response else None
    
    def insert_positive_feedback(self, query_text, response_text):
        """Insert positive feedback into positive_feedback_store table."""
        self.cursor.execute("INSERT INTO positive_feedback_store (query_text, response_text) VALUES (?, ?)", (query_text, response_text))
        self.conn.commit()
    
    def insert_negative_feedback(self, query_text, response_text):
        """Insert negative feedback into negative_feedback_store table."""
        self.cursor.execute("INSERT INTO negative_feedback_store (query_text, response_text) VALUES (?, ?)", (query_text, response_text))
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

# For now, the negative_feedback_store and positive_feedback_store tables are not yet created in the SQLite database.
# They can be added later as per the requirements.
