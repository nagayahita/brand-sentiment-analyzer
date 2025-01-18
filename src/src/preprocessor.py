# src/preprocessor.py

# Import libraries yang diperlukan
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from typing import List

# Class TextPreprocessor
class TextPreprocessor:
    """Text preprocessing pipeline for sentiment analysis."""
    
    def __init__(self):
        """Initialize the preprocessor with required NLTK resources."""
        try:
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stopwords = set(stopwords.words('english'))
            
            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            
        except LookupError:
            # Download NLTK resources if not available
            self.logger.warning("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            self.lemmatizer = WordNetLemmatizer()
            self.stopwords = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        try:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            cleaned_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stopwords
            ]
            
            return ' '.join(cleaned_tokens)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {str(e)}")
            return text

    def preprocess_dataset(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of preprocessed texts
        """
        try:
            return [self.preprocess_text(text) for text in texts]
        except Exception as e:
            self.logger.error(f"Error preprocessing dataset: {str(e)}")
            return texts
