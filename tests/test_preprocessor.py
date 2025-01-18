import pytest
from src.preprocessor import TextPreprocessor

@pytest.fixture
def preprocessor():
    """Create preprocessor instance for tests."""
    return TextPreprocessor()

def test_preprocess_text_basic(preprocessor):
    """Test basic text preprocessing."""
    text = "This is a TEST sentence!"
    processed = preprocessor.preprocess_text(text)
    assert processed.islower()  # Should be lowercase
    assert "!" not in processed  # Should remove punctuation

def test_preprocess_text_empty(preprocessor):
    """Test preprocessing empty text."""
    assert preprocessor.preprocess_text("") == ""
    assert preprocessor.preprocess_text(None) == ""

def test_preprocess_text_stopwords(preprocessor):
    """Test stopword removal."""
    text = "this is a test"
    processed = preprocessor.preprocess_text(text)
    assert "is" not in processed  
    assert "a" not in processed  

def test_preprocess_dataset(preprocessor):
    """Test dataset preprocessing."""
    texts = ["First text", "Second text", "Third text"]
    processed = preprocessor.preprocess_dataset(texts)
    assert len(processed) == len(texts)
    assert all(isinstance(text, str) for text in processed)

def test_error_handling(preprocessor):
    """Test error handling with invalid input."""
    # Should not raise error with invalid input
    result = preprocessor.preprocess_text(123)
    assert isinstance(result, str)
