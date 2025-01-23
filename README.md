# Brand Sentiment Analysis

## Overview
A comprehensive sentiment analysis system for brand monitoring on social media, implementing multiple machine learning models for sentiment classification. This project utilizes Natural Language Processing (NLP) techniques to analyze and classify brand-related sentiments from Twitter data.

## Features
- Multi-model sentiment analysis:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- Interactive prediction interface
- Comprehensive text preprocessing
- Model performance visualization and comparison
- Batch processing capabilities
- Detailed performance metrics and visualizations

## Model Performance
- Logistic Regression: 85% accuracy
- Naive Bayes: 70% accuracy
- Random Forest: 85% accuracy with detailed metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

## Installation


# Clone the repository
git clone https://github.com/yourusername/brand-sentiment-analyzer.git
cd brand-sentiment-analyzer

# Create virtual environment
python -m venv venv

## Install dependencies
pip install -r requirements.txt

## Using the Notebook
1. Navigate to `notebooks/` directory
2. Open `brand-sentiment-analyzer.ipynb`
3. Follow the step-by-step analysis

from src.preprocessor import TextPreprocessor
from src.models import SentimentAnalyzer

## Initialize preprocessor
preprocessor = TextPreprocessor()

## Initialize analyzer
analyzer = SentimentAnalyzer()

## Analyze sentiment
text = "Great product, amazing service!"
sentiment = analyzer.predict(text)

## Requirements

Python 3.8+
pandas
numpy
scikit-learn
nltk
jupyter
matplotlib
seaborn

## Development Setup
# Install development dependencies
pip install -r requirements-dev.txt

## Run tests
pytest tests/

## Run linting
flake8 src/

## Contributing

Fork the repository
Create your feature branch
bashCopygit checkout -b feature/amazing-feature

Commit your changes
bashCopygit commit -m 'Add amazing feature'

Push to the branch
bashCopygit push origin feature/amazing-feature

Open a Pull Request

## Testing

pytest tests/

## License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

## Dataset source: Twitter Brand Sentiment Dataset
NLTK library for text processing
scikit-learn for machine learning models
Inspiration from various sentiment analysis papers and implementations

## Contact
Your Name - @nagayahita
Project Link: https://github.com/yourusername/brand-sentiment-analyzer
