import nltk
import spacy

def setup():
    nltk.download(['punkt', 'stopwords', 'wordnet'])
    try:
        spacy.cli.download("en_core_web_sm")
    except:
        pass

if __name__ == "__main__":
    setup()