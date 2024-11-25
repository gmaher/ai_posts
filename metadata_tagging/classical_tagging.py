# Import necessary libraries
import nltk
import spacy
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.summarization import keywords as gensim_keywords

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set up stop words and spaCy model
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

def classical_nlp_tagging(documents):
    classical_tags = []

    for doc in documents:
        # Entity Extraction using spaCy
        spacy_doc = nlp(doc)
        entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
        
        # Keyword Extraction using gensim's keywords function
        gensim_kw = gensim_keywords(doc, words=5, split=True, lemmatize=True)
        
        # Topic Modeling using LDA
        tokens = [
            word for word in nltk.word_tokenize(doc.lower())
            if word.isalpha() and word not in stop_words
        ]
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        lda_model = models.LdaModel(
            corpus, num_topics=1, id2word=dictionary, passes=10, iterations=50
        )
        topics = lda_model.print_topics(num_words=3)
        
        # Collect the tags
        classical_tags.append({
            'document': doc,
            'entities': entities,
            'keywords': gensim_kw,
            'topics': topics
        })

    return classical_tags

# Example usage
documents = [
    "This paper explores the use of machine learning algorithms in predicting stock market trends.",
    "The study investigates the impact of climate change on global agricultural production.",
    "An analysis of quantum computing and its potential applications in cryptography.",
    "A review of recent advancements in renewable energy technologies.",
    "This research focuses on the social and economic effects of the COVID-19 pandemic.",
]
classical_tags = classical_nlp_tagging(documents)