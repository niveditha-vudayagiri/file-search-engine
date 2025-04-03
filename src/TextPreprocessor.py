import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer,PunktTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from nltk.util import ngrams
from   symspellpy  import SymSpell, Verbosity  # symspellpy forSpelling Correction
import pkg_resources
from gensim.models.phrases import Phrases, Phraser  # For n-gram formation
from textblob import TextBlob   # textblob for PoS tagging

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download("wordnet")



# Load spaCy model for semantic similarity
nlp = spacy.load("en_core_web_sm")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.max_synonyms = 3

        #Based on notebook by https://github.com/mdsharique/Information-Retrieval/blob/master/IR.ipynb
        self.sym_spell       = SymSpell(max_dictionary_edit_distance = 2, prefix_length = 7)     # Creating a Spell correction Object
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")  
        bigram_path     = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

        self.sym_spell.load_dictionary(dictionary_path, term_index = 0, count_index = 1)         # Loading the Unigram Dictionary
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index = 0, count_index = 2)      # Loading the Bigram  Dictionary

        ##------Stop Word Model-----##
        spacy_nlp       = spacy.load('en_core_web_sm')                                      # Loading the model of StopWord removal
        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS                               # Storing list of stopwords: https://raw.githubusercontent.com/explosion/spaCy/master/spacy/lang/en/stop_words.py

    def case_insensitive(self,text):
        text = text.lower()
        return text
    
    def punkt_tokenize(self,text):
        """
            Function for tokenizing a document into Sentences
            input:      Takes a string
            output:     Returns a list of separate sentence strings
        """
        tokenizer = PunktTokenizer()
        segmentedText = tokenizer.tokenize(text.strip())                  # Tokenize the document into sentences
        return segmentedText

    def clean_text(self,text,ngram_cond = False):
        """
        Function for parsing through the document sentences and generating cleaned tokens.
        INPUT       :   Takes a list of separate sentence strings
        OUTPUT      :   Returns a list of sentences which is a list of tokens for a document
        PARAMETERS  :
            MAX_EDIT_DIST -- The Maximum edit distance parameter controls up to which edit distance words from the dictionary should be treated as suggestions.
        """

        MAX_EDIT_DIST   = 2
        tokenizedText   = []      # Empty list for storing sentences of tokenized words

        for sentence in text:
            #sentence    = self.sym_spell.lookup_compound(sentence, max_edit_distance = MAX_EDIT_DIST)[0].term    # Spelling Correction
            token_words = TreebankWordTokenizer().tokenize(sentence)    
            token_words = [word for word in token_words if (word.isalnum() and word.isalpha())]             # Only considering tokens with ALPHABETS
            token_words = [word for word in token_words if word not in self.spacy_stopwords] 
            tokenizedText.append(token_words)  
            tokenizedText.append(self.generate_ngrams(sentence,2))                                                             # The tokens are then stored in a list

        return tokenizedText
    
    def lemmatization(self,text):
        """
        Function for word lemmatization.
        INPUT       :   Takes a list of sentences which is a list of tokenis for a document
        OUTPUT      :   Returns a list of sentences which is a list of lemmataized tokens for a document
        """
        reducedText = []                    # Empty list for storing sentences of lemmatized words
        lemmatizer = WordNetLemmatizer()    # Creating a WordNet lemmatization object

        for tokens in text:
            lem_word = []
            for word in tokens:
                pos_tag = TextBlob(word)
                if (pos_tag.tags):
                    if   (pos_tag.tags[0][1][0] == 'J'):
                        lem_word.append(lemmatizer.lemmatize(pos_tag.tags[0][0], pos = wordnet.ADJ ))
                    elif (pos_tag.tags[0][1][0] == 'N'):
                        lem_word.append(lemmatizer.lemmatize(pos_tag.tags[0][0], pos = wordnet.NOUN))
                    elif (pos_tag.tags[0][1][0] == 'R'):
                        lem_word.append(lemmatizer.lemmatize(pos_tag.tags[0][0], pos = wordnet.ADV ))
                    elif (pos_tag.tags[0][1][0] == 'V'):
                        lem_word.append(lemmatizer.lemmatize(pos_tag.tags[0][0], pos = wordnet.VERB))
                    else:
                        lem_word.append(word)

            reducedText.append(lem_word)
        return reducedText

    def get_wordnet_synonyms(self, word):
        """Fetch synonyms from WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))  # Replace underscores
        return list(synonyms)[: self.max_synonyms]  # Limit synonyms

    def get_spacy_synonyms(self, word):
        """Find semantically similar words using spaCy word vectors."""
        word_token = str(word)
        similar_words = []
        for candidate in nlp.vocab:
            if candidate.is_alpha and candidate.has_vector:  # Filter valid words
                similarity = word_token.similarity(nlp(candidate.text))
                if similarity > 0.6:  # Only include words with strong similarity
                    similar_words.append((candidate.text, similarity))
        similar_words.sort(key=lambda x: x[1], reverse=True)
        return [word[0] for word in similar_words[: self.max_synonyms]]

    def synonym_expansion(self, word):
        """Expand words with synonyms using WordNet and spaCy."""
        synonyms = set()
        synonyms.update(self.get_wordnet_synonyms(word))
        synonyms.update(self.get_spacy_synonyms(word))

        if synonyms:
            return f"({word} {' '.join(synonyms)})"
        return word
    
    def extract_named_entities(self, text):
        """Extract named entities from text using spaCy's NER."""
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities


    def generate_ngrams(self,text, n=2):
        tokens = text.split()
        return [" ".join(gram) for gram in ngrams(tokens, n)]
    
    def preprocess(self, text,isQuery=False):
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove punctuation and stopwords
        tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        processed_tokens = tokens.copy()

        # Stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        # Lemmatization
        #tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        #Expand for query
        if isQuery:
            tokens = [self.synonym_expansion(word) for word in tokens]  # Synonyms
        
        # Named Entity Recognition
        tokens.extend(self.extract_named_entities(text))

        tokens.extend(self.generate_ngrams(processed_tokens, 2))  # Bigrams
        tokens.extend(self.generate_ngrams(processed_tokens, 3)) # Trigrams

        return " ".join(tokens)