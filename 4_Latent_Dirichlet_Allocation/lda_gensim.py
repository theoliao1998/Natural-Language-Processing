from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from glob import glob
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.probability import FreqDist
kTOKENIZER = TreebankWordTokenizer()

def tokenize_file(filename):
    contents = open(filename, encoding="utf8").read()
    for ii in kTOKENIZER.tokenize(contents):
        yield ii.lower()

class VocabBuilder:
    """
    Creates a vocabulary after scanning a corpus.
    """

    def __init__(self, lang="english", min_length=3, cut_first=100):
        """
        Set the minimum length of words and which stopword list (by language) to
        use.
        """
        self._counts = FreqDist()
        self._stop = set(stopwords.words(lang))
        self._min_length = min_length
        self._cut_first = cut_first
        self.corpse = []

        print(("Using stopwords: %s ... " % " ".join(list(self._stop)[:10])))

    def scan(self, words):
        """
        Add a list of words as observed.
        """
        for ii in [x.lower() for x in words if x.lower() not in self._stop \
                       and len(x) >= self._min_length]:
            self._counts[ii] += 1

    def vocab(self, size=5000):
        """
        Return a list of the top words sorted by frequency.
        """
        if len(self._counts) > self._cut_first + size:
            return list(self._counts.keys())[self._cut_first:(size + self._cut_first)]
        else:
            return list(self._counts.keys())[:size]

vocab_scanner = VocabBuilder()

# Create a list of the files
search_path = "./wiki/*.txt"
files = glob(search_path)
assert len(files) > 0, "Did not find any input files in %s" % search_path
    
# Create the vocabulary
for ii in files:
    vocab_scanner.scan(tokenize_file(ii))
# Initialize the documents
vocab = vocab_scanner.vocab(1000)
print((len(vocab), vocab[:10])) 
vocab = Dictionary([vocab])

corpse = [vocab.doc2bow([t.lower() for t in tokenize_file(ii)]) for ii in files]
lda = LdaModel(corpse, num_topics=5, iterations=1000, alpha = [0.1 for _ in range(5)], eta = 0.01, id2word = vocab)


with open("gensim.txt",'w') as f:
    for i in range(5):
        f.write('------------\n')
        f.write('topic '+str(i)+'\n')
        f.write('------------\n')
        for j, p in lda.show_topic(i,50):
            f.write(str(p) + ' ' + j + '\n')