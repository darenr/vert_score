import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
from typing import List
import importlib
from statistics import mean


def sentence_splitter(text: str) -> List[str]:
    """
    This function takes in a string and returns a list of strings
    representing the sentences in the input string.
    """
    return nltk.sent_tokenize(text=text)


def vert_score_sequence(a: str, b: str, **kwargs) -> float:
    return difflib.SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()


def vert_score_lcs(a: str, b: str, **kwargs) -> float:
    scores = []

    for sentence_a, sentence_b in zip(sentence_splitter(a), sentence_splitter(b)):
        print(f"{sentence_a=} vs {sentence_b=}")
        tokens_a = sentence_a.lower().split()
        tokens_b = sentence_b.lower().split()
        d = difflib.SequenceMatcher(None, tokens_a, tokens_b).get_matching_blocks()

        scores.append(d[0].size / max(len(tokens_a), len(tokens_b)))

    return mean(scores)


def vert_score_vectorized(a: str, b: str, **kwargs) -> float:
    """
    This function takes in two lists of strings and returns a float
    representing the similarity between the two lists. It uses the
    nltk library to compare the two lists and return a similarity
    score.
    """

    if importlib.util.find_spec("nltk") and importlib.util.find_spec("sklearn"):
        nltk.download("punkt")  # if necessary...

        stemmer = nltk.stem.porter.PorterStemmer()
        remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

        def normalize(text):
            def stem_tokens(tokens):
                return [stemmer.stem(item) for item in tokens]

            return stem_tokens(
                nltk.word_tokenize(text.lower().translate(remove_punctuation_map))
            )

        def cosine_sim(text1, text2):
            """remove punctuation, lowercase, stem"""

            tfidf = TfidfVectorizer(
                tokenizer=normalize, stop_words="english"
            ).fit_transform([text1, text2])
            return ((tfidf * tfidf.T).A)[0, 1]

    else:
        return -1.0


def vert_score_transformers(a: str, b: str, **kwargs) -> float:
    if importlib.util.find_spec("sentence_transformers"):
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Two lists of sentences
        sentences1 = ["The cat sits outside"]
        sentences2 = ["The dog plays in the garden"]
        # Compute embedding for both lists
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)
        # Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # Output the pairs with their score
        for i in range(len(sentences1)):
            print(
                "{} \t\t {} \t\t Score: {:.4f}".format(
                    sentences1[i], sentences2[i], cosine_scores[i][i]
                )
            )
    else:
        return -1.0


def darens_rougeL(response: str, reference: str, **kwargs) -> float:
    scores = []

    for sentence_a, sentence_b in zip(
        sentence_splitter(response), sentence_splitter(reference)
    ):
        tokens_a = sentence_a.lower().split()
        tokens_b = sentence_b.lower().split()
        d = difflib.SequenceMatcher(None, tokens_a, tokens_b).get_matching_blocks()

        scores.append(d[0].size / len(tokens_b))

    return mean(scores)


if __name__ == "__main__":
    # a = "The cat sat on the mat"
    # b = "On the mat, the cat sat"

    print(darens_rougeL("The cat sat on the mat", "On the mat, the cat sat"))
