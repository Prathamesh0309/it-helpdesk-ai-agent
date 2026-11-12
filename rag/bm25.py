"""A small BM25 (Okapi) implementation with no dependencies.

Lexical search is the half of hybrid retrieval that embeddings are bad at:
exact identifiers, error strings, product names ("MTU", "TKT-", "ext. 4357").
"""
import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its my of on or "
    "so that the then there this to was we what when where which who why will with you your".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(token):
    """Very light stemming so 'dropping', 'drops' and 'dropped' meet, and 'minute' meets 'minutes'."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "aeiouls":
        token = token[:-1]  # dropp -> drop
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]  # minute -> minut
    return token


def tokenize(text):
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.term_freqs = [Counter(t) for t in self.doc_tokens]

        df = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        n = len(documents)
        # Non-negative IDF variant (as in Lucene) so common terms never subtract score.
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def scores(self, query):
        terms = tokenize(query)
        result = [0.0] * len(self.doc_tokens)
        for i, tf in enumerate(self.term_freqs):
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1.0))
            for term in terms:
                f = tf.get(term)
                if f:
                    result[i] += self.idf[term] * f * (self.k1 + 1) / (f + norm)
        return result

    def search(self, query, n=10):
        """Return up to n (document_index, score) pairs with score > 0, best first."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:n]
