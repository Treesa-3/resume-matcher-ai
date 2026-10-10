from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    """Lowercase the text and keep only letters, numbers and spaces."""
    result = ''
    for i in text:
        if i.isalnum() or i == ' ':
            result += i.lower()
    return result


def build_vectors(resume_text, jd_text):
    """Turn the resume and JD into TF-IDF number vectors."""
    texts = [clean_text(resume_text), clean_text(jd_text)]
    vectorizer = TfidfVectorizer(stop_words='english')
    # Create the TF-IDF tool. stop_words='english' tells it to ignore filler words
    # like "and", "for", "the", so they don't count toward the score or the keywords.
    matrix = vectorizer.fit_transform(texts)
    # fit_transform does two jobs in one step:
    #   fit       -> reads both texts and builds the vocabulary (the list of all unique words)
    #   transform -> converts each text into a row of numbers, one number per vocabulary word
    # The result, `matrix`, has 2 rows: row 0 is the resume, row 1 is the JD.
    return matrix, vectorizer


def get_score(matrix):
    """Cosine similarity between resume (row 0) and JD (row 1), as a 0-100 score."""
    score = cosine_similarity(matrix[0], matrix[1])[0][0]
    return round(score * 100, 1)


def get_missing_keywords(matrix, vectorizer):
    """Words the JD asks for that the resume lacks, most important first."""

    # The list of all unique words the vectorizer found across both texts.
    # Position 0 in this list matches column 0 in the matrix, and so on.
    words = vectorizer.get_feature_names_out()

    # Row 0 of the matrix is the resume. toarray() turns it into a normal
    # array of numbers, and [0] picks out that single row.
    # One weight per word; 0 means the word is not in the resume.
    resume_weights = matrix[0].toarray()[0]

    # Row 1 is the JD, converted the same way.
    # A weight above 0 means the JD contains that word.
    jd_weights = matrix[1].toarray()[0]

    # Empty list that will collect (word, weight) pairs for the missing words.
    missing = []

    # Go through every word position, one at a time (0, 1, 2, ...).
    for i in range(len(words)):
        # Missing = the JD has the word, but the resume doesn't.
        if resume_weights[i] == 0 and jd_weights[i] > 0:
            # Save the word with its JD weight, so we can sort by importance.
            # str() converts the word to a plain Python string.
            missing.append((str(words[i]), jd_weights[i]))

    # Sort the pairs by weight (pair[1]), highest first, so the words the JD
    # emphasizes most come at the top.
    missing.sort(key=lambda pair: pair[1], reverse=True)

    # Return just the words, dropping the weights.
    return [word for word, weight in missing]


if __name__ == "__main__":
    matrix, vectorizer = build_vectors(
        "I know Python and SQL",
        "Looking for Python, SQL and Docker"
    )                   #only can see this one's o/p in running matcher.py not app
    print(get_score(matrix))
    print(get_missing_keywords(matrix, vectorizer))