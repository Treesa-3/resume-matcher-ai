from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILLS = [
    "python", "sql", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "machine learning", "deep learning", "nlp", "docker",
    "aws", "azure", "fastapi", "flask", "git", "github", "tableau",
    "power bi", "excel", "matplotlib", "statistics", "postgresql",
    "mongodb", "spark", "airflow", "streamlit", "langchain",
]


def clean_text(text):
    """Lowercase the text; keep letters, numbers and spaces (newlines become spaces)."""
    result = ''
    for i in text:
        if i.isalnum():
            result += i.lower()
        elif i.isspace():
            result += ' '
    return result


def build_vectors(resume_text, jd_text):
    """Turn the resume and JD into TF-IDF number vectors."""
    texts = [clean_text(resume_text), clean_text(jd_text)]
    # stop_words='english' ignores filler words like "and", "for", "the"
    vectorizer = TfidfVectorizer(stop_words='english')
    # fit = build the vocabulary, transform = convert each text into numbers
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def get_score(matrix):
    """Cosine similarity between resume (row 0) and JD (row 1), as a 0-100 score."""
    score = cosine_similarity(matrix[0], matrix[1])[0][0]
    return round(score * 100, 1)


def get_missing_keywords(matrix, vectorizer):
    """Words the JD asks for that the resume lacks, highest JD weight first."""
    words = vectorizer.get_feature_names_out()
    resume_weights = matrix[0].toarray()[0]
    jd_weights = matrix[1].toarray()[0]

    missing = []
    for i in range(len(words)):
        # missing = the JD has the word, the resume doesn't
        if resume_weights[i] == 0 and jd_weights[i] > 0:
            missing.append((str(words[i]), jd_weights[i]))

    missing.sort(key=lambda pair: pair[1], reverse=True)
    return [word for word, weight in missing]


def get_skill_gaps(resume_text, jd_text):
    """Return (covered, missing): skills the JD asks for, split by whether the resume has them."""
    # Padding with spaces means "excel" won't match inside "excellent".
    resume = " " + clean_text(resume_text) + " "
    jd = " " + clean_text(jd_text) + " "

    covered = []
    missing = []
    for skill in SKILLS:
        padded_skill = " " + clean_text(skill) + " "
        if padded_skill in jd:
            if padded_skill in resume:
                covered.append(skill)
            else:
                missing.append(skill)
    return covered, missing


if __name__ == "__main__":
    resume = """
    Aspiring data analyst with hands-on experience in Python, SQL and Excel.
    Built a student performance dashboard using pandas and matplotlib.
    Completed a machine learning course covering regression, classification
    and model evaluation with scikit-learn. Created a sales analysis project
    that cleaned messy CSV data and visualized monthly trends.
    Familiar with Git, GitHub and Jupyter Notebook. Strong communication
    and teamwork skills. B.Tech in Computer Science.
    """

    jd = """
    We are looking for a Junior Data Scientist to join our analytics team.
    You will build machine learning models, analyze large datasets and
    present insights to stakeholders. Requirements: strong Python and SQL,
    experience with pandas, numpy and scikit-learn, knowledge of
    TensorFlow or PyTorch, familiarity with Docker and AWS, and experience
    deploying models with FastAPI. Experience with Tableau or Power BI is a plus.
    Good communication skills and ability to work in a team.
    """

    matrix, vectorizer = build_vectors(resume, jd)
    print(get_score(matrix))
    print(get_missing_keywords(matrix, vectorizer)[:10])

    covered, missing = get_skill_gaps(resume, jd)
    print(f"Covers {len(covered)} of {len(covered) + len(missing)} required skills")
    print("Missing:", missing)