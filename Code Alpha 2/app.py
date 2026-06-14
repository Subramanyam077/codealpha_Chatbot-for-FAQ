from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import faqs

app = Flask(__name__)

questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]

vectorizer = TfidfVectorizer(stop_words="english")
question_vectors = vectorizer.fit_transform(questions)


def get_best_answer(user_question):
    user_vector = vectorizer.transform([user_question])
    similarity_scores = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I could not find a matching answer. Please try asking differently."

    return answers[best_match_index]


@app.route("/", methods=["GET", "POST"])
def home():
    chat_history = []

    if request.method == "POST":
        user_question = request.form.get("question", "")
        bot_answer = get_best_answer(user_question)

        chat_history.append({
            "user": user_question,
            "bot": bot_answer
        })

    return render_template("index.html", chat_history=chat_history)


if __name__ == "__main__":
    app.run(debug=True)