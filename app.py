from flask import Flask, render_template, request

from src.helper import (
    download_hugging_face_embeddings,
    search_query,
    generate_response
)

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

chat_history = []

last_topic = ""


@app.route("/", methods=["GET", "POST"])
def home():

    global last_topic

    if request.method == "POST":

        question = request.form["question"]

        # Simple memory improvement
        if len(question.split()) < 5 and last_topic:

            query = f"{question} about {last_topic}"

        else:

            query = question

            last_topic = question

        results = search_query(query, embeddings)

        response = generate_response(query, results)

        chat_history.append({
            "question": question,
            "response": response
        })

    return render_template(
        "index.html",
        chat_history=chat_history
    )


if __name__ == "__main__":
    app.run(debug=True)