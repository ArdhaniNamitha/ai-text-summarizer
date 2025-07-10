from flask import Flask, render_template, request
from transformers import pipeline
import os
import textstat
import json
import PyPDF2
import docx

app = Flask(__name__)
summarizer = pipeline("summarization", model="t5-small")

HISTORY_FILE = "summary_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_summary(original, summary):
    history = load_history()
    history.insert(0, {"original": original, "summary": summary})
    history = history[:10]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def read_file(file):
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def summarize_text(text):
    summary = summarizer(text[:1000], max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

@app.route("/", methods=["GET", "POST"])
def index():
    summary = original_text = word_count = readability = ""
    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            original_text = read_file(request.files["file"])
        else:
            original_text = request.form["text"]

        if original_text.strip():
            summary = summarize_text(original_text)
            word_count = len(original_text.split())
            readability = textstat.flesch_reading_ease(original_text)
            save_summary(original_text, summary)

    history = load_history()
    return render_template("index.html", summary=summary, original=original_text,
                           word_count=word_count, readability=readability, history=history)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
