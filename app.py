from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import textstat
import json
import PyPDF2
import docx

app = Flask(__name__)

# Load lightweight summarizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

HISTORY_FILE = "summary_history.json"

# Load saved summaries
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# Save new summary
def save_summary(original, summary):
    history = load_history()
    history.insert(0, {"original": original, "summary": summary})
    history = history[:10]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# Read file upload
def read_file(file):
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in pdf.pages)
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

# Summarize using t5-small
def summarize_text(text):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
