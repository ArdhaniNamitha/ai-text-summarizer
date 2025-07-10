from flask import Flask, render_template, request, send_file
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
import textstat
import json
import PyPDF2
import docx
import io

app = Flask(__name__)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

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

def summarize_text(text, length):
    max_length = 60 if length == "short" else 120 if length == "medium" else 200
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = original_text = ""
    word_count = readability = 0
    length = "medium"

    if request.method == "POST":
        length = request.form.get("length", "medium")
        if "file" in request.files and request.files["file"].filename:
            original_text = read_file(request.files["file"])
        else:
            original_text = request.form.get("text", "")

        if original_text.strip():
            summary = summarize_text(original_text, length)
            word_count = len(original_text.split())
            readability = textstat.flesch_reading_ease(original_text)
            save_summary(original_text, summary)

    history = load_history()
    return render_template("index.html", summary=summary, original=original_text,
                           word_count=word_count, readability=readability, history=history, length=length)

@app.route("/download", methods=["POST"])
def download():
    summary = request.form.get("summary", "")
    buffer = io.BytesIO()
    buffer.write(summary.encode("utf-8"))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="summary.txt", mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
