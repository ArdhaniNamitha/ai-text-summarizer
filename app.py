from flask import Flask, render_template, request, send_file
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import textstat
import json
import os
import io
import PyPDF2
import docx

app = Flask(__name__)

# Load small model to fit within 512MB Render memory
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

HISTORY_FILE = "summary_history.json"

# ========== Utilities ==========

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

def summarize(text, length="medium"):
    prompt = "summarize: " + text.strip()
    max_len = {"short": 50, "medium": 120, "long": 200}.get(length, 120)
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=max_len, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def save_summary(original, summary):
    history = load_history()
    history.insert(0, {"original": original[:500], "summary": summary})
    history = history[:10]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# ========== Routes ==========

@app.route("/", methods=["GET", "POST"])
def index():
    summary = word_count = readability = original = ""
    selected_length = "medium"

    if request.method == "POST":
        selected_length = request.form.get("length", "medium")

        if "file" in request.files and request.files["file"].filename:
            original = read_file(request.files["file"])
        else:
            original = request.form["text"]

        if original.strip():
            summary = summarize(original, selected_length)
            word_count = len(original.split())
            readability = textstat.flesch_reading_ease(original)
            save_summary(original, summary)

    history = load_history()
    return render_template("index.html",
                           summary=summary,
                           word_count=word_count,
                           readability=readability,
                           history=history,
                           length=selected_length)

@app.route("/download", methods=["POST"])
def download():
    summary = request.form["summary"]
    file_stream = io.BytesIO()
    file_stream.write(summary.encode("utf-8"))
    file_stream.seek(0)
    return send_file(file_stream, as_attachment=True, download_name="summary.txt", mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)

