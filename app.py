from flask import Flask, render_template, request, send_file
from transformers import pipeline
import docx, PyPDF2, textstat, json, os
from datetime import datetime

app = Flask(__name__)
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

def chunk_text(text, max_tokens=400):
    sentences = text.split('. ')
    chunks, chunk, count = [], '', 0
    for sentence in sentences:
        if count + len(sentence.split()) > max_tokens:
            chunks.append(chunk.strip())
            chunk, count = '', 0
        chunk += sentence + '. '
        count += len(sentence.split())
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def read_file(file):
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return "Unsupported format."

def summarize_text(text, length="medium"):
    chunks = chunk_text(text)
    summaries = []

    max_len = {"short": 80, "medium": 130, "long": 200}.get(length, 130)

    for chunk in chunks:
        result = summarizer(chunk, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(result)

    final = ' '.join(summaries)
    formatted_summary = format_summary(final)

    words = len(text.split())
    chars = len(text)
    score = textstat.flesch_reading_ease(text)
    return formatted_summary, words, chars, score

def format_summary(raw_summary):
    points = raw_summary.split('. ')
    sections = {
        "Overview": [],
        "Details": [],
        "Conclusion": []
    }

    for i, sentence in enumerate(points):
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        if i < 2:
            sections["Overview"].append("• " + sentence)
        elif i < len(points) - 2:
            sections["Details"].append("• " + sentence)
        else:
            sections["Conclusion"].append("• " + sentence)

    formatted = ""
    for section, content in sections.items():
        formatted += f"\n\n🔹 {section}\n" + "\n".join(content)
    return formatted

def save_summary(summary):
    history = []
    if os.path.exists("summary_history.json"):
        try:
            with open("summary_history.json", "r") as f:
                history = json.load(f)
        except:
            history = []
    history.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "summary": summary})
    with open("summary_history.json", "w") as f:
        json.dump(history, f, indent=2)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = info = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        file = request.files.get("file")
        length = request.form.get("length", "medium")
        if file:
            text += read_file(file)
        if text.strip():
            summary, words, chars, score = summarize_text(text, length)
            info = f"📝 Words: {words} | 🔠 Characters: {chars} | 📊 Readability: {score:.2f}"
            save_summary(summary)
    return render_template("index.html", summary=summary, info=info)

@app.route("/download", methods=["POST"])
def download():
    content = request.form.get("summary", "")
    with open("summary_output.txt", "w", encoding="utf-8") as f:
        f.write(content)
    return send_file("summary_output.txt", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)


