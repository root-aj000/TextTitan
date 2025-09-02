import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from docx import Document
import os

# ---------------------------
# CONFIGURATION
# ---------------------------
INPUT_DOCX = "input.docx"        # Input Word file
OUTPUT_MD = "rewritten.md"       # Output Markdown file
CHUNK_SIZE = 80                  # Words per chunk

# Model: TinyLlama runs on CPU (8GB RAM ok)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ---------------------------
# LOAD MODEL
# ---------------------------
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# ---------------------------
# HELPERS
# ---------------------------
def chunk_text(text, size=CHUNK_SIZE):
    """Split long text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def llama_generate(prompt):
    """Generate rewritten text from TinyLlama."""
    output = generator(
        prompt,
        max_length=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    return output[0]["generated_text"]

def rewrite_text(text, is_heading=False, is_list=False):
    """Single-pass rewriting for speed."""
    if not text.strip():
        return text

    if is_heading:
        prompt = f"Rephrase this heading in simple human-like words, plagiarism-free:\n\n{text}\n\nNew Heading:"
    elif is_list:
        prompt = f"Rephrase this bullet point with synonyms in easy words:\n\n{text}\n\nNew Bullet:"
    else:
        prompt = f"Rephrase this paragraph in natural easy English, plagiarism-free:\n\n{text}\n\nRewritten:"
    
    out = llama_generate(prompt)
    return out.split(":")[-1].strip()

def process_paragraph(text, style):
    """Handle headings, lists, and paragraphs."""
    style_lower = style.lower()

    if "heading" in style_lower:
        rewritten = rewrite_text(text, is_heading=True)
        level = style_lower.replace("heading", "").strip()
        level = int(level) if level.isdigit() else 2
        return "#" * level + " " + rewritten + "\n"

    elif "list" in style_lower:
        rewritten = rewrite_text(text, is_list=True)
        return f"- {rewritten}"

    else:  # Normal paragraph
        chunks = chunk_text(text)
        rewritten_chunks = [rewrite_text(chunk) for chunk in chunks]
        return " ".join(rewritten_chunks) + "\n"

def process_table(table):
    """Convert Word tables into Markdown tables."""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")

    if rows:
        header = rows[0]
        separator = "| " + " | ".join(["---"] * (len(rows[0].split('|')) - 2)) + " |"
        return "\n".join([header, separator] + rows[1:]) + "\n"
    return ""

# ---------------------------
# MAIN PROCESSOR
# ---------------------------
def process_docx(input_file, output_file):
    doc = Document(input_file)
    md_lines = []

    table_idx = 0
    for block in doc.element.body:
        tag = block.tag.split("}")[-1]

        # Paragraphs
        if tag == "p":
            para = block
            text = para.text.strip()
            if not text:
                continue

            # Detect LaTeX equations
            if text.startswith("$") and text.endswith("$"):
                md_lines.append(text + "\n")
                continue

            style_name = para.style.name if hasattr(para, "style") else "Normal"
            rewritten = process_paragraph(text, style_name)
            md_lines.append(rewritten)

        # Tables
        elif tag == "tbl":
            table_obj = doc.tables[table_idx]
            md_lines.append(process_table(table_obj))
            table_idx += 1

        # Images
        elif tag in ["drawing", "pict"]:
            md_lines.append("![Image](image-placeholder.png)\n")

        else:
            md_lines.append("<!-- Other element preserved -->\n")

    # Save Markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✅ Rewritten Markdown saved as {output_file}")

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    process_docx(INPUT_DOCX, OUTPUT_MD)