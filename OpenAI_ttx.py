import os
import openai
from docx import Document
from tqdm import tqdm
import math

# ---------------------------
# CONFIGURATION
# ---------------------------
INPUT_DOCX = "./src/input.docx"
OUTPUT_MD = "./src/rewritten.md"
WARN_FILE = "./src/warnings.txt"
DEFAULT_FONT_SIZE = 12
OPENAI_MODEL = "gpt-4-turbo"  # or GPT-4-mini
MAX_TOKENS = 1000  # max tokens per chunk for OpenAI API
WORDS_PER_CHUNK = 250  # estimate per chunk

openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def estimate_tokens(text):
    """Rough estimate: 1 token ≈ 0.75 words"""
    return math.ceil(len(text.split()) / 0.75)

def rewrite_chunk(text):
    """Rewrite a chunk of text using OpenAI API."""
    prompt = (
        "Rewrite the following text to be plagiarism-free, natural, easy to read, "
        "keeping the meaning intact. Avoid common phrases and make it original:\n\n"
        f"{text}\n\nRewritten:"
    )
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=MAX_TOKENS
    )
    rewritten = response['choices'][0]['message']['content'].strip()
    return rewritten

def chunk_text(text, max_tokens=MAX_TOKENS):
    """Split a paragraph into chunks if it exceeds max_tokens."""
    words = text.split()
    chunks = []
    current_chunk = []
    token_count = 0

    for word in words:
        token_count += estimate_tokens(word)
        if token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            token_count = estimate_tokens(word)
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_font_size(para):
    if para.runs and para.runs[0].font.size:
        return para.runs[0].font.size.pt
    return DEFAULT_FONT_SIZE

def is_plain_paragraph(para):
    parent_tag = para._element.getparent().tag.split("}")[-1]
    return parent_tag == "body"

# ---------------------------
# MAIN PROCESSOR
# ---------------------------
def process_docx(input_file, output_file, warn_file):
    doc = Document(input_file)
    prev_font_size = None

    with open(output_file, "w", encoding="utf-8") as md, open(warn_file, "w", encoding="utf-8") as warn:
        paragraphs = [p for p in doc.paragraphs if p.text.strip()]
        with tqdm(total=len(paragraphs), desc="📖 Processing paragraphs") as para_bar:
            for idx, para in enumerate(paragraphs):
                text = para.text.strip()

                if not is_plain_paragraph(para):
                    warn.write(f"Paragraph {idx+1} skipped (inside box/shape/etc.): {text[:80]}...\n")
                    para_bar.update(1)
                    continue

                font_size = get_font_size(para)

                # Insert newline if font size changes or new paragraph
                md.write("\n")  # Always add a new line for paragraph change
                if prev_font_size and font_size != prev_font_size:
                    md.write("\n")  # Extra line if heading/font change
                prev_font_size = font_size

                # Heading detection
                is_heading = font_size > DEFAULT_FONT_SIZE

                # Preserve LaTeX or headings
                if text.startswith("$") and text.endswith("$"):
                    md.write(text + "\n")
                    para_bar.update(1)
                    continue
                if is_heading:
                    md.write(text + "\n")
                    para_bar.update(1)
                    continue

                # Rewrite paragraph in chunks
                chunks = chunk_text(text)
                rewritten_paragraph = ""
                with tqdm(total=len(chunks), desc=f"Rewriting paragraph {idx+1}", leave=False) as chunk_bar:
                    for chunk in chunks:
                        rewritten_chunk = rewrite_chunk(chunk)
                        rewritten_paragraph += rewritten_chunk + " "
                        chunk_bar.update(1)

                # Write rewritten paragraph with font size
                md.write(f'<span style="font-size:{font_size}pt">{rewritten_paragraph.strip()}</span>\n')
                para_bar.update(1)

        # Copy tables
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                header = rows[0]
                separator = "| " + " | ".join(["---"]*(len(rows[0].split("|"))-2)) + " |"
                md.write("\n".join([header, separator]+rows[1:]) + "\n")

        # Copy images
        for shape in doc.inline_shapes:
            md.write("![Image](image-placeholder.png)\n")

    print(f"✅ Rewritten Markdown saved as {output_file}")
    print(f"⚠️ Warning report saved as {warn_file}")

# ---------------------------
# RUN
# ---------------------------
if __name__=="__main__":
    process_docx(INPUT_DOCX, OUTPUT_MD, WARN_FILE)