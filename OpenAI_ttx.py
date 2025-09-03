import os
import openai
from docx import Document
from tqdm import tqdm

# ---------------------------
# CONFIGURATION
# ---------------------------
INPUT_DOCX = "./src/input.docx"
OUTPUT_MD = "./src/rewritten.md"
DEFAULT_FONT_SIZE = 12                  # Default font size (pt) if not found
OPENAI_MODEL = "gpt-3.5-turbo"          # or "gpt-4"
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def rewrite_paragraph(text):
    """Rewrite a paragraph using OpenAI API in meaningful way."""
    prompt = (
        "Rewrite the following paragraph to be plagiarism-free, "
        "natural, and easy to read, keeping the meaning intact:\n\n"
        f"{text}\n\nRewritten:"
    )
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )
    rewritten = response['choices'][0]['message']['content'].strip()
    return rewritten

def get_paragraph_font_size(para):
    """Get font size in points; fallback to default if missing."""
    if para.runs and para.runs[0].font.size:
        return para.runs[0].font.size.pt
    return DEFAULT_FONT_SIZE

def is_plain_paragraph(para):
    """
    Detect if paragraph is "plain", i.e., not inside:
    - Text boxes
    - Shapes
    - Footnotes
    - Tables (we process separately)
    """
    # Check if parent is the main document body
    parent_tag = para._element.getparent().tag.split("}")[-1]
    return parent_tag == "body"

# ---------------------------
# MAIN DOCX PROCESSOR (Memory-Efficient)
# ---------------------------
def process_docx(input_file, output_file):
    doc = Document(input_file)
    prev_font_size = None

    with open(output_file, "w", encoding="utf-8") as f:
        # Process only plain paragraphs
        for para in tqdm(doc.paragraphs, desc="✏️ Processing paragraphs"):
            if not is_plain_paragraph(para):
                continue  # skip text inside boxes or shapes

            text = para.text.strip()
            if not text:
                continue

            font_size = get_paragraph_font_size(para)

            # Insert newline if font size changed (treat as heading)
            if prev_font_size and font_size != prev_font_size:
                f.write("\n")
            prev_font_size = font_size

            # Detect heading by font size difference
            is_heading = font_size > DEFAULT_FONT_SIZE

            # Preserve LaTeX as-is
            if text.startswith("$") and text.endswith("$"):
                f.write(text + "\n")
                continue

            # Preserve headings as-is
            if is_heading:
                f.write(text + "\n")
                continue

            # Rewrite paragraph
            rewritten = rewrite_paragraph(text)
            f.write(f'<span style="font-size:{font_size}pt">{rewritten}</span>\n')

        # ---------------------------
        # Copy tables as-is
        # ---------------------------
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                header = rows[0]
                separator = "| " + " | ".join(["---"]*(len(rows[0].split("|"))-2)) + " |"
                f.write("\n".join([header, separator]+rows[1:]) + "\n")

        # ---------------------------
        # Copy images as-is
        # ---------------------------
        for shape in doc.inline_shapes:
            f.write("![Image](image-placeholder.png)\n")

    print(f"\n✅ Rewritten Markdown saved as {output_file}")

# ---------------------------
# RUN
# ---------------------------
if __name__=="__main__":
    process_docx(INPUT_DOCX, OUTPUT_MD)