import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from docx import Document
from tqdm import tqdm
import os
import sys
import time

# ---------------------------
# CONFIGURATION
# ---------------------------
INPUT_DOCX = "./src/input.docx"        # Input Word file
OUTPUT_MD = "./src/rewritten.md"       # Output Markdown file
CHUNK_SIZE = 80                        # Words per chunk
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR = "./model"                  # Local save directory for model

# ---------------------------
# ENABLE LOGGING
# ---------------------------
logging.set_verbosity_info()      # Options: debug / info / warning / error
logging.enable_explicit_format()

os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
torch.set_printoptions(precision=4, sci_mode=False)

# ---------------------------
# LOAD / CACHE MODEL
# ---------------------------
if not os.path.exists(MODEL_DIR):
    print("=====================================================")
    print(" Downloading model from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# HELPERS
# ---------------------------
def chunk_text(text, size=CHUNK_SIZE):
    """Split long text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def llama_generate(prompt, stream=True):
    """Generate rewritten text from TinyLlama with logs and streaming output."""
    print("\n🧠 Model generating...")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if stream:
        generated = inputs["input_ids"]
        prev_text = ""

        for _ in range(512):  # max tokens
            with torch.no_grad():
                outputs = model(input_ids=generated)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            generated = torch.cat((generated, next_token_id), dim=1)

            # Decode full sequence so far
            decoded_so_far = tokenizer.decode(generated[0], skip_special_tokens=True)

            # Find new part to print
            new_text = decoded_so_far[len(prev_text):]
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                time.sleep(0.01)

            prev_text = decoded_so_far

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        print("\n✅ Streaming complete.\n")
        print(f"📝 Generated {len(prev_text.split())} words.\n")
        return prev_text

    else:
        # Normal generation
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Generated {len(decoded.split())} words.\n")
        return decoded



def rewrite_text(text, is_heading=False, is_list=False):
    """Rewrites text with logging."""
    if not text.strip():
        return text

    if is_heading:
        prompt = f"Rephrase this heading in simple human-like words, plagiarism-free:\n\n{text}\n\nNew Heading:"
    elif is_list:
        prompt = f"Rephrase this bullet point with synonyms in easy words:\n\n{text}\n\nNew Bullet:"
    else:
        prompt = f"Rephrase this paragraph in natural easy English, plagiarism-free:\n\n{text}\n\nRewritten:"

    print(f"\n🔄 Rewriting: {text[:80]}...")
    out = llama_generate(prompt, stream=True)  # STREAMING ENABLED
    return out.strip()

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
# MAIN PROCESSOR WITH PROGRESS BAR
# ---------------------------
def process_docx(input_file, output_file):
    doc = Document(input_file)
    md_lines = []

    elements = list(doc.element.body)
    total = len(elements)
    table_idx = 0

    with tqdm(total=total, desc="📖 Processing DOCX", unit="block") as pbar:
        for block in elements:
            tag = block.tag.split("}")[-1]

            if tag == "p":
                para = block
                text = para.text.strip()
                if text:
                    style_name = getattr(para.style, "name", para.style if isinstance(para.style, str) else "Normal")
                    rewritten = process_paragraph(text, style_name)
                    md_lines.append(rewritten)

            elif tag == "tbl":
                table_obj = doc.tables[table_idx]
                print("📊 Processing table...")
                md_lines.append(process_table(table_obj))
                table_idx += 1

            elif tag in ["drawing", "pict"]:
                print("🖼️ Found image placeholder")
                md_lines.append("![Image](image-placeholder.png)\n")

            else:
                md_lines.append("<!-- Other element preserved -->\n")

            pbar.update(1)
            pbar.set_postfix({"Remaining": total - pbar.n})

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n✅ Rewritten Markdown saved as {output_file}")

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    process_docx(INPUT_DOCX, OUTPUT_MD)
