import os
import re

def extract_10k_document(filepath):
    """Extract the main 10-K HTML document from SGML full-submission.txt"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Find all documents inside the SGML wrapper
    documents = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', content, re.DOTALL)

    for doc in documents:
        # Get document type
        type_match = re.search(r'<TYPE>(.*?)\n', doc)
        if not type_match:
            continue

        doc_type = type_match.group(1).strip()

        # We want the main 10-K document only
        if doc_type == "10-K":
            # Extract the text body
            text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
            if text_match:
                return text_match.group(1).strip()

    return None

def extract_text_from_html(html_content):
    """Strip HTML tags and extract clean text"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)

def extract_tables_from_html(html_content):
    """Extract tables as structured data"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    tables = []

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):  # skip empty rows
                rows.append(cells)
        if rows:
            tables.append(rows)

    return tables

def table_to_markdown(table):
    """Convert table rows to markdown string"""
    if not table:
        return ""
    header = "| " + " | ".join(str(c) for c in table[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
    rows = ["| " + " | ".join(str(c) for c in row) + " |" for row in table[1:]]
    return "\n".join([header, separator] + rows)

def process_filing(filepath, company, year, filing_type="10-K"):
    """Full pipeline: SGML → extract → parse → chunks with metadata"""
    print(f"  Processing {company} {year}...")

    raw_doc = extract_10k_document(filepath)
    if not raw_doc:
        print(f"  ⚠ Could not extract 10-K document from {filepath}")
        return []

    chunks = []

    # Extract and chunk tables
    tables = extract_tables_from_html(raw_doc)
    for i, table in enumerate(tables):
        md = table_to_markdown(table)
        if len(md) > 50:  # skip tiny/empty tables
            chunks.append({
                "text": md,
                "type": "table",
                "company": company,
                "year": year,
                "filing_type": filing_type,
                "page": i  # table index as proxy for position
            })

    # Extract and chunk plain text
    text = extract_text_from_html(raw_doc)
    # Split into ~1000 char chunks with overlap
    chunk_size = 1000
    overlap = 100
    words = text.split()
    current_chunk = []
    current_len = 0

    for word in words:
        current_chunk.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append({
                "text": " ".join(current_chunk),
                "type": "text",
                "company": company,
                "year": year,
                "filing_type": filing_type,
                "page": 0
            })
            # Keep last overlap chars
            overlap_words = current_chunk[-20:]
            current_chunk = overlap_words
            current_len = sum(len(w) + 1 for w in overlap_words)

    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "type": "text",
            "company": company,
            "year": year,
            "filing_type": filing_type,
            "page": 0
        })

    print(f"  ✓ {len(chunks)} chunks ({len(tables)} tables + text)")
    return chunks


# --- Run on all downloaded filings ---
filings = [
    ("./data/raw/sec-edgar-filings/AAPL/10-K/0000320193-24-000123/full-submission.txt", "AAPL", "2024"),
    ("./data/raw/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt", "AAPL", "2025"),
    ("./data/raw/sec-edgar-filings/GS/10-K/0000886982-25-000005/full-submission.txt",   "GS",   "2024"),
    ("./data/raw/sec-edgar-filings/GS/10-K/0000886982-26-000091/full-submission.txt",   "GS",   "2025"),
    ("./data/raw/sec-edgar-filings/MSFT/10-K/0000950170-24-087843/full-submission.txt", "MSFT", "2024"),
    ("./data/raw/sec-edgar-filings/MSFT/10-K/0000950170-25-100235/full-submission.txt", "MSFT", "2025"),
]

all_chunks = []
print("Parsing filings...\n")
for filepath, company, year in filings:
    chunks = process_filing(filepath, company, year)
    all_chunks.extend(chunks)

print(f"\nTotal chunks: {len(all_chunks)}")
print(f"Table chunks: {sum(1 for c in all_chunks if c['type'] == 'table')}")
print(f"Text chunks:  {sum(1 for c in all_chunks if c['type'] == 'text')}")