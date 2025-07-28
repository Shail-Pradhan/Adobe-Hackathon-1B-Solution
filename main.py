import os
import json
import datetime
import re
import sys

import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

INPUT_DIR = "input"
OUTPUT_DIR = "output"
INPUT_FILE = os.path.join(INPUT_DIR, "input.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")
MAX_SECTIONS = 5

AVOID_SECTIONS = [
    "introduction", "conclusion", "contents", "summary", "about the authors", "references", "table of contents"
]

# Mapping from common job/persona phrases to section-title focus keywords
JOB_SECTION_KEYWORDS = {
    "literature review": [
        "methodology", "methods", "dataset", "data set", "benchmark", "performance", "experiment", "results",
        "discussion", "evaluation", "comparative", "study", "review"
    ],
    "drug discovery": [
        "graph neural network", "gnn", "drug", "compound", "bioactivity"
    ],
    "business analysis": [
        "revenue", "income", "financial", "r&d", "research and development", "market", "strategy",
        "position", "overview", "investments", "growth", "segment"
    ],
    "exam preparation": [
        "key concept", "summary", "mechanism", "reaction", "practice", "important", "exam", "problem", "review",
        "study guide", "kinetics"
    ],
    "reaction kinetics": [
        "mechanism", "rate", "law", "activation energy", "transition state", "arrhenius", "order", "reaction", "kinetics"
    ],
    # Add more as needed
}

# For vegetarian/vegan/gluten-free filtering
MEAT_WORDS = [
    "chicken", "beef", "pork", "lamb", "fish", "shrimp", "crab", "turkey", "duck", "seafood",
    "bacon", "ham", "anchovy", "anchovies", "sausage", "meat", "steak", "mutton", "octopus", "calamari", "goat", "veal"
]
ANIMAL_PRODUCTS = [
    "egg", "cheese", "milk", "butter", "cream", "yogurt", "honey"
]
GLUTEN_WORDS = [
    "wheat", "barley", "rye", "bread", "noodle", "pasta", "bun", "cake", "cracker", "biscuit", "flour",
    "couscous", "semolina", "spaghetti", "ravioli", "crouton"
]

def extract_text_from_pdf(pdf_path):
    texts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
            except Exception:
                text = ""
            texts.append({"page_number": i + 1, "text": text or ""})
    return texts

def split_into_sections(text, page_number):
    lines = text.split('\n')
    sections = []
    current_section = {"title": None, "text": "", "page_number": page_number}
    for line in lines:
        # Accept numbered, bulleted, or title-case section titles
        if re.match(r'^([0-9]+\.|[A-Z][A-Za-z0-9\s\-\&\,]+)$', line.strip()) and 6 < len(line.strip()) < 80:
            if current_section["title"] is not None and current_section["text"].strip():
                sections.append(current_section)
            current_section = {"title": line.strip(), "text": "", "page_number": page_number}
        else:
            current_section["text"] += line + "\n"
    if current_section["title"] is not None and current_section["text"].strip():
        sections.append(current_section)
    return sections

def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("•", "-")
    text = text.replace("–", "-")
    text = text.strip()
    return text

def refine_text(text):
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)-\s*', text)
    sentences = [s.strip(" -•") for s in sentences if s.strip(" -•")]
    start_idx = 0
    for i, s in enumerate(sentences):
        if len(s) > 30 and s[0].isupper():
            start_idx = i
            break
    refined = " ".join(sentences[start_idx:start_idx+7])
    if not refined and sentences:
        refined = " ".join(sentences[:7])
    if len(refined) > 800:
        refined = refined[:800] + "..."
    return refined

def extract_focus_keywords(persona, job):
    base = (persona + " " + job).lower()
    keywords = []
    for phrase, kws in JOB_SECTION_KEYWORDS.items():
        if phrase in base:
            keywords += kws
    words = [w for w in re.findall(r'\w+', base) if len(w) > 4]
    keywords += words
    return list(set(keywords))

def get_dietary_filters(job_to_be_done):
    job_lower = job_to_be_done.lower()
    dietary = {
        "vegetarian": False,
        "vegan": False,
        "gluten_free": False,
        "exclude_meat": set(),
        "exclude_gluten": set(),
    }
    if "vegetarian" in job_lower:
        dietary["vegetarian"] = True
        dietary["exclude_meat"] = set(MEAT_WORDS)
    if "vegan" in job_lower:
        dietary["vegan"] = True
        dietary["exclude_meat"] = set(MEAT_WORDS + ANIMAL_PRODUCTS)
    if "gluten" in job_lower and "free" in job_lower:
        dietary["gluten_free"] = True
        dietary["exclude_gluten"] = set(GLUTEN_WORDS)
    return dietary

def extract_ingredients(text):
    text = text.lower()
    ingredients = set()
    lines = text.split('\n')
    started = False
    for line in lines:
        l = line.strip()
        if 'ingredient' in l:
            started = True
            continue
        if started:
            if l == "" or len(l) > 100 or any(h in l for h in ["instruction", "method", "directions", "how to", "step"]):
                break
            if re.match(r'^[-•o*]?\s*[\w\s,]+', l):
                for entry in re.split(r',| and |/', l):
                    entry = entry.strip(" -•o*").strip()
                    if 2 < len(entry) < 40:
                        ingredients.add(entry)
    return ingredients

def section_is_allowed(section, dietary):
    if not (dietary["vegetarian"] or dietary["vegan"] or dietary["gluten_free"]):
        return True
    text = ((section.get("section_title") or "") + "\n" + (section.get("text") or "")).lower()
    ingredients = extract_ingredients(section.get("text") or "")
    for forbidden in dietary["exclude_meat"]:
        pattern = r'\b' + re.escape(forbidden) + r's?\b'
        if any(re.search(pattern, ing) for ing in ingredients):
            return False
        if re.search(pattern, text):
            return False
    if dietary["gluten_free"]:
        for forbidden in dietary["exclude_gluten"]:
            pattern = r'\b' + re.escape(forbidden) + r's?\b'
            if any(re.search(pattern, ing) for ing in ingredients):
                return False
            if re.search(pattern, text):
                return False
    return True

def section_priority(section, doc_title, persona, job_to_be_done, tfidf_score, doc_tfidf_score, focus_keywords):
    score = 0
    title = (section["section_title"] or "").lower()
    doc = doc_title.lower()
    for i, kw in enumerate(focus_keywords):
        if kw in title:
            score -= 100 - i
        elif kw in doc:
            score -= 10 - i
    if title in AVOID_SECTIONS:
        score += 20
    score -= doc_tfidf_score * 3 + tfidf_score
    return score

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    doc_infos = input_data["documents"]
    persona = input_data["persona"]["role"]
    job_to_be_done = input_data["job_to_be_done"]["task"]
    timestamp = datetime.datetime.utcnow().isoformat()
    focus_keywords = extract_focus_keywords(persona, job_to_be_done)
    dietary = get_dietary_filters(job_to_be_done)
    all_sections = []
    doc_text_map = {}
    for doc in doc_infos:
        pdf_path = os.path.join(INPUT_DIR, doc["filename"])
        if not os.path.isfile(pdf_path):
            continue
        pages = extract_text_from_pdf(pdf_path)
        doc_full_text = " ".join([p["text"] for p in pages])
        doc_text_map[doc["title"]] = doc_full_text
        for page in pages:
            secs = split_into_sections(page["text"], page["page_number"])
            for s in secs:
                section = {
                    "document": doc["title"],
                    "page_number": s["page_number"],
                    "section_title": s["title"],
                    "text": s["text"]
                }
                if section_is_allowed(section, dietary):
                    all_sections.append(section)
    if not all_sections:
        print("No valid sections found.")
        sys.exit(1)
    query = f"{persona} planning: {job_to_be_done}"
    doc_titles = list(doc_text_map.keys())
    doc_texts = list(doc_text_map.values())
    doc_vectorizer = TfidfVectorizer(stop_words="english").fit(doc_texts + [query])
    doc_query_vec = doc_vectorizer.transform([query])
    doc_vecs = doc_vectorizer.transform(doc_texts)
    doc_scores = (doc_vecs * doc_query_vec.T).toarray().flatten()
    doc_score_map = dict(zip(doc_titles, doc_scores))
    section_texts = [s["section_title"] + " " + s["text"] for s in all_sections]
    sec_vectorizer = TfidfVectorizer(stop_words="english").fit(section_texts + [query])
    sec_query_vec = sec_vectorizer.transform([query])
    sec_vecs = sec_vectorizer.transform(section_texts)
    sec_scores = (sec_vecs * sec_query_vec.T).toarray().flatten()
    section_rankings = []
    for i, section in enumerate(all_sections):
        doc_title = section["document"]
        doc_score = doc_score_map.get(doc_title, 0)
        sec_score = sec_scores[i]
        score = section_priority(section, doc_title, persona, job_to_be_done, sec_score, doc_score, focus_keywords)
        section_rankings.append((score, section))
    picked_sections = []
    picked_titles = set()
    doc_seen = set()
    for score, s in sorted(section_rankings, key=lambda x: x[0]):
        if (s["document"], s["section_title"]) in picked_titles:
            continue
        if s["document"] not in doc_seen:
            picked_sections.append(s)
            picked_titles.add((s["document"], s["section_title"]))
            doc_seen.add(s["document"])
        if len(picked_sections) >= MAX_SECTIONS:
            break
    if len(picked_sections) < MAX_SECTIONS:
        for score, s in sorted(section_rankings, key=lambda x: x[0]):
            if (s["document"], s["section_title"]) not in picked_titles:
                picked_sections.append(s)
                picked_titles.add((s["document"], s["section_title"]))
            if len(picked_sections) >= MAX_SECTIONS:
                break
    extracted_sections = []
    sub_section_analysis = []
    for i, s in enumerate(picked_sections[:MAX_SECTIONS]):
        extracted_sections.append({
            "document": s["document"],
            "page_number": s["page_number"],
            "section_title": s["section_title"],
            "importance_rank": i+1
        })
        refined_txt = refine_text(s["text"])
        sub_section_analysis.append({
            "document": s["document"],
            "refined_text": refined_txt,
            "page_number": s["page_number"]
        })
    output = {
        "metadata": {
            "input_documents": [doc["title"] for doc in doc_infos],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": timestamp
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": sub_section_analysis
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"Extraction complete. Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()