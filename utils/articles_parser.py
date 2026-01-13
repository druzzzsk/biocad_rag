import requests
import json
import time
import re
from xml.etree import ElementTree as ET


EMAIL = "your_email@example.com"
DELAY = 0.4
MIN_YEAR = 2020


def search_pmc(query, max_results=50, sort="relevance"):

    query_with_date = f"({query}) AND (2020:2025[dp])"
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pmc",
        "term": query_with_date,
        "retmax": max_results,
        "sort": sort,
        "retmode": "json",
        "email": EMAIL
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    pmc_ids = data["esearchresult"]["idlist"]

    print(f"Запрос: {query[:50]}... -> {len(pmc_ids)} статей")
    return pmc_ids


def get_text(element):

    if element is None:
        return ""

    parts = []
    if element.text:
        parts.append(element.text)

    for child in element:
        parts.append(get_text(child))
        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)  
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_heading(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'^[\d\.\)\(]+\s*', '', s)  
    s = s.strip(" :.-\n\t")
    return s


def extract_sec_text(sec) -> str:
    ps = sec.findall(".//p")
    if ps:
        parts = []
        for p in ps:
            t = clean_text(get_text(p))
            if t:
                parts.append(t)
        return " ".join(parts).strip()

    return clean_text(get_text(sec))


def fetch_pmc_article(pmc_id):

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmc_id,
        "retmode": "xml",
        "email": EMAIL
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception as e:
        print(f"Ошибка загрузки PMC {pmc_id}: {e}")
        return None

    try:
        article = root.find(".//article")
        if article is None:
            return None

        pmc_id_clean = ""
        for aid in root.findall(".//article-id"):
            if aid.get("pub-id-type") == "pmc":
                pmc_id_clean = (aid.text or "").strip()
                break

        if not pmc_id_clean:
            pmc_id_clean = str(pmc_id).strip()

        pmc_num = pmc_id_clean.replace("PMC", "")

        title_elem = root.find(".//article-title")
        title = clean_text(get_text(title_elem))

        authors = []
        for contrib in root.findall(".//contrib[@contrib-type='author']"):
            surname = contrib.find(".//surname")
            given = contrib.find(".//given-names")
            if surname is not None and (surname.text or "").strip():
                name = (surname.text or "").strip()
                if given is not None and (given.text or "").strip():
                    name = (given.text or "").strip() + " " + name
                authors.append(name)

        year = ""
        for y in root.findall(".//pub-date/year"):
            if y is not None and (y.text or "").strip():
                year = (y.text or "").strip()
                break

        if year:
            try:
                if int(year) < MIN_YEAR:
                    return None
            except ValueError:
                pass

        abstract_parts = []
        for аб in root.findall(".//abstract"):
            txt = extract_sec_text(аб)
            if txt and len(txt) > 20:
                abstract_parts.append(txt)
        abstract = " ".join(abstract_parts).strip()

        intro_parts = []
        concl_parts = []

        intro_keys = {
            "introduction", "background", "objective", "objectives", "aim", "aims", "purpose"
        }
        concl_keys = {
            "conclusion", "conclusions", "concluding remarks", "summary", "final remarks",
            "implications", "key points", "closing remarks"
        }
        discussion_keys = {
            "discussion", "discussion and conclusion", "discussion & conclusion"
        }

        for sec in root.findall(".//sec"):
            sec_type = (sec.get("sec-type") or "").lower().strip()

            title_elem = sec.find("title")
            sec_title = normalize_heading(get_text(title_elem)) if title_elem is not None else ""

            text = extract_sec_text(sec)
            if not text or len(text) < 50:
                continue

            if sec_type in ("intro", "introduction", "background"):
                intro_parts.append(text)
                continue

            if sec_type in ("conclusion", "conclusions"):
                concl_parts.append(text)
                continue

            if sec_title and (sec_title in intro_keys or any(k in sec_title for k in intro_keys)):
                intro_parts.append(text)
                continue

            if sec_title and (sec_title in concl_keys or any(k in sec_title for k in concl_keys)):
                concl_parts.append(text)
                continue

        introduction = " ".join(intro_parts).strip()
        conclusion = " ".join(concl_parts).strip()

        if not conclusion:
            for sec in root.findall(".//sec"):
                title_elem = sec.find("title")
                sec_title = normalize_heading(get_text(title_elem)) if title_elem is not None else ""
                if sec_title and (sec_title in discussion_keys or "discussion" in sec_title):
                    text = extract_sec_text(sec)
        
                    if text and len(text) > 200:
                        conclusion = text
                        break

        url_pmc = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_num}/"

        return {
            "pmc_id": pmc_num,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": clean_text(abstract),
            "introduction": clean_text(introduction),
            "conclusion": clean_text(conclusion),
            "url": url_pmc
        }

    except Exception as e:
        print(f"Ошибка парсинга PMC {pmc_id}: {e}")
        return None


def save_to_jsonl(articles, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    print(f"Сохранено {len(articles)} статей в {filename}")


def main():
    queries = [
        "Alzheimer's disease targets",
        "Alzheimer therapeutic targets",
        "Alzheimer drug targets",
    ]

    all_pmc_ids = []
    seen_ids = set()


    print(f"=== Поиск в PMC (с {MIN_YEAR} года) ===\n")

    for query in queries:
        pmc_ids = search_pmc(query, max_results=33, sort="relevance")
        time.sleep(DELAY)

        for pmc_id in pmc_ids:
            if pmc_id not in seen_ids:
                seen_ids.add(pmc_id)
                all_pmc_ids.append(pmc_id)

    print(f"\Найдено уникальных статей: {len(all_pmc_ids)}")

    print("\n=== Загрузка статей из PMC ===\n")

    articles = []

    for i, pmc_id in enumerate(all_pmc_ids):
        print(f"[{i+1}/{len(all_pmc_ids)}] Загрузка PMC: {pmc_id}")

        article = fetch_pmc_article(pmc_id)
        time.sleep(DELAY)

        if article:
            articles.append(article)

    print(f"\nУспешно загружено: {len(articles)} статей")

    final_articles = []

    for article in articles:
        parts = []

        if article["introduction"]:
            parts.append("[INTRODUCTION] " + article["introduction"])

        if article["abstract"]:
            parts.append("[ABSTRACT] " + article["abstract"])

        if article["conclusion"]:
            parts.append("[CONCLUSION] " + article["conclusion"])

        content = " ".join(parts)

        if len(content) < 100:
            continue

        final_articles.append({
            "id": int(article['pmc_id']),
            "title": article["title"],
            "authors": article["authors"],
            "year": article["year"],
            "content": content,
            "url": article["url"]
        })

    print(f"Статей с контентом: {len(final_articles)}")

    save_to_jsonl(final_articles, "data/articles.jsonl")

    return final_articles


if __name__ == "__main__":
    articles = main()