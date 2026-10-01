"""Phase 2 search for docs/review/protocol.md (§B.4, §B.5).

Runs the four frozen search strings against OpenAlex, PubMed and arXiv and
writes one deduplicated JSONL record per work to docs/review/data/.
Usage: python docs/review/search.py
"""

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

OUT = Path(__file__).parent / "data"
UA = {"User-Agent": "map-stdp-review (mailto:vikramr2@illinois.edu)"}

# Each block is a list of OR-terms; a string is the AND of its blocks (§B.5).
NEURO_A = ["neuron", "neurons", "neural", "synapse", "synaptic", "axon", "brain", "cortex", "spiking"]
# Optional exclusion block and title-only scope, per string (protocol §B.13, amendment 3).
NOT = {"S-A": ["deep learning", "neural network accelerator"]}
TITLE_ONLY = {"S-D"}
STRINGS = {
    "S-A": [
        ["mutual information", "information rate", "bits per", "description length",
         "map equation", "coding efficiency", "information transmission",
         "transfer entropy", "entropy rate"],
        ["metabolic cost", "energy cost", "energetic cost", "energy efficiency",
         "energy consumption", "ATP", "entropy production"],
        NEURO_A,
    ],
    "S-B": [
        ["neuron", "neurons", "neural", "synapse", "synaptic", "axon", "brain",
         "cortex", "cortical"],
        ["metabolic cost", "energy cost", "energetic cost", "wiring cost", "metabolic"],
        ["wiring cost", "wiring economy", "communication cost", "network economy",
         "modular organization", "brain modularity", "network modularity"],
    ],
    "S-C1": [
        ["energy budget", "ATP consumption", "ATP cost", "energy use"],
        ["action potential", "action potentials", "synaptic transmission",
         "resting potential", "grey matter", "gray matter", "signalling", "signaling"],
        ["neuron", "neurons", "brain", "cortex"],
    ],
    "S-C2": [
        ["entropy production", "Landauer", "stochastic thermodynamics",
         "nonequilibrium thermodynamics", "non-equilibrium thermodynamics"],
        ["neuron", "neurons", "synapse", "synaptic", "neural activity", "brain dynamics",
         "human brain", "brain activity", "whole-brain"],
    ],
    "S-D": [
        ["neural information", "neural code", "neural codes", "neural coding",
         "cortical computation", "neural computation", "neural signalling", "neural signaling"],
        ["metabolic cost", "energy cost", "energetic cost", "energy efficiency",
         "energy consumption", "metabolic", "cost of"],
    ],
}


def get(url, params=None, retries=5):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(2 * (k + 1))
    raise RuntimeError(url)


def q(term, fmt):
    return fmt.format(f'"{term}"' if " " in term or "-" in term else term)


def build(blocks, fmt, sid, neg="NOT"):
    expr = " AND ".join("(" + " OR ".join(q(t, fmt) for t in b) + ")" for b in blocks)
    if sid in NOT:
        expr += f" {neg} (" + " OR ".join(q(t, fmt) for t in NOT[sid]) + ")"
    return expr


def openalex(blocks, sid):
    expr = build(blocks, "{}", sid)
    field = "title.search" if sid in TITLE_ONLY else "title_and_abstract.search"
    cursor, out = "*", []
    while cursor:
        time.sleep(1.2)
        d = json.loads(get("https://api.openalex.org/works", {
            "filter": f"{field}:{expr}", "per-page": 200, "cursor": cursor,
            "select": "id,doi,title,publication_year,type,abstract_inverted_index,"
                      "primary_location,authorships"}))
        for w in d["results"]:
            inv = w.get("abstract_inverted_index") or {}
            pos = sorted((i, word) for word, idx in inv.items() for i in idx)
            src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            out.append({
                "openalex": w["id"], "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower() or None,
                "title": w.get("title"), "year": w.get("publication_year"), "type": w.get("type"),
                "venue": src, "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:6]],
                "abstract": " ".join(word for _, word in pos)})
        cursor = d["meta"].get("next_cursor") if d["results"] else None
    return out


def pubmed(blocks, sid):
    expr = build(blocks, "{}[ti]" if sid in TITLE_ONLY else "{}[tiab]", sid)
    ids = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {
        "db": "pubmed", "term": expr, "retmax": 10000, "retmode": "json"}))["esearchresult"]["idlist"]
    out = []
    for i in range(0, len(ids), 200):
        time.sleep(0.5)
        root = ET.fromstring(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", {
            "db": "pubmed", "id": ",".join(ids[i:i + 200]), "retmode": "xml"}))
        for a in root.iter("PubmedArticle"):
            art = a.find(".//Article")
            doi = a.find(".//ArticleIdList/ArticleId[@IdType='doi']")
            yr = a.find(".//PubDate/Year")
            out.append({
                "pmid": a.findtext(".//PMID"), "doi": doi.text.lower() if doi is not None and doi.text else None,
                "title": "".join(art.find("ArticleTitle").itertext()), "year": int(yr.text) if yr is not None else None,
                "type": "article", "venue": art.findtext("Journal/Title"),
                "authors": [f"{x.findtext('LastName')} {x.findtext('Initials')}" for x in art.findall(".//Author")[:6]],
                "abstract": " ".join("".join(t.itertext()) for t in art.findall(".//AbstractText"))})
    return out


def arxiv(blocks, sid):
    expr = build(blocks, "ti:{}" if sid in TITLE_ONLY else "abs:{}", sid, neg="ANDNOT")
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    out, start = [], 0
    while True:
        time.sleep(3.5)
        root = ET.fromstring(get("http://export.arxiv.org/api/query", {
            "search_query": expr, "start": start, "max_results": 200}))
        entries = root.findall("a:entry", ns)
        for e in entries:
            doi = e.findtext("x:doi", namespaces=ns)
            out.append({
                "arxiv": e.findtext("a:id", namespaces=ns).split("/abs/")[-1],
                "doi": doi.lower() if doi else None, "title": " ".join(e.findtext("a:title", namespaces=ns).split()),
                "year": int(e.findtext("a:published", namespaces=ns)[:4]), "type": "preprint", "venue": "arXiv",
                "authors": [x.findtext("a:name", namespaces=ns) for x in e.findall("a:author", ns)[:6]],
                "abstract": " ".join(e.findtext("a:summary", namespaces=ns).split())})
        if len(entries) < 200:
            return out
        start += 200


def norm(title):
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def main():
    OUT.mkdir(exist_ok=True)
    raw, log = [], {"date": date.today().isoformat(), "counts": {}}
    for sid, blocks in STRINGS.items():
        for name, fn in [("openalex", openalex), ("pubmed", pubmed), ("arxiv", arxiv)]:
            recs = fn(blocks, sid)
            log["counts"][f"{sid}/{name}"] = len(recs)
            print(sid, name, len(recs), flush=True)
            for r in recs:
                r["hits"] = [f"{sid}/{name}"]
            raw += recs
    # §B.6 step 1: deduplicate by DOI, then by normalised title + year.
    by_key, merged = {}, []
    for r in raw:
        keys = [k for k in (r.get("doi") and "doi:" + r["doi"],
                            norm(r["title"]) and f"t:{norm(r['title'])}:{r.get('year')}") if k]
        hit = next((by_key[k] for k in keys if k in by_key), None)
        if hit is None:
            hit = {"rid": f"R{len(merged) + 1:04d}", **r}
            merged.append(hit)
        else:
            hit["hits"] = sorted(set(hit["hits"]) | set(r["hits"]))
            for f in ("doi", "openalex", "pmid", "arxiv", "venue"):
                hit[f] = hit.get(f) or r.get(f)
            if len(r.get("abstract") or "") > len(hit.get("abstract") or ""):
                hit["abstract"] = r["abstract"]
        for k in keys:
            by_key[k] = hit
    log["raw"], log["deduplicated"] = len(raw), len(merged)
    with open(OUT / "records.jsonl", "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    (OUT / "search_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
