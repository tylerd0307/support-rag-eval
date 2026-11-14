# small retrieval + llm + eval demo
# 1) load support articles  2) find the most relevant one for a question
# 3) ask an llm to answer using only that article  4) score every answer
import json, os, sys, time, pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

kb_dir = pathlib.Path(__file__).parent / "kb"
docs = {p.name: p.read_text() for p in sorted(kb_dir.glob("*.md"))}
names = list(docs)

# retrieval: turn every article into a word-weight vector, then compare the question to each
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform([docs[n] for n in names])

def retrieve(question, k=2, min_score=0.1):
    scores = cosine_similarity(vectorizer.transform([question]), doc_vectors)[0]
    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)[:k]
    return [(n, s) for n, s in ranked if s >= min_score]

def ask_llm(question, context):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from your environment
    prompt = (
        "answer the question using only the support articles below. "
        "name the article file you used. if the articles don't contain the answer, "
        "say exactly: i don't know based on these articles.\n\n"
        f"{context}\n\nquestion: {question}"
    )
    start = time.time()
    msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text, time.time() - start, msg.usage.input_tokens + msg.usage.output_tokens

dry_run = "--dry-run" in sys.argv  # checks retrieval only, no api key needed
questions = json.loads((pathlib.Path(__file__).parent / "questions.json").read_text())
results = []
for item in questions:
    hits = retrieve(item["q"])
    hit_names = [n for n, _ in hits]
    # retrieval is right if it found the expected article, or found nothing for an out-of-scope question
    retrieval_ok = (item["doc"] in hit_names) if item["doc"] else True
    row = {"q": item["q"], "retrieved": hit_names, "retrieval_ok": retrieval_ok}
    if not dry_run:
        context = "\n\n".join(f"[{n}]\n{docs[n]}" for n in hit_names) or "(no matching articles)"
        answer, secs, tokens = ask_llm(item["q"], context)
        low = answer.lower()
        # grounded = names the right article when it was retrieved,
        # or admits it doesn't know when the right article wasn't retrieved
        grounded = (item["doc"] in hit_names and item["doc"] in low) or \
                   (item["doc"] not in hit_names and "don't know" in low)
        row.update(answer=answer, seconds=round(secs, 2), tokens=tokens,
                   correct=item["must_include"] in low,
                   cites_source=grounded)
    results.append(row)
    print(json.dumps(row, indent=2))

n = len(results)
print(f"\nretrieval accuracy: {sum(r['retrieval_ok'] for r in results)}/{n}")
if not dry_run:
    print(f"answer accuracy:    {sum(r['correct'] for r in results)}/{n}")
    print(f"grounded (cites the retrieved article, or says it doesn't know when retrieval missed): {sum(r['cites_source'] for r in results)}/{n}")
    print(f"avg latency: {sum(r['seconds'] for r in results)/n:.2f}s   total tokens: {sum(r['tokens'] for r in results)}")