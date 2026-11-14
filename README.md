# support rag eval

a small demo of retrieval-augmented generation (rag) with an evaluation step.
sample support articles in `kb/` are made up for testing.

## run
pip install anthropic scikit-learn
export ANTHROPIC_API_KEY=your-key   # windows powershell: $env:ANTHROPIC_API_KEY="your-key"
python rag_eval.py --dry-run        # retrieval only, free
python rag_eval.py                  # full run, costs a few cents

## what it measures
- retrieval accuracy: did it pull the right article
- answer accuracy: does the answer contain the key fact
- groundedness: does the answer name its source, or admit it doesn't know when the answer isn't there
- latency and tokens (cost) per question
