#Execute the following code in the console to RUN extract_features.py.

from pathlib import Path
from extract_features import do_tfidf, do_w2v, do_electra

csv = Path(r"E:\sml\imdb_clean.csv")
out = csv.parent / "features"

do_tfidf(csv, out)                                   # ① TF‑IDF
do_w2v(csv, out, vec_size=100, epochs=5)             # ② 平均词向量（gensim）
do_electra(csv, out, batch_size=16, max_len=256, pool="cls")  # ③ ELECTRA 句向量