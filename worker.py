#!/usr/bin/env python3
"""Cloud embed worker. Reads un-embedded rows from Neon table ide_chunks (role embed_worker: select + update vec only),
embeds them with nomic-ai/nomic-embed-text-v1.5 via FastEmbed (parity-checked against the local Qdrant collection:
cosine >= 0.995 on a 20-chunk sample), and writes the vectors back. Prints counts only, never text or credentials."""
import os, sys, time, psycopg2
from fastembed import TextEmbedding

shard, of = int(sys.argv[1]), int(sys.argv[2])
deadline = time.time() + float(os.environ.get("MAX_SECONDS", "19800"))
con = psycopg2.connect(os.environ["NEON_URL"])
con.autocommit = True
cur = con.cursor()
model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5", threads=os.cpu_count())
done = 0
while time.time() < deadline:
    cur.execute(
        "select id, text from ide_chunks where vec is null and abs(hashtext(id::text)) %% %s = %s limit 128",
        (of, shard),
    )
    rows = cur.fetchall()
    if not rows:
        break
    vecs = list(model.embed([t for _, t in rows], batch_size=32))
    for (i, _), v in zip(rows, vecs):
        cur.execute("update ide_chunks set vec = %s, embedded_at = now() where id = %s", ([float(x) for x in v], i))
    done += len(rows)
    print(f"shard {shard}/{of}: {done} embedded", flush=True)
print(f"shard {shard}/{of}: finished, {done} embedded")
