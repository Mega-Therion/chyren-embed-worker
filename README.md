# chyren-embed-worker

Free cloud embedding for Chyren's second brain. A scheduled GitHub Actions matrix (20 jobs) reads un-embedded text chunks from a Neon table, embeds them with `nomic-ai/nomic-embed-text-v1.5` (FastEmbed, CPU) and writes the vectors back. A local puller then upserts them into the local Qdrant collection, so the local CPU does no embedding.

No data lives in this repository. The worker connects through a database role that can only read the table and write vectors. Logs print counts only.

R.W. Yett · Arkansas
