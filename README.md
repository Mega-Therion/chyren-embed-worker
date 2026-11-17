# chyren-embed-worker

Cloud and local embedding bridge for Chyren's second brain:

- **`worker.py`** (GitHub Actions 20-shard matrix via `.github/workflows/embed.yml`): reads un-embedded text chunks from Neon `ide_chunks`, embeds them on GitHub CPUs with `nomic-ai/nomic-embed-text-v1.5` (FastEmbed), and writes vectors back to Neon.
- **`ingest_to_brain.py`**: extracts `.md`, `.txt`, `.pdf`, `.html`, `.json`, `.csv`, and `.zip` archives, chunks them, writes records into `~/.chyren/knowledge.db` and atomic notes into both `~/.chyren/vault/20_Sources/` and `~/Chyren/2_knowledge/Chyren_Second_Brain/20_Sources/`, and either:
  - `--mode local`: embeds locally via Ollama (`nomic-embed-text:latest`) and upserts to Qdrant (`chyren_memory_knowledge` + `chyren_memory`), or
  - `--mode cloud-push` / `--mode cloud-pull`: pushes chunks to Neon `ide_chunks` for `worker.py` and pulls completed vectors back into Qdrant.

No data lives in this repository. Credentials come only from environment variables (`NEON_URL` / `CHYREN_DB_URL_NEON_CH3`).

R.W. Yett · Arkansas
