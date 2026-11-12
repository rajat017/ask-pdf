CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          SERIAL PRIMARY KEY,
  content     TEXT NOT NULL,
  embedding   vector(768),
  metadata    JSONB DEFAULT '{}',
  parent_text TEXT DEFAULT '',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_embedding
  ON documents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_documents_fts
  ON documents USING gin (to_tsvector('german', content || ' ' || parent_text));
