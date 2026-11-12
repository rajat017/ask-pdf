package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgvector "github.com/pgvector/pgvector-go"
)

type pgvecStore struct {
	pool *pgxpool.Pool

	emb ai.Embedder

	retriever ai.Retriever

	dim int

	embeddingModel string
}

func newPgvecStore(ctx context.Context, g *genkit.Genkit, emb ai.Embedder, dsn string, dim int) (*pgvecStore, error) {

	pool, err := pgxpool.New(ctx, dsn)

	if err != nil {

		return nil, fmt.Errorf("pgxpool.New: %w", err)

	}

	if err := pool.Ping(ctx); err != nil {

		pool.Close()

		return nil, fmt.Errorf("pgvector ping: %w", err)

	}

	s := &pgvecStore{

		pool: pool,

		emb: emb,

		dim: dim,

		embeddingModel: strings.TrimSpace(os.Getenv("EMBEDDING_MODEL")),
	}

	if err := s.ensureSchema(ctx); err != nil {

		pool.Close()

		return nil, err

	}

	log.Printf("[pgvecStore] embedding model formatting: %s", s.embeddingModel)

	s.retriever = genkit.DefineRetriever(g, "pgvec/docQA", nil, s.retrieve)

	return s, nil

}

func normalizeTags(tags []string) []string {

	var out []string

	seen := map[string]bool{}

	for _, t := range tags {

		t = strings.ToLower(strings.TrimSpace(t))

		if t == "" || seen[t] {

			continue

		}

		seen[t] = true

		out = append(out, t)

	}

	return out

}

func (s *pgvecStore) usesHalfVec() bool {

	return s.dim > 2000

}

func (s *pgvecStore) embeddingDBValue(values []float32) any {

	if s.usesHalfVec() {

		return pgvector.NewHalfVector(values)

	}

	return pgvector.NewVector(values)

}

func (s *pgvecStore) embeddingType() string {

	if s.usesHalfVec() {

		return fmt.Sprintf("halfvec(%d)", s.dim)

	}

	return fmt.Sprintf("vector(%d)", s.dim)

}

func (s *pgvecStore) embeddingCosineOps() string {

	if s.usesHalfVec() {

		return "halfvec_cosine_ops"

	}

	return "vector_cosine_ops"

}

func (s *pgvecStore) ensureSchema(ctx context.Context) error {

	const createExtension = `CREATE EXTENSION IF NOT EXISTS vector;`

	if _, err := s.pool.Exec(ctx, createExtension); err != nil {

		return fmt.Errorf("ensureSchema create extension: %w", err)

	}

	wantType := s.embeddingType()

	cosineOps := s.embeddingCosineOps()

	var tableExists bool

	err := s.pool.QueryRow(ctx, `

SELECT EXISTS (

    SELECT 1

    FROM information_schema.tables

    WHERE table_schema = 'public'

      AND table_name = 'documents'

)

`).Scan(&tableExists)

	if err != nil {

		return fmt.Errorf("ensureSchema check table exists: %w", err)

	}

	if tableExists {

		var existingType string

		err := s.pool.QueryRow(ctx, `

SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)

FROM pg_attribute a

JOIN pg_class c ON a.attrelid = c.oid

JOIN pg_namespace n ON c.relnamespace = n.oid

WHERE n.nspname = 'public'

  AND c.relname = 'documents'

  AND a.attname = 'embedding'

  AND a.attnum > 0

  AND NOT a.attisdropped

`).Scan(&existingType)

		if err != nil {

			return fmt.Errorf("ensureSchema read existing embedding type: %w", err)

		}

		// Changing embedding model/dimension invalidates all stored vectors.

		// Recreate the documents table automatically so the caller can re-index.

		if existingType != wantType {

			log.Printf(

				"[pgvecStore] embedding storage mismatch: have %s, want %s; dropping documents table so it can be re-indexed",

				existingType, wantType,
			)

			if _, err := s.pool.Exec(ctx, `DROP TABLE documents`); err != nil {

				return fmt.Errorf("ensureSchema drop incompatible documents table: %w", err)

			}

			tableExists = false

		}

	}

	if !tableExists {

		createTable := fmt.Sprintf(`

CREATE TABLE documents (

  id          SERIAL PRIMARY KEY,

  content     TEXT NOT NULL,

  embedding   %s,

  metadata    JSONB DEFAULT '{}',

  parent_text TEXT DEFAULT '',

  created_at  TIMESTAMPTZ DEFAULT NOW()

)`, wantType)

		if _, err := s.pool.Exec(ctx, createTable); err != nil {

			return fmt.Errorf("ensureSchema create documents table: %w", err)

		}

	}

	createEmbeddingIndex := fmt.Sprintf(`

CREATE INDEX IF NOT EXISTS idx_documents_embedding

  ON documents USING hnsw (embedding %s)`, cosineOps)

	if _, err := s.pool.Exec(ctx, createEmbeddingIndex); err != nil {

		return fmt.Errorf("ensureSchema create embedding index: %w", err)

	}

	if _, err := s.pool.Exec(ctx, `

CREATE INDEX IF NOT EXISTS idx_documents_fts

  ON documents USING gin (

    (

      to_tsvector('english', content || ' ' || parent_text) ||

      to_tsvector('german',  content || ' ' || parent_text)

    )

  )

`); err != nil {

		return fmt.Errorf("ensureSchema create FTS index: %w", err)

	}

	if _, err := s.pool.Exec(ctx, `

CREATE INDEX IF NOT EXISTS idx_documents_tag_lower

  ON documents ((LOWER(metadata->>'tag')))

`); err != nil {

		return fmt.Errorf("ensureSchema create tag index: %w", err)

	}

	log.Printf(

		"[pgvecStore] schema ensured (embedding type=%s, hnsw ops=%s)",

		wantType, cosineOps,
	)

	return nil

}

func (s *pgvecStore) formatDocumentForEmbedding(text string) string {

	model := strings.ToLower(strings.TrimSpace(s.embeddingModel))

	switch {

	case strings.Contains(model, "nomic"):

		return "search_document: " + text

	case strings.Contains(model, "qwen3-embedding"):

		// Qwen3-Embedding documents should be embedded as plain document text.

		return text

	default:

		return text

	}

}

func (s *pgvecStore) formatQueryForEmbedding(query string) string {

	model := strings.ToLower(strings.TrimSpace(s.embeddingModel))

	switch {

	case strings.Contains(model, "nomic"):

		return "search_query: " + query

	case strings.Contains(model, "qwen3-embedding"):

		// Qwen3-Embedding benefits from an explicit retrieval-task instruction

		// on the query side; documents remain unprefixed.

		return "Instruct: Given a question, retrieve relevant passages from the document that answer the question\nQuery: " + query

	default:

		return query

	}

}

func (s *pgvecStore) Index(ctx context.Context, docs []*ai.Document) error {
	if len(docs) == 0 {
		return nil
	}

	batchSize := 8
	if raw := strings.TrimSpace(os.Getenv("EMBED_BATCH_SIZE")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			batchSize = parsed
		}
	}
	if batchSize > 32 {
		batchSize = 32
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	const q = `INSERT INTO documents (content, embedding, metadata, parent_text) VALUES ($1, $2, $3, $4)`

	type preparedDocument struct {
		contentStr string
		parentText string
		metaJSON   []byte
		embedDoc   *ai.Document
		sourceIdx  int
	}

	prepared := make([]preparedDocument, 0, len(docs))

	for i, doc := range docs {
		var content strings.Builder
		for _, part := range doc.Content {
			content.WriteString(part.Text)
		}
		contentStr := strings.TrimSpace(content.String())

		if contentStr == "" {
			log.Printf("[pgvecStore] skipping empty doc chunk %d", i)
			continue
		}

		contentStr = strings.ReplaceAll(contentStr, "\x00", " ")
		contentStr = strings.ReplaceAll(contentStr, "\u0000", " ")

		if len(contentStr) > 12000 {
			log.Printf("[pgvecStore] truncating oversized stored chunk %d from %d chars to 12000 chars", i, len(contentStr))
			contentStr = contentStr[:12000]
		}

		parentText := ""
		if doc.Metadata != nil {
			if pt, ok := doc.Metadata["parentText"].(string); ok {
				parentText = strings.TrimSpace(pt)
			}
		}

		parentText = strings.ReplaceAll(parentText, "\x00", " ")
		parentText = strings.ReplaceAll(parentText, "\u0000", " ")

		// P12-S1 note:
		// document.go sets parentText = text, so this does not introduce expanded parent context.
		// embedText may still be used for contextual embeddings if provided.
		embeddingInput := contentStr
		if doc.Metadata != nil {
			if et, ok := doc.Metadata["embedText"].(string); ok && strings.TrimSpace(et) != "" {
				embeddingInput = strings.TrimSpace(et)
			}
		}

		embeddingInput = strings.ReplaceAll(embeddingInput, "\x00", " ")
		embeddingInput = strings.ReplaceAll(embeddingInput, "\u0000", " ")

		if len(embeddingInput) > 12000 {
			log.Printf("[pgvecStore] truncating oversized embedding text %d from %d chars to 12000 chars", i, len(embeddingInput))
			embeddingInput = embeddingInput[:12000]
		}

		storedMetadata := map[string]any{}
		if doc.Metadata != nil {
			for k, v := range doc.Metadata {
				if k == "embedText" || k == "parentText" {
					continue
				}
				storedMetadata[k] = v
			}
		}

		metaJSON := []byte("{}")
		if len(storedMetadata) > 0 {
			metaJSON, _ = json.Marshal(storedMetadata)
		}

		embedText := s.formatDocumentForEmbedding(embeddingInput)
		embedDoc := ai.DocumentFromText(embedText, storedMetadata)

		prepared = append(prepared, preparedDocument{
			contentStr: contentStr,
			parentText: parentText,
			metaJSON:   metaJSON,
			embedDoc:   embedDoc,
			sourceIdx:  i,
		})
	}

	indexedCount := 0

	for batchStart := 0; batchStart < len(prepared); batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > len(prepared) {
			batchEnd = len(prepared)
		}

		batch := prepared[batchStart:batchEnd]
		inputs := make([]*ai.Document, 0, len(batch))
		for _, item := range batch {
			inputs = append(inputs, item.embedDoc)
		}

		embedResp, err := s.emb.Embed(ctx, &ai.EmbedRequest{
			Input: inputs,
		})
		if err != nil {
			return fmt.Errorf("embed batch %d-%d failed: %w", batchStart, batchEnd-1, err)
		}

		if len(embedResp.Embeddings) != len(batch) {
			return fmt.Errorf(
				"embed batch %d-%d returned %d embeddings for %d inputs",
				batchStart,
				batchEnd-1,
				len(embedResp.Embeddings),
				len(batch),
			)
		}

		for i, item := range batch {
			if len(embedResp.Embeddings[i].Embedding) == 0 {
				return fmt.Errorf("embed doc %d failed: empty embedding returned", item.sourceIdx)
			}

			vec := s.embeddingDBValue(embedResp.Embeddings[i].Embedding)

			if _, err := tx.Exec(
				ctx,
				q,
				item.contentStr,
				vec,
				item.metaJSON,
				item.parentText,
			); err != nil {
				return fmt.Errorf("insert doc %d: %w", item.sourceIdx, err)
			}

			indexedCount++
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}

	log.Printf(
		"[pgvecStore] indexed %d/%d documents using embedding batch size %d",
		indexedCount,
		len(docs),
		batchSize,
	)
	return nil
}

func (s *pgvecStore) retrieve(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {

	k := 10

	var allowedTags []string

	if opts, ok := req.Options.(map[string]any); ok {

		if v, ok := opts["k"]; ok {

			switch kv := v.(type) {

			case float64:

				k = int(kv)

			case int:

				k = kv

			}

		}

		if v, ok := opts["tags"]; ok {

			switch tv := v.(type) {

			case []string:

				allowedTags = normalizeTags(tv)

			case []any:

				var tags []string

				for _, item := range tv {

					if s, ok := item.(string); ok {

						tags = append(tags, s)

					}

				}

				allowedTags = normalizeTags(tags)

			case string:

				allowedTags = normalizeTags([]string{tv})

			}

		}

	}

	var queryBuilder strings.Builder

	for _, part := range req.Query.Content {

		queryBuilder.WriteString(part.Text)

	}

	queryText := strings.TrimSpace(queryBuilder.String())

	if queryText == "" {

		return &ai.RetrieverResponse{Documents: nil}, nil

	}

	embedQuery := ai.DocumentFromText(s.formatQueryForEmbedding(queryText), req.Query.Metadata)

	embedResp, err := s.emb.Embed(ctx, &ai.EmbedRequest{

		Input: []*ai.Document{embedQuery},
	})

	if err != nil {

		return nil, fmt.Errorf("embed query: %w", err)

	}

	if len(embedResp.Embeddings) == 0 || len(embedResp.Embeddings[0].Embedding) == 0 {

		return nil, fmt.Errorf("embed query failed: empty embedding returned")

	}

	qvec := s.embeddingDBValue(embedResp.Embeddings[0].Embedding)

	var rows pgx.Rows

	if len(allowedTags) == 0 {

		log.Printf("[pgvecStore.retrieve] no tag filter applied")

		const q = `

SELECT content, metadata, parent_text, 1 - (embedding <=> $1) AS score

FROM documents

ORDER BY embedding <=> $1

LIMIT $2

`

		rows, err = s.pool.Query(ctx, q, qvec, k)

	} else {

		log.Printf("[pgvecStore.retrieve] tag filter applied: %v", allowedTags)

		const q = `

SELECT content, metadata, parent_text, 1 - (embedding <=> $1) AS score

FROM documents

WHERE LOWER(metadata->>'tag') = ANY($3)

ORDER BY embedding <=> $1

LIMIT $2

`

		rows, err = s.pool.Query(ctx, q, qvec, k, allowedTags)

	}

	if err != nil {

		return nil, fmt.Errorf("query: %w", err)

	}

	defer rows.Close()

	var docs []*ai.Document

	for rows.Next() {

		var content, parentText string

		var metaJSON []byte

		var score float64

		if err := rows.Scan(&content, &metaJSON, &parentText, &score); err != nil {

			return nil, fmt.Errorf("scan: %w", err)

		}

		var meta map[string]any

		if err := json.Unmarshal(metaJSON, &meta); err != nil {

			meta = map[string]any{}

		}

		if parentText != "" {

			meta["parentText"] = parentText

		}

		meta["retrievalScore"] = score

		meta["retrievalType"] = "vector"

		docs = append(docs, ai.DocumentFromText(content, meta))

	}

	if err := rows.Err(); err != nil {

		return nil, fmt.Errorf("rows: %w", err)

	}

	return &ai.RetrieverResponse{Documents: docs}, nil

}

func (s *pgvecStore) Retriever() ai.Retriever {

	return s.retriever

}

func (s *pgvecStore) KeywordSearch(ctx context.Context, query string, maxResults int, allowedTags []string) ([]*ai.Document, error) {

	keywords := extractKeywords(query)

	if len(keywords) == 0 {

		return nil, nil

	}

	allowedTags = normalizeTags(allowedTags)

	tsq := strings.Join(keywords, " | ")

	log.Printf("[pgvecStore.keywordSearch] tsquery=%v tags=%v", tsq, allowedTags)

	tagFilter := ""

	args := []any{tsq, maxResults}

	if len(allowedTags) > 0 {

		tagFilter = "AND LOWER(metadata->>'tag') = ANY($3)"

		args = append(args, allowedTags)

	}

	q := fmt.Sprintf(`

WITH matches AS (

  SELECT content, metadata, parent_text,

         ts_rank(

           to_tsvector('english', content || ' ' || parent_text),

           to_tsquery('english', $1)

         ) AS rank

  FROM documents

  WHERE to_tsvector('english', content || ' ' || parent_text) @@ to_tsquery('english', $1)

  %s



  UNION ALL



  SELECT content, metadata, parent_text,

         ts_rank(

           to_tsvector('german', content || ' ' || parent_text),

           to_tsquery('german', $1)

         ) AS rank

  FROM documents

  WHERE to_tsvector('german', content || ' ' || parent_text) @@ to_tsquery('german', $1)

  %s

)

SELECT content, metadata, parent_text, MAX(rank) AS best_rank

FROM matches

GROUP BY content, metadata, parent_text

ORDER BY best_rank DESC

LIMIT $2

`, tagFilter, tagFilter)

	rows, err := s.pool.Query(ctx, q, args...)

	if err != nil {

		return nil, fmt.Errorf("keyword query: %w", err)

	}

	defer rows.Close()

	var docs []*ai.Document

	for rows.Next() {

		var content, parentText string

		var metaJSON []byte

		var rank float64

		if err := rows.Scan(&content, &metaJSON, &parentText, &rank); err != nil {

			return nil, fmt.Errorf("scan: %w", err)

		}

		var meta map[string]any

		if err := json.Unmarshal(metaJSON, &meta); err != nil {

			meta = map[string]any{}

		}

		if parentText != "" {

			meta["parentText"] = parentText

		}

		meta["retrievalScore"] = rank

		meta["retrievalType"] = "keyword"

		docs = append(docs, ai.DocumentFromText(content, meta))

	}

	if err := rows.Err(); err != nil {

		return nil, fmt.Errorf("rows: %w", err)

	}

	return docs, nil

}

func (s *pgvecStore) ListDocuments(ctx context.Context) ([]indexedDocument, error) {
	const q = `
SELECT
  COALESCE(
    NULLIF(metadata->>'filename', ''),
    NULLIF(metadata->>'url', ''),
    NULLIF(metadata->>'tag', ''),
    'Indexed document'
  ) AS name,
  CASE
    WHEN COALESCE(metadata->>'url', '') <> '' THEN 'website'
    ELSE 'pdf'
  END AS type,
  COALESCE(metadata->>'url', '')   AS url,
  COALESCE(metadata->>'tag', '')   AS tag,
  COALESCE(metadata->>'datum', '') AS datum,
  COUNT(*)                          AS chunks
FROM documents
GROUP BY
  metadata->>'filename',
  metadata->>'url',
  metadata->>'tag',
  metadata->>'datum'
ORDER BY name
`

	rows, err := s.pool.Query(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("list documents query: %w", err)
	}
	defer rows.Close()

	documents := make([]indexedDocument, 0)
	for rows.Next() {
		var doc indexedDocument
		if err := rows.Scan(
			&doc.Name,
			&doc.Type,
			&doc.URL,
			&doc.Tag,
			&doc.Datum,
			&doc.Chunks,
		); err != nil {
			return nil, fmt.Errorf("list documents scan: %w", err)
		}
		documents = append(documents, doc)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list documents rows: %w", err)
	}

	return documents, nil
}

func (s *pgvecStore) Sources(ctx context.Context) ([]source, error) {

	const q = `

SELECT

  COALESCE(metadata->>'url', '')   AS url,

  COALESCE(metadata->>'tag', '')   AS tag,

  COALESCE(metadata->>'datum', '') AS datum,

  COUNT(*)                         AS chunks

FROM documents

WHERE metadata->>'url' IS NOT NULL

   OR metadata->>'tag' IS NOT NULL

   OR metadata->>'datum' IS NOT NULL

GROUP BY metadata->>'url', metadata->>'tag', metadata->>'datum'

ORDER BY chunks DESC

`

	rows, err := s.pool.Query(ctx, q)

	if err != nil {

		return nil, fmt.Errorf("sources query: %w", err)

	}

	defer rows.Close()

	var sources []source

	for rows.Next() {

		var src source

		if err := rows.Scan(&src.URL, &src.Tag, &src.Datum, &src.Chunks); err != nil {

			return nil, fmt.Errorf("scan: %w", err)

		}

		if src.URL == "" && src.Tag == "" && src.Datum == "" {

			continue

		}

		sources = append(sources, src)

	}

	if err := rows.Err(); err != nil {

		return nil, fmt.Errorf("rows: %w", err)

	}

	return sources, nil

}

func (s *pgvecStore) Reset(ctx context.Context) error {

	_, err := s.pool.Exec(ctx, "TRUNCATE documents RESTART IDENTITY")

	if err != nil {

		return fmt.Errorf("truncate: %w", err)

	}

	return nil

}

func (s *pgvecStore) Close() error {

	s.pool.Close()

	return nil

}

// TagProfiles builds one lightweight routing profile per stored tag.

// P15-C uses this at question time to improve automatic tag routing without

// hardcoded domain rules. The profile is generated from the database itself:

// section titles, breadcrumbs, and small chunk snippets.

func (s *pgvecStore) TagProfiles(ctx context.Context, limitPerTag int) (map[string]string, error) {

	if limitPerTag <= 0 {

		limitPerTag = 80

	}

	const q = `

WITH ranked AS (

  SELECT

    LOWER(COALESCE(metadata->>'tag', '')) AS tag,

    COALESCE(metadata->>'section', '')    AS section,

    COALESCE(metadata->>'breadcrumb', '') AS breadcrumb,

    LEFT(content, 700)                    AS snippet,

    ROW_NUMBER() OVER (

      PARTITION BY LOWER(COALESCE(metadata->>'tag', ''))

      ORDER BY id ASC

    ) AS rn

  FROM documents

  WHERE COALESCE(metadata->>'tag', '') <> ''

)

SELECT tag, section, breadcrumb, snippet

FROM ranked

WHERE rn <= $1

ORDER BY tag, rn

`

	rows, err := s.pool.Query(ctx, q, limitPerTag)

	if err != nil {

		return nil, fmt.Errorf("tag profiles query: %w", err)

	}

	defer rows.Close()

	builders := map[string]*strings.Builder{}

	seen := map[string]map[string]bool{}

	for rows.Next() {

		var tag, section, breadcrumb, snippet string

		if err := rows.Scan(&tag, &section, &breadcrumb, &snippet); err != nil {

			return nil, fmt.Errorf("tag profiles scan: %w", err)

		}

		tag = strings.ToLower(strings.TrimSpace(tag))

		if tag == "" {

			continue

		}

		if builders[tag] == nil {

			builders[tag] = &strings.Builder{}

			seen[tag] = map[string]bool{}

		}

		parts := []string{tag, section, breadcrumb, snippet}

		for _, part := range parts {

			part = strings.Join(strings.Fields(strings.TrimSpace(part)), " ")

			if part == "" {

				continue

			}

			key := strings.ToLower(part)

			if seen[tag][key] {

				continue

			}

			seen[tag][key] = true

			builders[tag].WriteString(" ")

			builders[tag].WriteString(part)

		}

	}

	if err := rows.Err(); err != nil {

		return nil, fmt.Errorf("tag profiles rows: %w", err)

	}

	profiles := make(map[string]string, len(builders))

	for tag, b := range builders {

		profiles[tag] = b.String()

	}

	return profiles, nil

}
