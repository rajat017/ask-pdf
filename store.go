package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/localvec"
)

type source struct {
	URL    string `json:"url,omitempty"`
	Tag    string `json:"tag,omitempty"`
	Datum  string `json:"datum,omitempty"`
	Chunks int    `json:"chunks"`
}

type VectorStore interface {
	Index(ctx context.Context, docs []*ai.Document) error
	Retriever() ai.Retriever
	KeywordSearch(ctx context.Context, query string, maxResults int, allowedTags []string) ([]*ai.Document, error)
	Sources(ctx context.Context) ([]source, error)
	Reset(ctx context.Context) error
	Close() error
}

type localvecStore struct {
	indexer   *localvec.DocStore
	retriever ai.Retriever
	dbPath    string
}

func newLocalvecStore(g *genkit.Genkit, emb ai.Embedder, dir string) (*localvecStore, error) {
	if err := localvec.Init(); err != nil {
		return nil, fmt.Errorf("localvec.Init: %w", err)
	}

	indexer, retriever, err := localvec.DefineRetriever(
		g, "docQA", localvec.Config{
			Embedder: emb,
			Dir:      dir,
		},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("localvec.DefineRetriever: %w", err)
	}

	return &localvecStore{
		indexer:   indexer,
		retriever: retriever,
		dbPath:    filepath.Join(dir, "__db_docQA.json"),
	}, nil
}

func (s *localvecStore) Index(ctx context.Context, docs []*ai.Document) error {
	return localvec.Index(ctx, docs, s.indexer)
}

func (s *localvecStore) Retriever() ai.Retriever {
	return s.retriever
}

func (s *localvecStore) KeywordSearch(ctx context.Context, query string, maxResults int, allowedTags []string) ([]*ai.Document, error) {
	docs, err := keywordSearch(s.dbPath, query, maxResults)
	if err != nil {
		return nil, err
	}
	return FilterByTag(docs, allowedTags), nil
}

func (s *localvecStore) Sources(ctx context.Context) ([]source, error) {
	return localvecSources(s.dbPath)
}

func (s *localvecStore) Reset(ctx context.Context) error {
	if err := os.Remove(s.dbPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove %s: %w", s.dbPath, err)
	}
	return nil
}

func (s *localvecStore) Close() error {
	return nil
}

func localvecSources(dbPath string) ([]source, error) {
	raw, err := os.ReadFile(dbPath)
	if err != nil {
		return []source{}, nil
	}

	var db map[string]localvecEntry
	if err := json.Unmarshal(raw, &db); err != nil {
		return nil, fmt.Errorf("cannot parse vector store: %w", err)
	}

	type sourceKey struct {
		URL   string
		Tag   string
		Datum string
	}

	agg := make(map[sourceKey]*source)

	for _, entry := range db {
		m := entry.Doc.Metadata
		if m == nil {
			continue
		}

		sk := sourceKey{
			URL:   strVal(m, "url"),
			Tag:   strVal(m, "tag"),
			Datum: strVal(m, "datum"),
		}

		if sk.URL == "" && sk.Tag == "" && sk.Datum == "" {
			continue
		}

		s, ok := agg[sk]
		if !ok {
			s = &source{
				URL:   sk.URL,
				Tag:   sk.Tag,
				Datum: sk.Datum,
			}
			agg[sk] = s
		}

		s.Chunks++
	}

	sources := make([]source, 0, len(agg))
	for _, s := range agg {
		sources = append(sources, *s)
	}

	return sources, nil
}

func newVectorStore(g *genkit.Genkit, emb ai.Embedder, embeddingDim int) (VectorStore, error) {
	backend := os.Getenv("STORE_BACKEND")
	if backend == "" {
		backend = "localvec"
	}

	switch backend {
	case "localvec":
		dir := os.Getenv("VECTOR_STORE_DIR")
		if dir == "" {
			dir = "./data"
		}
		return newLocalvecStore(g, emb, dir)

	case "pgvector":
		dsn := os.Getenv("DATABASE_URL")
		if dsn == "" {
			return nil, fmt.Errorf("STORE_BACKEND=pgvector requires DATABASE_URL to be set")
		}
		return newPgvecStore(context.Background(), g, emb, dsn, embeddingDim)

	default:
		return nil, fmt.Errorf("unknown STORE_BACKEND %q (expected \"localvec\" or \"pgvector\")", backend)
	}
}
