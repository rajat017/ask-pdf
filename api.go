package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Shared input/output types for flows and handlers.

type indexInput struct {
	Path     string `json:"path"`
	Filename string `json:"filename,omitempty"`
	Tag      string `json:"tag,omitempty"`
	Datum    string `json:"datum,omitempty"` // e.g. "2025-01-15"
}

type indexURLInput struct {
	URL string `json:"url"`

	Tag string `json:"tag,omitempty"`

	Datum string `json:"datum,omitempty"`
}

type crawlInput struct {
	SeedURLs []string `json:"seedUrls"`

	Tag string `json:"tag,omitempty"`

	Datum string `json:"datum,omitempty"`

	MaxPages int `json:"maxPages,omitempty"`

	MaxDepth int `json:"maxDepth,omitempty"`

	DelayMs int `json:"delayMs,omitempty"`
}

type qaSource struct {
	Pages string `json:"pages,omitempty"`

	Section string `json:"section,omitempty"`

	Breadcrumb string `json:"breadcrumb,omitempty"`

	Tag string `json:"tag,omitempty"`

	Datum string `json:"datum,omitempty"`

	URL string `json:"url,omitempty"`
}

type qaResponse struct {
	Answer string `json:"answer"`

	Sources []qaSource `json:"sources"`

	TotalMs float64 `json:"total_ms,omitempty"`

	HydeMs float64 `json:"hyde_ms,omitempty"`

	RetrievalMs float64 `json:"retrieval_ms,omitempty"`

	RerankMs float64 `json:"rerank_ms,omitempty"`

	AnswerGenerationMs float64 `json:"answer_generation_ms,omitempty"`

	OtherOverheadMs float64 `json:"other_overhead_ms,omitempty"`
}

func strVal(m map[string]any, key string) string {

	if v, ok := m[key].(string); ok {

		return v

	}

	return ""

}

// routeConfig holds all dependencies needed by the HTTP handlers.

type routeConfig struct {
	ctx context.Context

	store VectorStore

	qaHandler http.HandlerFunc

	indexPDFHandler http.HandlerFunc

	indexWebHandler http.HandlerFunc

	crawlWebHandler http.HandlerFunc

	runIndexPDF func(context.Context, indexInput) (any, error)
}

type indexedDocument struct {
	Name   string `json:"name"`
	Type   string `json:"type"`
	URL    string `json:"url,omitempty"`
	Tag    string `json:"tag,omitempty"`
	Datum  string `json:"datum,omitempty"`
	Chunks int    `json:"chunks"`
}

type documentListProvider interface {
	ListDocuments(context.Context) ([]indexedDocument, error)
}

// makeAskPDFCompatHandler lets /askPDF accept both request shapes:

//

//  1. Frontend-friendly shape:

//     {"question":"...","tag":"amcn","history":[...]}

//

//  2. Genkit HTTP handler shape:

//     {"data":{"question":"...","tag":"amcn","history":[...]}}

//

// Genkit's handler expects the second shape. This wrapper converts the first

// shape into the second shape before forwarding the request to the Genkit flow.

//

// P20-A note:

// This handler does not need special conversation logic. It preserves every

// incoming field, including "history", and forwards it unchanged inside "data".

// main.go is responsible for parsing and using the history.

func makeAskPDFCompatHandler(next http.HandlerFunc) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		if r.Body == nil {

			next(w, r)

			return

		}

		bodyBytes, err := io.ReadAll(r.Body)

		if err != nil {

			w.Header().Set("Content-Type", "application/json")

			w.WriteHeader(http.StatusBadRequest)

			_ = json.NewEncoder(w).Encode(map[string]string{"error": "cannot read request body"})

			return

		}

		_ = r.Body.Close()

		trimmed := strings.TrimSpace(string(bodyBytes))

		if trimmed == "" {

			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

			r.ContentLength = int64(len(bodyBytes))

			next(w, r)

			return

		}

		var raw map[string]any

		if err := json.Unmarshal(bodyBytes, &raw); err != nil {

			// Let Genkit return its normal validation error for malformed/non-JSON bodies.

			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

			r.ContentLength = int64(len(bodyBytes))

			next(w, r)

			return

		}

		// Already in Genkit format, so forward unchanged.

		if _, ok := raw["data"]; ok {

			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

			r.ContentLength = int64(len(bodyBytes))

			next(w, r)

			return

		}

		// Compatibility mode: wrap plain flow input into Genkit's HTTP envelope.

		// All fields are preserved, including P20-A's "history".

		wrapped, err := json.Marshal(map[string]any{"data": raw})

		if err != nil {

			w.Header().Set("Content-Type", "application/json")

			w.WriteHeader(http.StatusInternalServerError)

			_ = json.NewEncoder(w).Encode(map[string]string{"error": "cannot wrap request body"})

			return

		}

		r.Body = io.NopCloser(bytes.NewReader(wrapped))

		r.ContentLength = int64(len(wrapped))

		r.Header.Set("Content-Type", "application/json")

		next(w, r)

	}

}

func setupRoutes(mux *http.ServeMux, cfg routeConfig) {

	mux.HandleFunc("POST /askPDF", makeAskPDFCompatHandler(cfg.qaHandler))

	mux.HandleFunc("POST /addPDFDocument", cfg.indexPDFHandler)

	mux.HandleFunc("POST /addWebDocument", cfg.indexWebHandler)

	mux.HandleFunc("POST /crawlWeb", cfg.crawlWebHandler)

	mux.HandleFunc("POST /api/upload", makeUploadHandler(cfg.ctx, cfg.runIndexPDF))

	mux.HandleFunc("GET /sources", makeSourcesHandler(cfg.store))
	mux.HandleFunc("GET /api/documents", makeDocumentsHandler(cfg.store))

	mux.HandleFunc("GET /api/keyword-search", makeKeywordSearchHandler(cfg.store))

	mux.HandleFunc("DELETE /api/reset", makeResetHandler(cfg.store))

	mux.Handle("/", http.FileServer(http.Dir("./static")))

}

func makeUploadHandler(

	ctx context.Context,

	runIndex func(context.Context, indexInput) (any, error),

) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		const maxUpload = 50 << 20 // 50 MB

		r.Body = http.MaxBytesReader(w, r.Body, maxUpload)

		if err := r.ParseMultipartForm(maxUpload); err != nil {

			http.Error(w, `{"error":"Datei zu groß (max 50 MB)"}`, http.StatusBadRequest)

			return

		}

		file, header, err := r.FormFile("file")

		if err != nil {

			http.Error(w, `{"error":"Kein PDF-Feld gefunden"}`, http.StatusBadRequest)

			return

		}

		defer file.Close()

		if err := os.MkdirAll("./uploads", 0o755); err != nil {

			http.Error(w, `{"error":"Kann Upload-Verzeichnis nicht erstellen"}`, http.StatusInternalServerError)

			return

		}

		filename := fmt.Sprintf("%d_%s", time.Now().UnixMilli(), filepath.Base(header.Filename))

		destPath := filepath.Join("./uploads", filename)

		dest, err := os.Create(destPath)

		if err != nil {

			http.Error(w, `{"error":"Kann Datei nicht speichern"}`, http.StatusInternalServerError)

			return

		}

		defer dest.Close()

		if _, err := io.Copy(dest, file); err != nil {

			http.Error(w, `{"error":"Fehler beim Schreiben der Datei"}`, http.StatusInternalServerError)

			return

		}

		tag := r.FormValue("tag")

		datum := r.FormValue("datum")

		result, err := runIndex(ctx, indexInput{
			Path:     destPath,
			Filename: filepath.Base(header.Filename),
			Tag:      tag,
			Datum:    datum,
		})

		if err != nil {

			w.Header().Set("Content-Type", "application/json")

			w.WriteHeader(http.StatusInternalServerError)

			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})

			return

		}

		w.Header().Set("Content-Type", "application/json")

		_ = json.NewEncoder(w).Encode(result)

	}

}

func makeResetHandler(store VectorStore) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		if err := store.Reset(r.Context()); err != nil {

			w.Header().Set("Content-Type", "application/json")

			w.WriteHeader(http.StatusInternalServerError)

			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})

			return

		}

		w.Header().Set("Content-Type", "application/json")

		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	}

}

// makeKeywordSearchHandler handles GET /api/keyword-search

//

// Query parameters:

//

//  q       string  — search query (required)

//  k       int     — max results (default 20)

//  tags    string  — comma-separated tag allowlist, e.g. "ads" or "ads,acg"

//                    If absent or empty, searches all datasets.

//

// Examples:

//

//  /api/keyword-search?q=campaign&tags=ads

//  /api/keyword-search?q=anime&tags=acg,amcn&k=10

//  /api/keyword-search?q=report

func makeKeywordSearchHandler(store VectorStore) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		q := r.URL.Query().Get("q")

		if q == "" {

			http.Error(w, `{"error":"query parameter 'q' is required"}`, http.StatusBadRequest)

			return

		}

		maxResults := 20

		if v := r.URL.Query().Get("k"); v != "" {

			if n, err := strconv.Atoi(v); err == nil && n > 0 {

				maxResults = n

			}

		}

		var allowedTags []string

		if raw := strings.TrimSpace(r.URL.Query().Get("tags")); raw != "" {

			for _, t := range strings.Split(raw, ",") {

				t = strings.TrimSpace(t)

				if t != "" {

					allowedTags = append(allowedTags, t)

				}

			}

		}

		docs, err := store.KeywordSearch(r.Context(), q, maxResults, allowedTags)

		if err != nil {

			w.Header().Set("Content-Type", "application/json")

			w.WriteHeader(http.StatusInternalServerError)

			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})

			return

		}

		type result struct {
			Text string `json:"text"`

			Metadata map[string]any `json:"metadata,omitempty"`
		}

		results := make([]result, 0, len(docs))

		for _, doc := range docs {

			var sb strings.Builder

			for _, part := range doc.Content {

				sb.WriteString(part.Text)

			}

			results = append(results, result{

				Text: sb.String(),

				Metadata: doc.Metadata,
			})

		}

		w.Header().Set("Content-Type", "application/json")

		_ = json.NewEncoder(w).Encode(map[string]any{

			"query": q,

			"count": len(results),

			"results": results,

			"allowedTags": allowedTags,
		})

	}

}

func makeDocumentsHandler(store VectorStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		provider, ok := store.(documentListProvider)
		if !ok {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotImplemented)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error": "document listing is not supported by this store",
			})
			return
		}

		documents, err := provider.ListDocuments(r.Context())
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}

		if documents == nil {
			documents = []indexedDocument{}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"documents": documents,
			"count":     len(documents),
		})
	}
}

func makeSourcesHandler(store VectorStore) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		sources, err := store.Sources(r.Context())

		if err != nil {

			http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusInternalServerError)

			return

		}

		w.Header().Set("Content-Type", "application/json")

		_ = json.NewEncoder(w).Encode(map[string]any{"sources": sources})

	}

}
