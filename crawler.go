package main

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/firebase/genkit/go/ai"
	"golang.org/x/net/html"
)

// crawlPage represents a URL to visit with its current depth in the BFS.
type crawlPage struct {
	URL   string
	Depth int
}

// crawlResult summarises one crawl run.
type crawlResult struct {
	PagesVisited int      `json:"pagesVisited"`
	PagesSaved   int      `json:"pagesSaved"`
	PagesIndexed int      `json:"pagesIndexed"`
	Errors       []string `json:"errors,omitempty"`
}

// Safety limits for web chunks before embedding.
// These protect Ollama from oversized or malformed extracted web text.
const (
	minEmbedChars       = 20
	maxEmbedChars       = 6000
	maxParentTextChars  = 9000
	failedPreviewLength = 400
)

// firstNRunes safely returns the first n runes without breaking UTF-8.
func firstNRunes(s string, n int) string {
	if n <= 0 || s == "" {
		return ""
	}

	if utf8.RuneCountInString(s) <= n {
		return s
	}

	var b strings.Builder
	b.Grow(n)
	count := 0

	for _, r := range s {
		if count >= n {
			break
		}
		b.WriteRune(r)
		count++
	}

	return b.String()
}

// truncateRunes safely truncates a string to at most n runes.
func truncateRunes(s string, n int) string {
	if n <= 0 || s == "" {
		return ""
	}
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	return firstNRunes(s, n)
}

// cleanForEmbedding removes problematic control characters and normalizes whitespace.
// This helps prevent Ollama embedding 400 errors caused by malformed extracted text.
func cleanForEmbedding(s string) string {
	s = strings.TrimSpace(s)

	if s == "" {
		return ""
	}

	s = strings.Map(func(r rune) rune {
		// Remove null bytes completely.
		if r == '\u0000' {
			return -1
		}

		// Replace most control characters with spaces.
		// Keep common whitespace characters.
		if r < 32 && r != '\n' && r != '\t' && r != '\r' {
			return ' '
		}

		return r
	}, s)

	// Normalize all whitespace to single spaces.
	s = strings.Join(strings.Fields(s), " ")

	return strings.TrimSpace(s)
}

// cleanMetadataText keeps metadata useful but prevents very large parentText values.
func cleanMetadataText(s string, maxChars int) string {
	s = cleanForEmbedding(s)
	if s == "" {
		return ""
	}
	return truncateRunes(s, maxChars)
}

// buildWebContextualEmbeddingText creates metadata-enriched text for embedding only.
// The original chunk text is still stored in metadata/parentText for answer generation.
func buildWebContextualEmbeddingText(pageURL string, c chunk, bodyText string) string {
	var b strings.Builder

	b.WriteString("Web page context for retrieval:\n")

	b.WriteString("Source URL: ")
	b.WriteString(pageURL)
	b.WriteString("\n")

	if c.breadcrumb != "" {
		b.WriteString("Section path: ")
		b.WriteString(c.breadcrumb)
		b.WriteString("\n")
	}

	if c.section != "" {
		b.WriteString("Current section: ")
		b.WriteString(c.section)
		b.WriteString("\n")
	}

	b.WriteString("\nOriginal web chunk:\n")
	b.WriteString(strings.TrimSpace(bodyText))

	return strings.TrimSpace(b.String())
}

// dedupWebChunks removes generic near-duplicate chunks before embedding.
// This helps reduce repeated context caused by HTML boilerplate and chunk overlap.
func dedupWebChunks(chunks []chunk) []chunk {
	if len(chunks) == 0 {
		return chunks
	}

	seen := map[string]bool{}
	out := make([]chunk, 0, len(chunks))

	for _, c := range chunks {
		key := webChunkDedupKey(c.text)
		if key == "" {
			continue
		}

		if seen[key] {
			continue
		}

		seen[key] = true
		out = append(out, c)
	}

	return out
}

// webChunkDedupKey creates a generic normalized key for near-duplicate detection.
// It avoids website-specific logic.
func webChunkDedupKey(s string) string {
	s = cleanForEmbedding(s)
	s = strings.ToLower(s)

	var b strings.Builder
	prevSpace := false

	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			prevSpace = false
		} else if unicode.IsSpace(r) {
			if !prevSpace {
				b.WriteRune(' ')
				prevSpace = true
			}
		}
	}

	key := strings.TrimSpace(b.String())
	if key == "" {
		return ""
	}

	// Use the beginning of the normalized chunk for near-dedup.
	// This removes repeated overlap/chrome while keeping genuinely different sections.
	if utf8.RuneCountInString(key) > 700 {
		key = firstNRunes(key, 700)
	}

	return key
}

// extractLinks parses raw HTML and returns all same-scheme links resolved
// against baseURL. Fragment-only, javascript:, mailto:, and non-HTML
// resource links are filtered out.
func extractLinks(htmlContent string, baseURL string) ([]string, error) {
	base, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}

	doc, err := html.Parse(strings.NewReader(htmlContent))
	if err != nil {
		return nil, fmt.Errorf("parsing HTML: %w", err)
	}

	// Extensions that are clearly not HTML pages.
	skipExts := map[string]bool{
		".pdf": true, ".jpg": true, ".jpeg": true, ".png": true,
		".gif": true, ".svg": true, ".webp": true, ".ico": true,
		".css": true, ".js": true, ".json": true, ".xml": true,
		".zip": true, ".gz": true, ".tar": true, ".mp3": true,
		".mp4": true, ".avi": true, ".mov": true, ".woff": true,
		".woff2": true, ".ttf": true, ".eot": true,
	}

	seen := make(map[string]bool)
	var links []string

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" {
			href := getAttr(n, "href")
			href = strings.TrimSpace(href)

			if href == "" ||
				href == "#" ||
				strings.HasPrefix(strings.ToLower(href), "javascript:") ||
				strings.HasPrefix(strings.ToLower(href), "mailto:") ||
				strings.HasPrefix(strings.ToLower(href), "tel:") {
				goto children
			}

			resolved, err := base.Parse(href)
			if err != nil {
				goto children
			}

			// Only http/https.
			if resolved.Scheme != "http" && resolved.Scheme != "https" {
				goto children
			}

			// Strip fragment and query for dedup/cleanliness.
			resolved.Fragment = ""
			resolved.RawQuery = ""

			// Skip non-HTML extensions.
			ext := strings.ToLower(filepath.Ext(resolved.Path))
			if skipExts[ext] {
				goto children
			}

			canonical := resolved.String()

			if !seen[canonical] {
				seen[canonical] = true
				links = append(links, canonical)
			}
		}

	children:
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	walk(doc)

	return links, nil
}

// urlPrefix describes the host + path prefix derived from a seed URL.
type urlPrefix struct {
	Host string
	Path string // always ends with "/"
}

// buildAllowedPrefixes extracts host+path prefixes from seed URLs.
// The prefix limits crawling so the crawler does not escape too far from
// the seed area.
func buildAllowedPrefixes(seedURLs []string) []urlPrefix {
	var prefixes []urlPrefix

	for _, seed := range seedURLs {
		u, err := url.Parse(seed)
		if err != nil {
			continue
		}

		p := u.Path
		if p == "" {
			p = "/"
		}

		// Ensure trailing slash so "/docs" won't accidentally match "/documents".
		if !strings.HasSuffix(p, "/") {
			p += "/"
		}

		prefixes = append(prefixes, urlPrefix{
			Host: strings.ToLower(u.Hostname()),
			Path: p,
		})
	}

	return prefixes
}

// allowedByPrefix returns true when rawURL's host and path fall within at
// least one of the allowed prefixes.
func allowedByPrefix(rawURL string, prefixes []urlPrefix) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}

	host := strings.ToLower(u.Hostname())
	path := u.Path

	if path == "" {
		path = "/"
	}

	if !strings.HasSuffix(path, "/") {
		path += "/"
	}

	for _, pfx := range prefixes {
		if host == pfx.Host && strings.HasPrefix(path, pfx.Path) {
			return true
		}
	}

	return false
}

// urlToFilename converts a URL into a safe filename like
// "example.com_docs_getting-started.md". Query and fragment are stripped,
// the result is capped at 200 characters.
func urlToFilename(rawURL string) string {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "unknown.md"
	}

	host := u.Hostname()
	path := strings.Trim(u.Path, "/")

	var name string
	if path == "" {
		name = host
	} else {
		name = host + "_" + strings.ReplaceAll(path, "/", "_")
	}

	// Replace any remaining unsafe characters.
	name = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '-' ||
			r == '_' ||
			r == '.' {
			return r
		}
		return '_'
	}, name)

	if len(name) > 200 {
		name = name[:200]
	}

	return name + ".md"
}

// savePageAsMarkdown writes a Markdown file with YAML frontmatter for one
// crawled page.
func savePageAsMarkdown(outputDir, filename, pageURL string, paras []paragraph) error {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("creating output dir: %w", err)
	}

	var sb strings.Builder

	sb.WriteString("---\n")
	sb.WriteString(fmt.Sprintf("url: %s\n", pageURL))
	sb.WriteString(fmt.Sprintf("crawled: %s\n", time.Now().Format(time.RFC3339)))
	sb.WriteString("---\n\n")

	for _, p := range paras {
		text := strings.TrimSpace(p.text)
		if text == "" {
			continue
		}

		if p.heading {
			sb.WriteString("## ")
		}

		sb.WriteString(text)
		sb.WriteString("\n\n")
	}

	dest := filepath.Join(outputDir, filename)
	return os.WriteFile(dest, []byte(sb.String()), 0o644)
}

// indexFn is the callback signature used by runCrawl to index one page's
// chunks into the vector store.
type indexFn func(ctx context.Context, pageURL string, paras []paragraph) (int, error)

// runCrawl performs a breadth-first crawl starting from cfg.SeedURLs.
// For each page it fetches, extracts structured text, saves a Markdown
// file, and calls indexPage to embed+index the chunks.
// Errors on individual pages are collected in crawlResult.Errors; only
// fatal errors cause an early return.
func runCrawl(ctx context.Context, cfg crawlInput, outputDir string, indexPage indexFn) (crawlResult, error) {
	if len(cfg.SeedURLs) == 0 {
		return crawlResult{}, fmt.Errorf("no seed URLs provided")
	}

	maxPages := cfg.MaxPages
	if maxPages <= 0 {
		maxPages = 50
	}

	maxDepth := cfg.MaxDepth
	if maxDepth <= 0 {
		maxDepth = 3
	}

	delayMs := cfg.DelayMs
	if delayMs <= 0 {
		delayMs = 500
	}

	allowedPrefixes := buildAllowedPrefixes(cfg.SeedURLs)

	visited := make(map[string]bool)
	queue := make([]crawlPage, 0, len(cfg.SeedURLs))

	for _, seed := range cfg.SeedURLs {
		seed = strings.TrimSpace(seed)
		if seed == "" {
			continue
		}

		u, err := url.Parse(seed)
		if err == nil {
			u.Fragment = ""
			u.RawQuery = ""
			seed = u.String()
		}

		queue = append(queue, crawlPage{URL: seed, Depth: 0})
	}

	var result crawlResult

	for len(queue) > 0 && result.PagesVisited < maxPages {
		select {
		case <-ctx.Done():
			result.Errors = append(result.Errors, "crawl cancelled: "+ctx.Err().Error())
			return result, nil
		default:
		}

		page := queue[0]
		queue = queue[1:]

		if visited[page.URL] {
			continue
		}

		visited[page.URL] = true
		result.PagesVisited++

		log.Printf("[crawl] visiting (%d/%d, depth=%d): %s",
			result.PagesVisited,
			maxPages,
			page.Depth,
			page.URL,
		)

		// Fetch page.
		htmlContent, err := fetchWebPage(page.URL)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("fetch %s: %v", page.URL, err))
			continue
		}

		// Parse into structured paragraphs.
		paras, err := htmlToStructuredText(strings.NewReader(htmlContent))
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("parse %s: %v", page.URL, err))
			continue
		}

		if len(paras) == 0 {
			result.Errors = append(result.Errors, fmt.Sprintf("parse %s: no text paragraphs extracted", page.URL))
		}

		// Save as Markdown.
		filename := urlToFilename(page.URL)
		if err := savePageAsMarkdown(outputDir, filename, page.URL, paras); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("save %s: %v", page.URL, err))
			// Continue anyway — indexing may still work.
		} else {
			result.PagesSaved++
		}

		// Index into vector store.
		indexed, err := indexPage(ctx, page.URL, paras)
		if indexed > 0 {
			result.PagesIndexed += indexed
		}
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("index %s: %v", page.URL, err))
		}

		// Extract links and enqueue if within depth.
		if page.Depth < maxDepth {
			links, err := extractLinks(htmlContent, page.URL)
			if err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("links %s: %v", page.URL, err))
			} else {
				for _, link := range links {
					if !visited[link] && allowedByPrefix(link, allowedPrefixes) {
						queue = append(queue, crawlPage{
							URL:   link,
							Depth: page.Depth + 1,
						})
					}
				}
			}
		}

		// Politeness delay.
		if result.PagesVisited < maxPages && len(queue) > 0 {
			time.Sleep(time.Duration(delayMs) * time.Millisecond)
		}
	}

	return result, nil
}

// buildAndIndexPage creates an indexFn closure that chunks paragraphs and
// indexes them into the vector store with the given metadata.
//
// Important:
// Chunks are indexed one-by-one. This is slightly slower than batch indexing,
// but much more robust. If one web chunk causes an Ollama 400 error, the
// remaining chunks can still be indexed.
//
// W1 web contextual enrichment:
// - embeddingText is metadata-enriched text used for retrieval/embedding.
// - original text and parentText are preserved in metadata for answer generation.
// - This avoids replacing evidence text with artificial enrichment text.
func buildAndIndexPage(chunkSize, chunkOverlap int, store VectorStore, tag, datum string) indexFn {
	return func(ctx context.Context, pageURL string, paras []paragraph) (int, error) {
		chunks := buildChunks(paras, chunkSize, chunkOverlap)
		if len(chunks) == 0 {
			return 0, nil
		}

		beforeDedup := len(chunks)
		chunks = dedupWebChunks(chunks)
		removed := beforeDedup - len(chunks)

		if removed > 0 {
			log.Printf("[index-web] dedup removed %d/%d near-duplicate chunks from %s",
				removed,
				beforeDedup,
				pageURL,
			)
		}

		indexed := 0
		failed := 0
		skipped := 0

		for i, c := range chunks {
			// Original evidence text.
			originalText := cleanForEmbedding(c.text)

			if len(originalText) < minEmbedChars {
				skipped++
				log.Printf("[index-web] skip chunk %d from %s: too short after cleaning, chars=%d",
					i,
					pageURL,
					len(originalText),
				)
				continue
			}

			if utf8.RuneCountInString(originalText) > maxEmbedChars {
				originalLen := utf8.RuneCountInString(originalText)
				originalText = truncateRunes(originalText, maxEmbedChars)

				log.Printf("[index-web] trimmed original chunk %d from %s: runes=%d -> %d",
					i,
					pageURL,
					originalLen,
					maxEmbedChars,
				)
			}

			// Parent text for answer generation.
			parentText := cleanMetadataText(c.parentText, maxParentTextChars)
			if parentText == "" {
				parentText = originalText
			}

			// W1: contextual enriched text for embedding/retrieval only.
			embeddingText := buildWebContextualEmbeddingText(pageURL, c, originalText)
			embeddingText = cleanForEmbedding(embeddingText)

			if utf8.RuneCountInString(embeddingText) > maxEmbedChars {
				originalLen := utf8.RuneCountInString(embeddingText)
				embeddingText = truncateRunes(embeddingText, maxEmbedChars)

				log.Printf("[index-web] trimmed enriched embedding text %d from %s: runes=%d -> %d",
					i,
					pageURL,
					originalLen,
					maxEmbedChars,
				)
			}

			meta := map[string]any{
				"parentText":   parentText,
				"originalText": originalText,
				"url":          pageURL,
			}

			if c.breadcrumb != "" {
				meta["breadcrumb"] = cleanMetadataText(c.breadcrumb, 1000)
			}

			if c.section != "" {
				meta["section"] = cleanMetadataText(c.section, 1000)
			}

			if tag != "" {
				meta["tag"] = tag
			}

			if datum != "" {
				meta["datum"] = datum
			}

			// Important:
			// The document text is the enriched embedding text.
			// Retrieval should use this, but answer generation should prefer parentText/originalText metadata.
			doc := ai.DocumentFromText(embeddingText, meta)

			log.Printf("[index-web] embedding chunk %d/%d from %s | embedChars=%d | originalChars=%d | preview=%q",
				i+1,
				len(chunks),
				pageURL,
				len(embeddingText),
				len(originalText),
				firstNRunes(embeddingText, 160),
			)

			if err := store.Index(ctx, []*ai.Document{doc}); err != nil {
				failed++

				log.Printf("[index-web] failed chunk %d from %s: %v | embedChars=%d | preview=%q",
					i,
					pageURL,
					err,
					len(embeddingText),
					firstNRunes(embeddingText, failedPreviewLength),
				)

				// Continue indexing other chunks instead of failing the whole page.
				continue
			}

			indexed++
		}

		log.Printf("[index-web] finished %s | chunks=%d | indexed=%d | skipped=%d | failed=%d",
			pageURL,
			len(chunks),
			indexed,
			skipped,
			failed,
		)

		if failed > 0 {
			return indexed, fmt.Errorf("%d chunk(s) failed during indexing; %d indexed successfully", failed, indexed)
		}

		return indexed, nil
	}
}
