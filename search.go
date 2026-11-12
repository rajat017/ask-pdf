package main

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
	"unicode"

	"github.com/firebase/genkit/go/ai"
)

// localvecEntry mirrors the JSON structure of a single document in localvec's DB file.
type localvecEntry struct {
	Doc struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Metadata map[string]any `json:"metadata"`
	} `json:"doc"`
}

// deStopWords contains common German and English stop words that carry
// little semantic weight for keyword matching.
var deStopWords = map[string]bool{
	// German
	"der": true, "die": true, "das": true, "den": true, "dem": true, "des": true,
	"ein": true, "eine": true, "einem": true, "einen": true, "einer": true,
	"und": true, "oder": true, "aber": true, "nicht": true, "auch": true,
	"ist": true, "sind": true, "war": true, "hat": true, "haben": true,
	"wird": true, "werden": true, "kann": true, "können": true,
	"mit": true, "von": true, "für": true, "auf": true, "aus": true,
	"bei": true, "nach": true, "über": true, "unter": true, "vor": true,
	"durch": true, "als": true, "wie": true, "wer": true,
	"warum": true, "welche": true, "welcher": true,
	"welches": true, "welchem": true, "welchen": true,
	"ich": true, "du": true, "er": true, "sie": true, "es": true,
	"wir": true, "ihr": true, "man": true, "sich": true,
	"noch": true, "schon": true, "nur": true, "sehr": true,
	"gibt": true, "dort": true, "hier": true,
	"dann": true, "also": true, "denn": true, "weil": true,
	"wenn": true, "dass": true, "ob": true,
	"zum": true, "zur": true, "beim": true, "vom": true,
	"bis": true, "seit": true, "kein": true, "keine": true,
	"alle": true, "alles": true, "jeder": true, "jede": true, "jedes": true,
	// English + overlapping
	"the": true, "was": true, "were": true, "are": true,
	"been": true, "have": true, "has": true, "had": true,
	"does": true, "did": true, "will": true, "would": true,
	"could": true, "should": true, "may": true, "might": true,
	"can": true, "need": true, "must": true,
	"for": true, "with": true, "from": true, "about": true,
	"into": true, "through": true,
	"and": true, "but": true, "not": true,
	"this": true, "that": true, "these": true, "those": true,
	"its": true, "they": true, "them": true, "their": true,
	"what": true, "which": true, "who": true, "when": true, "where": true, "how": true,
	// Shared (appear in both languages)
	"an": true, "in": true, "zu": true, "im": true, "am": true,
	"is": true, "be": true, "do": true, "so": true,
	"to": true, "on": true, "at": true, "by": true, "as": true, "or": true, "of": true,
	"it": true, "he": true, "we": true,
}

// extractKeywords splits text into unique lowercase keywords,
// filtering out stop words and tokens shorter than 3 characters.
func extractKeywords(text string) []string {
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	seen := make(map[string]bool)
	var keywords []string
	for _, w := range words {
		if len(w) < 3 || deStopWords[w] || seen[w] {
			continue
		}
		seen[w] = true
		keywords = append(keywords, w)
	}
	return keywords
}

// keywordSearch loads the localvec DB file and returns documents whose
// content (chunk text + parentText) contains keywords from the query.
// Results are sorted by match count (descending).
func keywordSearch(dbPath string, query string, maxResults int) ([]*ai.Document, error) {
	keywords := extractKeywords(query)
	if len(keywords) == 0 {
		return nil, nil
	}

	data, err := os.ReadFile(dbPath)
	if err != nil {
		return nil, err
	}

	var db map[string]localvecEntry
	if err := json.Unmarshal(data, &db); err != nil {
		return nil, err
	}

	type scored struct {
		doc   *ai.Document
		score int
	}
	var results []scored

	for _, entry := range db {
		// Build searchable text from chunk content + parentText.
		var searchBuf strings.Builder
		var contentBuf strings.Builder
		for _, c := range entry.Doc.Content {
			contentBuf.WriteString(c.Text)
			searchBuf.WriteString(c.Text)
		}
		if pt, ok := entry.Doc.Metadata["parentText"].(string); ok {
			searchBuf.WriteString(" ")
			searchBuf.WriteString(pt)
		}
		fullText := strings.ToLower(searchBuf.String())

		matchCount := 0
		for _, kw := range keywords {
			if strings.Contains(fullText, kw) {
				matchCount++
			}
		}
		if matchCount == 0 {
			continue
		}

		doc := ai.DocumentFromText(contentBuf.String(), entry.Doc.Metadata)
		results = append(results, scored{doc: doc, score: matchCount})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	var docs []*ai.Document
	for i, r := range results {
		if i >= maxResults {
			break
		}
		docs = append(docs, r.doc)
	}
	return docs, nil
}
