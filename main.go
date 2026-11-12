package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/evaluators"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/ollama"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/joho/godotenv"
)

func ragVerboseLogs() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("RAG_VERBOSE_LOGS")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func debugf(format string, args ...any) {
	if ragVerboseLogs() {
		log.Printf(format, args...)
	}
}

func debugln(args ...any) {
	if ragVerboseLogs() {
		log.Println(args...)
	}
}

type conversationTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type conversationResolution struct {
	FollowUp           bool   `json:"follow_up"`
	StandaloneQuestion string `json:"standalone_question"`
}

type qaInput struct {
	Question string             `json:"question"`
	Tag      string             `json:"tag,omitempty"`
	History  []conversationTurn `json:"history,omitempty"`
}

func parseRerankScores(text string, n int) []int {
	scores := make([]int, n)
	for i := range scores {
		scores[i] = 5
	}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		idx := 0
		for _, ch := range strings.TrimSpace(parts[0]) {
			if ch >= '0' && ch <= '9' {
				idx = idx*10 + int(ch-'0')
			}
		}
		idx--
		if idx < 0 || idx >= n {
			continue
		}
		score := 0
		for _, ch := range strings.TrimSpace(parts[1]) {
			if ch >= '0' && ch <= '9' {
				score = score*10 + int(ch-'0')
			}
		}
		if score > 10 {
			score = 10
		}
		scores[idx] = score
	}
	return scores
}

func normalizeAllowedTags(tags []string) []string {
	seen := map[string]bool{}
	var out []string
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

func FilterByTag(docs []*ai.Document, allowedTags []string) []*ai.Document {
	allowedTags = normalizeAllowedTags(allowedTags)
	if len(allowedTags) == 0 {
		return docs
	}

	allowed := map[string]bool{}
	for _, t := range allowedTags {
		allowed[t] = true
	}

	filtered := make([]*ai.Document, 0, len(docs))
	for _, doc := range docs {
		tag := ""
		if doc.Metadata != nil {
			if v, ok := doc.Metadata["tag"].(string); ok {
				tag = strings.ToLower(strings.TrimSpace(v))
			}
		}
		if allowed[tag] {
			filtered = append(filtered, doc)
		}
	}

	return filtered
}

func AuditContamination(docs []*ai.Document, expectedTags []string, label string) {
	expectedTags = normalizeAllowedTags(expectedTags)
	if len(expectedTags) == 0 || len(docs) == 0 {
		return
	}

	allowed := map[string]bool{}
	for _, t := range expectedTags {
		allowed[t] = true
	}

	matched := 0
	tagCounts := map[string]int{}

	for _, doc := range docs {
		tag := ""
		if doc.Metadata != nil {
			if v, ok := doc.Metadata["tag"].(string); ok {
				tag = strings.ToLower(strings.TrimSpace(v))
			}
		}
		tagCounts[tag]++
		if allowed[tag] {
			matched++
		}
	}

	contaminated := len(docs) - matched
	purity := float64(matched) / float64(len(docs)) * 100.0

	log.Printf("[contamination:%s] total=%d matched=%d contaminated=%d purity=%.2f%% allowedTags=%v tagDistribution=%v",
		label, len(docs), matched, contaminated, purity, expectedTags, tagCounts)
}

func docMetadataString(doc *ai.Document, key string) string {
	if doc == nil || doc.Metadata == nil {
		return ""
	}
	if v, ok := doc.Metadata[key].(string); ok {
		return v
	}
	return ""
}

func previewText(text string, maxLen int) string {
	text = strings.Join(strings.Fields(text), " ")
	if len(text) > maxLen {
		return text[:maxLen] + "..."
	}
	return text
}

var questionTokenRE = regexp.MustCompile(`[[:alnum:]]+`)

var contextStopWords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
	"be": true, "by": true, "for": true, "from": true, "how": true, "in": true,
	"is": true, "it": true, "of": true, "on": true, "or": true, "the": true,
	"to": true, "what": true, "when": true, "where": true, "which": true,
	"who": true, "why": true, "with": true, "does": true, "do": true, "did": true,
	"this": true, "that": true, "these": true, "those": true, "main": true,
	"explain": true, "describe": true, "difference": true, "between": true,
}

func stableTextHash(text string) string {
	h := fnv.New64a()
	_, _ = h.Write([]byte(normalizeContextForDedup(text)))
	return fmt.Sprintf("%x", h.Sum64())
}

func importantQuestionTerms(question string) []string {
	raw := questionTokenRE.FindAllString(strings.ToLower(question), -1)
	seen := map[string]bool{}
	terms := make([]string, 0, len(raw))

	for _, tok := range raw {
		tok = strings.TrimSpace(tok)
		if len(tok) < 3 || contextStopWords[tok] || seen[tok] {
			continue
		}
		seen[tok] = true
		terms = append(terms, tok)
	}

	return terms
}

func addUniqueExpansion(out []string, seen map[string]bool, query string, max int) []string {
	query = strings.Join(strings.Fields(strings.TrimSpace(query)), " ")
	if query == "" {
		return out
	}
	key := strings.ToLower(query)
	if seen[key] || len(out) >= max {
		return out
	}
	seen[key] = true
	return append(out, query)
}

func buildQuestionExpansions(question string, max int) []string {
	terms := importantQuestionTerms(question)
	seen := map[string]bool{strings.ToLower(strings.Join(strings.Fields(question), " ")): true}
	expansions := make([]string, 0, max)
	if max <= 0 {
		return expansions
	}

	keyPhrase := strings.Join(terms, " ")
	lowerQuestion := strings.ToLower(question)

	if keyPhrase != "" {
		expansions = addUniqueExpansion(expansions, seen, keyPhrase, max)
	}

	if strings.Contains(lowerQuestion, "difference") || strings.Contains(lowerQuestion, "different") || strings.Contains(lowerQuestion, "compare") || strings.Contains(lowerQuestion, " vs ") || strings.Contains(lowerQuestion, " versus ") {
		expansions = addUniqueExpansion(expansions, seen, keyPhrase+" comparison difference", max)
		expansions = addUniqueExpansion(expansions, seen, keyPhrase+" advantages disadvantages", max)
	} else if strings.HasPrefix(lowerQuestion, "what is") || strings.HasPrefix(lowerQuestion, "what are") || strings.Contains(lowerQuestion, "define") {
		expansions = addUniqueExpansion(expansions, seen, keyPhrase+" definition meaning explanation", max)
		expansions = addUniqueExpansion(expansions, seen, keyPhrase+" concept summary", max)
	} else if strings.HasPrefix(lowerQuestion, "how") || strings.Contains(lowerQuestion, "process") || strings.Contains(lowerQuestion, "steps") {
		expansions = addUniqueExpansion(expansions, seen, keyPhrase+" process steps method", max)
		expansions = addUniqueExpansion(expansions, seen, keyPhrase+" application mechanism", max)
	} else if strings.Contains(lowerQuestion, "why") {
		expansions = addUniqueExpansion(expansions, seen, keyPhrase+" reason cause explanation", max)
		expansions = addUniqueExpansion(expansions, seen, keyPhrase+" motivation effect", max)
	}

	// Generic fallback expansions for lecture slides and PDFs, where headings often use short noun phrases.
	expansions = addUniqueExpansion(expansions, seen, keyPhrase+" lecture slide heading", max)
	expansions = addUniqueExpansion(expansions, seen, keyPhrase+" summary explanation", max)

	return expansions
}

func questionTermOverlap(question, text string) (matched int, total int, ratio float64) {
	terms := importantQuestionTerms(question)
	total = len(terms)
	if total == 0 {
		return 0, 0, 0
	}

	text = strings.ToLower(text)
	for _, term := range terms {
		if strings.Contains(text, term) {
			matched++
		}
	}

	return matched, total, float64(matched) / float64(total)
}

func metadataMatchScore(question string, doc *ai.Document) float64 {
	if doc == nil || doc.Metadata == nil {
		return 0
	}

	metaText := strings.ToLower(strings.Join([]string{
		docMetadataString(doc, "section"),
		docMetadataString(doc, "breadcrumb"),
		docMetadataString(doc, "pages"),
		docMetadataString(doc, "url"),
	}, " "))

	if strings.TrimSpace(metaText) == "" {
		return 0
	}

	matched, total, _ := questionTermOverlap(question, metaText)
	if total == 0 {
		return 0
	}
	return float64(matched) / float64(total)
}

func p16HeadingTerms(question string) []string {
	raw := questionTokenRE.FindAllString(question, -1)
	seen := map[string]bool{}
	terms := make([]string, 0, len(raw))

	for _, tok := range raw {
		trimmed := strings.TrimSpace(tok)
		lower := strings.ToLower(trimmed)
		if lower == "" || contextStopWords[lower] || seen[lower] {
			continue
		}

		// Keep normal terms and also short technical tokens/acronyms such as HA, IP, CoA, CN, 3D.
		if len(lower) < 3 {
			hasUpper := false
			hasDigit := false
			for _, r := range trimmed {
				if unicode.IsUpper(r) {
					hasUpper = true
				}
				if unicode.IsDigit(r) {
					hasDigit = true
				}
			}
			if !hasUpper && !hasDigit {
				continue
			}
		}

		seen[lower] = true
		terms = append(terms, lower)
	}

	return terms
}

func p16NormalizeHeadingText(text string) string {
	tokens := questionTokenRE.FindAllString(strings.ToLower(text), -1)
	if len(tokens) == 0 {
		return ""
	}
	return " " + strings.Join(tokens, " ") + " "
}

func p16HeadingCoverage(terms []string, normalizedText string) (matched int, total int, ratio float64) {
	total = len(terms)
	if total == 0 || strings.TrimSpace(normalizedText) == "" {
		return 0, total, 0
	}

	for _, term := range terms {
		needle := " " + strings.ToLower(strings.TrimSpace(term)) + " "
		if strings.Contains(normalizedText, needle) {
			matched++
		}
	}

	return matched, total, float64(matched) / float64(total)
}

var p16HeadingGenericTerms = map[string]bool{
	"level": true, "levels": true, "type": true, "types": true, "kind": true, "kinds": true,
	"part": true, "parts": true, "phase": true, "phases": true, "step": true, "steps": true,
	"main": true, "basic": true, "according": true, "definition": true, "goal": true, "task": true,
	"three": true, "four": true, "five": true, "six": true, "seven": true, "eight": true, "nine": true, "ten": true,
	"dbms": true, "database": true, "system": true, "systems": true, "technology": true, "technique": true,
	"used": true, "use": true, "using": true, "stand": true, "stands": true,
}

func p16CoreHeadingTerms(question string) []string {
	raw := questionTokenRE.FindAllString(question, -1)
	seen := map[string]bool{}
	terms := make([]string, 0, len(raw))

	for _, tok := range raw {
		trimmed := strings.TrimSpace(tok)
		lower := strings.ToLower(trimmed)
		if lower == "" || contextStopWords[lower] || p16HeadingGenericTerms[lower] || seen[lower] {
			continue
		}

		if len(lower) < 3 {
			hasUpper := false
			hasDigit := false
			for _, r := range trimmed {
				if unicode.IsUpper(r) {
					hasUpper = true
				}
				if unicode.IsDigit(r) {
					hasDigit = true
				}
			}
			if !hasUpper && !hasDigit {
				continue
			}
		}

		seen[lower] = true
		terms = append(terms, lower)
	}

	return terms
}

func p16QuestionHeadingPhrases(question string) []string {
	terms := p16CoreHeadingTerms(question)
	if len(terms) < 2 {
		return nil
	}

	seen := map[string]bool{}
	phrases := make([]string, 0)
	maxN := 4
	if len(terms) < maxN {
		maxN = len(terms)
	}

	for n := maxN; n >= 2; n-- {
		for i := 0; i+n <= len(terms); i++ {
			phrase := strings.Join(terms[i:i+n], " ")
			if seen[phrase] {
				continue
			}
			seen[phrase] = true
			phrases = append(phrases, phrase)
		}
	}

	return phrases
}

func p16ContainsQuestionHeadingPhrase(question string, normalizedText string) (bool, string) {
	if strings.TrimSpace(normalizedText) == "" {
		return false, ""
	}
	for _, phrase := range p16QuestionHeadingPhrases(question) {
		needle := " " + phrase + " "
		if strings.Contains(normalizedText, needle) {
			return true, strings.ReplaceAll(phrase, " ", "_")
		}
	}
	return false, ""
}

func p16ProtectedEvidenceStrength(question string, doc *ai.Document) float64 {
	if doc == nil {
		return 0
	}
	text := extractContextText(doc)
	matchedTerms, totalTerms, overlapRatio := questionTermOverlap(question, text)
	metaOverlap := metadataMatchScore(question, doc)
	lengthScore := float64(len(strings.TrimSpace(text))) / 600.0
	if lengthScore > 1.0 {
		lengthScore = 1.0
	}
	coverageScore := 0.0
	if totalTerms > 0 {
		coverageScore = float64(matchedTerms) / float64(totalTerms)
	}
	return lengthScore + overlapRatio + metaOverlap + coverageScore
}

func p16FirstLikelyTitleLine(text string) string {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// A short first line is often a slide title or chunk heading.
		if len([]rune(line)) <= 120 {
			return line
		}
		return ""
	}
	return ""
}

func p16MetadataHeadingBoost(question string, entry fusedDocEntry, app *appSetup) (float64, string) {
	if entry.doc == nil || entry.doc.Metadata == nil {
		return 0, "no_metadata"
	}

	terms := p16HeadingTerms(question)
	if len(terms) == 0 {
		return 0, "no_heading_terms"
	}

	keyPhrase := p16NormalizeHeadingText(strings.Join(terms, " "))

	sectionText := strings.Join([]string{
		docMetadataString(entry.doc, "section"),
		docMetadataString(entry.doc, "heading"),
		docMetadataString(entry.doc, "chunk_title"),
		docMetadataString(entry.doc, "chunkTitle"),
		docMetadataString(entry.doc, "title"),
	}, " ")
	breadcrumbText := docMetadataString(entry.doc, "breadcrumb")
	firstTitleLine := p16FirstLikelyTitleLine(entry.text)

	sectionNorm := p16NormalizeHeadingText(sectionText)
	breadcrumbNorm := p16NormalizeHeadingText(breadcrumbText)
	firstTitleNorm := p16NormalizeHeadingText(firstTitleLine)
	allMetaNorm := p16NormalizeHeadingText(strings.Join([]string{sectionText, breadcrumbText, firstTitleLine}, " "))

	sectionMatched, _, sectionRatio := p16HeadingCoverage(terms, sectionNorm)
	breadcrumbMatched, _, breadcrumbRatio := p16HeadingCoverage(terms, breadcrumbNorm)
	firstTitleMatched, _, firstTitleRatio := p16HeadingCoverage(terms, firstTitleNorm)
	allMetaMatched, _, allMetaRatio := p16HeadingCoverage(terms, allMetaNorm)

	// Strong boost: exact phrase or near-complete term coverage in the direct section/title field.
	if ok, phrase := p16ContainsQuestionHeadingPhrase(question, sectionNorm); ok {
		return app.headingBoostStrong, "section_core_phrase_" + phrase
	}
	if keyPhrase != "" && sectionNorm != "" && strings.Contains(sectionNorm, keyPhrase) && len(terms) >= 2 {
		return app.headingBoostStrong, "section_exact_phrase"
	}
	if sectionMatched >= 2 && sectionRatio >= 0.80 {
		return app.headingBoostStrong, fmt.Sprintf("section_high_coverage_%d/%d", sectionMatched, len(terms))
	}

	// Medium boost: breadcrumb match or inferred first-line title match.
	if keyPhrase != "" && breadcrumbNorm != "" && strings.Contains(breadcrumbNorm, keyPhrase) && len(terms) >= 2 {
		return app.headingBoostMedium, "breadcrumb_exact_phrase"
	}
	if breadcrumbMatched >= 2 && breadcrumbRatio >= 0.60 {
		return app.headingBoostMedium, fmt.Sprintf("breadcrumb_coverage_%d/%d", breadcrumbMatched, len(terms))
	}
	if firstTitleMatched >= 2 && firstTitleRatio >= 0.70 {
		return app.headingBoostMedium, fmt.Sprintf("first_title_coverage_%d/%d", firstTitleMatched, len(terms))
	}

	// Weak boost: partial metadata/title evidence. This helps but should not dominate semantic ranking.
	if allMetaMatched >= 2 && allMetaRatio >= 0.35 {
		return app.headingBoostWeak, fmt.Sprintf("metadata_partial_%d/%d", allMetaMatched, len(terms))
	}

	return 0, "no_heading_match"
}

func sortFusedDocEntries(entries []fusedDocEntry) {
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].score != entries[j].score {
			return entries[i].score > entries[j].score
		}

		if len(entries[i].sources) != len(entries[j].sources) {
			return len(entries[i].sources) > len(entries[j].sources)
		}

		if entries[i].bestRank != entries[j].bestRank {
			return entries[i].bestRank < entries[j].bestRank
		}

		if entries[i].firstOrder != entries[j].firstOrder {
			return entries[i].firstOrder < entries[j].firstOrder
		}

		return entries[i].dedupKey < entries[j].dedupKey
	})
}

func applyPreRerankHeadingBoost(question string, merged []fusedDocEntry, app *appSetup) []fusedDocEntry {
	if len(merged) == 0 || app == nil || !app.usePreRerankHeadingBoost {
		return merged
	}

	boosted := 0
	strong := 0
	medium := 0
	weak := 0

	for i := range merged {
		boost, reason := p16MetadataHeadingBoost(question, merged[i], app)
		if boost <= 0 {
			continue
		}

		oldScore := merged[i].score
		merged[i].score = oldScore * (1.0 + boost)
		boosted++

		if boost >= app.headingBoostStrong && app.headingBoostStrong > 0 {
			strong++
		} else if boost >= app.headingBoostMedium && app.headingBoostMedium > 0 {
			medium++
		} else {
			weak++
		}

		debugf(
			"[P16-A:headingBoost] reason=%s boost=%.3f oldScore=%.6f newScore=%.6f pages=%q section=%q breadcrumb=%q tag=%q text=%q",
			reason,
			boost,
			oldScore,
			merged[i].score,
			docMetadataString(merged[i].doc, "pages"),
			docMetadataString(merged[i].doc, "section"),
			docMetadataString(merged[i].doc, "breadcrumb"),
			docMetadataString(merged[i].doc, "tag"),
			previewText(merged[i].text, 120),
		)
	}

	if boosted > 0 {
		sortFusedDocEntries(merged)
	}

	log.Printf("[P16-A:headingBoost] enabled=%v candidates=%d boosted=%d strong=%d medium=%d weak=%d", app.usePreRerankHeadingBoost, len(merged), boosted, strong, medium, weak)

	return merged
}

func p17BExactSectionMatchBoost(question string, entry fusedDocEntry, app *appSetup) (float64, string) {
	if app == nil || !app.useP17BExactSectionBoost || entry.doc == nil || entry.doc.Metadata == nil {
		return 0, "disabled_or_no_metadata"
	}

	sectionText := strings.Join([]string{
		docMetadataString(entry.doc, "section"),
		docMetadataString(entry.doc, "heading"),
		docMetadataString(entry.doc, "chunk_title"),
		docMetadataString(entry.doc, "chunkTitle"),
		docMetadataString(entry.doc, "title"),
	}, " ")
	breadcrumbText := docMetadataString(entry.doc, "breadcrumb")
	firstTitleLine := p16FirstLikelyTitleLine(entry.text)

	sectionNorm := p16NormalizeHeadingText(sectionText)
	firstTitleNorm := p16NormalizeHeadingText(firstTitleLine)
	breadcrumbNorm := p16NormalizeHeadingText(breadcrumbText)

	// Strongest evidence: the candidate's direct section/title exactly contains
	// a core phrase from the question. Example: question phrase "storage hierarchy"
	// matches section "Storage Hierarchy".
	if ok, phrase := p16ContainsQuestionHeadingPhrase(question, sectionNorm); ok {
		return app.p17BExactSectionBoost, "p17b_section_exact_core_phrase_" + phrase
	}
	if ok, phrase := p16ContainsQuestionHeadingPhrase(question, firstTitleNorm); ok {
		return app.p17BExactSectionBoost, "p17b_first_title_exact_core_phrase_" + phrase
	}

	// Slightly weaker evidence: the breadcrumb contains the phrase. This helps when
	// the exact section is nested inside a larger path, but it should not dominate
	// as strongly as a direct section/title match.
	if ok, phrase := p16ContainsQuestionHeadingPhrase(question, breadcrumbNorm); ok {
		return app.p17BBreadcrumbBoost, "p17b_breadcrumb_exact_core_phrase_" + phrase
	}

	return 0, "no_p17b_exact_section_match"
}

func applyP17BExactSectionMatchBoost(question string, merged []fusedDocEntry, app *appSetup) []fusedDocEntry {
	if len(merged) == 0 || app == nil || !app.useP17BExactSectionBoost {
		return merged
	}

	boosted := 0
	sectionBoosted := 0
	breadcrumbBoosted := 0

	for i := range merged {
		boost, reason := p17BExactSectionMatchBoost(question, merged[i], app)
		if boost <= 0 {
			continue
		}

		oldScore := merged[i].score
		merged[i].score = oldScore * (1.0 + boost)
		boosted++

		if strings.Contains(reason, "breadcrumb") {
			breadcrumbBoosted++
		} else {
			sectionBoosted++
		}

		debugf(
			"[P17-B2:exactSectionBoost] reason=%s boost=%.3f oldScore=%.6f newScore=%.6f pages=%q section=%q breadcrumb=%q tag=%q text=%q",
			reason,
			boost,
			oldScore,
			merged[i].score,
			docMetadataString(merged[i].doc, "pages"),
			docMetadataString(merged[i].doc, "section"),
			docMetadataString(merged[i].doc, "breadcrumb"),
			docMetadataString(merged[i].doc, "tag"),
			previewText(merged[i].text, 120),
		)
	}

	if boosted > 0 {
		sortFusedDocEntries(merged)
	}

	log.Printf("[P17-B2:exactSectionBoost] enabled=%v candidates=%d boosted=%d direct=%d breadcrumb=%d", app.useP17BExactSectionBoost, len(merged), boosted, sectionBoosted, breadcrumbBoosted)

	return merged
}

func p17BDirectHeadingPhraseForDoc(question string, doc *ai.Document) (bool, string) {
	if doc == nil || doc.Metadata == nil {
		return false, ""
	}

	sectionText := strings.Join([]string{
		docMetadataString(doc, "section"),
		docMetadataString(doc, "heading"),
		docMetadataString(doc, "chunk_title"),
		docMetadataString(doc, "chunkTitle"),
		docMetadataString(doc, "title"),
	}, " ")
	contextTitle := p16FirstLikelyTitleLine(extractContextText(doc))

	sectionNorm := p16NormalizeHeadingText(sectionText)
	contextTitleNorm := p16NormalizeHeadingText(contextTitle)

	for _, phrase := range p16QuestionHeadingPhrases(question) {
		needle := " " + phrase + " "
		if sectionNorm != "" && strings.Contains(sectionNorm, needle) {
			return true, phrase
		}
		if contextTitleNorm != "" && strings.Contains(contextTitleNorm, needle) {
			return true, phrase
		}
	}

	return false, ""
}

func p17BHeadingMetaNormForDoc(doc *ai.Document) string {
	if doc == nil {
		return ""
	}

	sectionText := strings.Join([]string{
		docMetadataString(doc, "section"),
		docMetadataString(doc, "heading"),
		docMetadataString(doc, "chunk_title"),
		docMetadataString(doc, "chunkTitle"),
		docMetadataString(doc, "title"),
	}, " ")
	breadcrumbText := docMetadataString(doc, "breadcrumb")
	contextTitle := p16FirstLikelyTitleLine(extractContextText(doc))

	return p16NormalizeHeadingText(strings.Join([]string{sectionText, breadcrumbText, contextTitle}, " "))
}

func p17BPhraseOverlapWithHeading(phrase string, headingNorm string) float64 {
	phraseTerms := strings.Fields(strings.ToLower(strings.TrimSpace(phrase)))
	if len(phraseTerms) == 0 || strings.TrimSpace(headingNorm) == "" {
		return 0
	}

	matched := 0
	for _, term := range phraseTerms {
		if term == "" {
			continue
		}
		if strings.Contains(headingNorm, " "+term+" ") {
			matched++
		}
	}

	return float64(matched) / float64(len(phraseTerms))
}

func p17BConflictingSiblingReasonForDoc(question string, doc *ai.Document, targetPhrases []string, minOverlap float64) (bool, string) {
	if doc == nil || len(targetPhrases) == 0 {
		return false, ""
	}

	// Direct exact matches are the protected target, not siblings to demote.
	if ok, phrase := p17BDirectHeadingPhraseForDoc(question, doc); ok {
		return false, "direct_match_" + strings.ReplaceAll(phrase, " ", "_")
	}

	headingNorm := p17BHeadingMetaNormForDoc(doc)
	if strings.TrimSpace(headingNorm) == "" {
		return false, ""
	}

	bestOverlap := 0.0
	bestPhrase := ""
	for _, phrase := range targetPhrases {
		phrase = strings.ToLower(strings.TrimSpace(phrase))
		if phrase == "" {
			continue
		}

		// If the whole target phrase appears anywhere in the heading metadata,
		// do not demote it. It is still relevant evidence.
		if strings.Contains(headingNorm, " "+phrase+" ") {
			continue
		}

		overlap := p17BPhraseOverlapWithHeading(phrase, headingNorm)
		if overlap > bestOverlap {
			bestOverlap = overlap
			bestPhrase = phrase
		}
	}

	if bestOverlap >= minOverlap && bestPhrase != "" {
		return true, fmt.Sprintf("sibling_overlap_%s_%.2f", strings.ReplaceAll(bestPhrase, " ", "_"), bestOverlap)
	}

	return false, ""
}

func p17BCollectDirectTargetPhrasesByTagFromEntries(question string, merged []fusedDocEntry) map[string][]string {
	byTag := map[string][]string{}
	seen := map[string]map[string]bool{}

	for _, entry := range merged {
		tag := strings.ToLower(strings.TrimSpace(docMetadataString(entry.doc, "tag")))
		if tag == "" {
			continue
		}
		ok, phrase := p17BDirectHeadingPhraseForDoc(question, entry.doc)
		if !ok || phrase == "" {
			continue
		}
		if seen[tag] == nil {
			seen[tag] = map[string]bool{}
		}
		if seen[tag][phrase] {
			continue
		}
		seen[tag][phrase] = true
		byTag[tag] = append(byTag[tag], phrase)
	}

	return byTag
}

func applyP17BSiblingSectionDemotion(question string, merged []fusedDocEntry, app *appSetup) []fusedDocEntry {
	if len(merged) == 0 || app == nil || !app.useP17BSiblingDemotion {
		return merged
	}

	minOverlap := app.p17BSiblingMinOverlap
	if minOverlap <= 0 || minOverlap > 1 {
		minOverlap = 0.34
	}

	targetsByTag := p17BCollectDirectTargetPhrasesByTagFromEntries(question, merged)
	if len(targetsByTag) == 0 {
		log.Printf("[P17-B2:siblingDemotion] enabled=%v candidates=%d targets=0 demoted=0", app.useP17BSiblingDemotion, len(merged))
		return merged
	}

	demoted := 0
	for i := range merged {
		tag := strings.ToLower(strings.TrimSpace(docMetadataString(merged[i].doc, "tag")))
		targets := targetsByTag[tag]
		if len(targets) == 0 {
			continue
		}

		conflict, reason := p17BConflictingSiblingReasonForDoc(question, merged[i].doc, targets, minOverlap)
		if !conflict {
			continue
		}

		oldScore := merged[i].score
		factor := 1.0 - app.p17BSiblingDemotion
		if factor < 0.05 {
			factor = 0.05
		}
		if factor > 1.0 {
			factor = 1.0
		}
		merged[i].score = oldScore * factor
		demoted++

		debugf(
			"[P17-B2:siblingDemotion] reason=%s factor=%.3f oldScore=%.6f newScore=%.6f pages=%q section=%q breadcrumb=%q tag=%q text=%q",
			reason,
			factor,
			oldScore,
			merged[i].score,
			docMetadataString(merged[i].doc, "pages"),
			docMetadataString(merged[i].doc, "section"),
			docMetadataString(merged[i].doc, "breadcrumb"),
			docMetadataString(merged[i].doc, "tag"),
			previewText(merged[i].text, 120),
		)
	}

	if demoted > 0 {
		sortFusedDocEntries(merged)
	}

	log.Printf("[P17-B2:siblingDemotion] enabled=%v candidates=%d targetTags=%d demoted=%d demotion=%.2f minOverlap=%.2f", app.useP17BSiblingDemotion, len(merged), len(targetsByTag), demoted, app.p17BSiblingDemotion, minOverlap)
	return merged
}

func p16FinalHeadingProtectionPriority(question string, doc *ai.Document) (int, string) {
	if doc == nil || doc.Metadata == nil {
		return 0, "no_metadata"
	}

	terms := p16HeadingTerms(question)
	if len(terms) == 0 {
		return 0, "no_heading_terms"
	}

	keyPhrase := p16NormalizeHeadingText(strings.Join(terms, " "))
	contextTitle := p16FirstLikelyTitleLine(extractContextText(doc))

	sectionText := strings.Join([]string{
		docMetadataString(doc, "section"),
		docMetadataString(doc, "heading"),
		docMetadataString(doc, "chunk_title"),
		docMetadataString(doc, "chunkTitle"),
		docMetadataString(doc, "title"),
	}, " ")
	breadcrumbText := docMetadataString(doc, "breadcrumb")

	sectionNorm := p16NormalizeHeadingText(sectionText)
	breadcrumbNorm := p16NormalizeHeadingText(breadcrumbText)
	contextTitleNorm := p16NormalizeHeadingText(contextTitle)
	allMetaNorm := p16NormalizeHeadingText(strings.Join([]string{sectionText, breadcrumbText, contextTitle}, " "))

	sectionMatched, _, sectionRatio := p16HeadingCoverage(terms, sectionNorm)
	breadcrumbMatched, _, breadcrumbRatio := p16HeadingCoverage(terms, breadcrumbNorm)
	contextTitleMatched, _, contextTitleRatio := p16HeadingCoverage(terms, contextTitleNorm)
	allMetaMatched, _, allMetaRatio := p16HeadingCoverage(terms, allMetaNorm)

	// Priority 3: direct section/title match. This is the P16-B/P16-C protection target.
	// Example: question contains "storage hierarchy" and section is "Storage Hierarchy".
	if ok, phrase := p16ContainsQuestionHeadingPhrase(question, sectionNorm); ok {
		return 3, "section_core_phrase_" + phrase
	}
	if ok, phrase := p16ContainsQuestionHeadingPhrase(question, contextTitleNorm); ok {
		return 3, "context_title_core_phrase_" + phrase
	}
	if keyPhrase != "" && len(terms) >= 2 && sectionNorm != "" && strings.Contains(sectionNorm, keyPhrase) {
		return 3, "section_exact_phrase"
	}
	if sectionMatched >= 2 && sectionRatio >= 0.80 {
		return 3, fmt.Sprintf("section_high_coverage_%d/%d", sectionMatched, len(terms))
	}

	// Priority 2: strong breadcrumb or inferred first-title match.
	if ok, phrase := p16ContainsQuestionHeadingPhrase(question, breadcrumbNorm); ok {
		return 2, "breadcrumb_core_phrase_" + phrase
	}
	if keyPhrase != "" && len(terms) >= 2 && breadcrumbNorm != "" && strings.Contains(breadcrumbNorm, keyPhrase) {
		return 2, "breadcrumb_exact_phrase"
	}
	if breadcrumbMatched >= 2 && breadcrumbRatio >= 0.70 {
		return 2, fmt.Sprintf("breadcrumb_high_coverage_%d/%d", breadcrumbMatched, len(terms))
	}
	if contextTitleMatched >= 2 && contextTitleRatio >= 0.70 {
		return 2, fmt.Sprintf("context_title_coverage_%d/%d", contextTitleMatched, len(terms))
	}

	// Priority 1: weak metadata evidence. It is logged but normally not enough to override reranking.
	if allMetaMatched >= 2 && allMetaRatio >= 0.50 {
		return 1, fmt.Sprintf("metadata_partial_%d/%d", allMetaMatched, len(terms))
	}

	return 0, "no_heading_protection"
}

func isDefinitionQuestion(question string) bool {
	q := strings.ToLower(strings.TrimSpace(question))
	patterns := []string{
		"what is ",
		"what are ",
		"define ",
		"what does ",
		"stand for",
		"meaning of",
	}
	for _, p := range patterns {
		if strings.Contains(q, p) {
			return true
		}
	}
	return false
}

func isAbbreviationQuestion(question string) bool {
	q := strings.ToLower(strings.TrimSpace(question))
	patterns := []string{
		"stand for",
		"what does",
		"abbreviation",
		"acronym",
		"meaning of",
	}
	for _, p := range patterns {
		if strings.Contains(q, p) {
			return true
		}
	}
	return false
}

func isFormulaQuestion(question string) bool {
	q := strings.ToLower(strings.TrimSpace(question))
	patterns := []string{
		"formula",
		"cost",
		"costs",
		"equation",
		"calculate",
		"calculation",
		"complexity",
		"estimate",
		"computed",
		"derived",
	}
	for _, p := range patterns {
		if strings.Contains(q, p) {
			return true
		}
	}
	return false
}

func isExactValueQuestion(question string) bool {
	q := strings.ToLower(strings.TrimSpace(question))
	patterns := []string{
		"maximum",
		"minimum",
		"how many",
		"how much",
		"which year",
		"what year",
		"which date",
		"what date",
		"what price",
		"which price",
		"what value",
		"which value",
		"which version",
		"what version",
		"which programming language",
		"what programming language",
		"exact",
		"specific",
		"speed",
		"rate",
		"percentage",
		"percent",
	}
	for _, p := range patterns {
		if strings.Contains(q, p) {
			return true
		}
	}
	return false
}

func isWhyHowQuestion(question string) bool {
	q := strings.ToLower(strings.TrimSpace(question))
	return strings.HasPrefix(q, "why ") || strings.HasPrefix(q, "how ")
}

func cleanQuestionToken(s string) string {
	return strings.TrimFunc(s, func(r rune) bool {
		return !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_')
	})
}

func extractLikelyAbbreviation(question string) string {
	words := strings.Fields(question)

	for i, w := range words {
		clean := cleanQuestionToken(w)
		if strings.EqualFold(clean, "does") && i+1 < len(words) {
			next := cleanQuestionToken(words[i+1])
			if len(next) >= 2 && len(next) <= 12 {
				return next
			}
		}
	}

	for _, w := range words {
		clean := cleanQuestionToken(w)
		if len(clean) < 2 || len(clean) > 12 {
			continue
		}

		hasUpper := false
		hasLetter := false
		for _, r := range clean {
			if unicode.IsLetter(r) {
				hasLetter = true
			}
			if unicode.IsUpper(r) {
				hasUpper = true
			}
		}

		if hasLetter && hasUpper {
			return clean
		}
	}

	return ""
}

func abbreviationEvidenceScore(question, text string) float64 {
	abbr := extractLikelyAbbreviation(question)
	if abbr == "" {
		return 0
	}

	lowerText := strings.ToLower(text)
	lowerAbbr := strings.ToLower(abbr)
	score := 0.0

	if strings.Contains(lowerText, lowerAbbr) {
		score += 0.15
	}

	// Generic pattern evidence: long phrase followed by abbreviation in parentheses.
	parenPattern := "(" + lowerAbbr + ")"
	if strings.Contains(lowerText, parenPattern) {
		score += 0.60
	}

	// Generic pattern evidence: abbreviation followed by a definition delimiter.
	for _, delim := range []string{" = ", ": ", " - ", " – ", " — "} {
		if strings.Contains(lowerText, lowerAbbr+delim) {
			score += 0.55
		}
	}

	// Generic pattern evidence: explicit phrase.
	if strings.Contains(lowerText, lowerAbbr+" stands for") || strings.Contains(lowerText, "stands for "+lowerAbbr) {
		score += 0.65
	}

	if strings.Contains(lowerText, "abbreviated as "+lowerAbbr) || strings.Contains(lowerText, "denoted as "+lowerAbbr) {
		score += 0.45
	}

	if score > 1.0 {
		score = 1.0
	}
	return score
}

func formulaEvidenceScore(question, text string) float64 {
	score := 0.0

	for _, marker := range []string{"=", "+", "-", "/", "*", "×", "·", "∑", "√", "≤", "≥", "<", ">"} {
		if strings.Contains(text, marker) {
			score += 0.06
		}
	}

	// Generic function/formula-like evidence: parentheses with symbols or variables.
	if strings.Contains(text, "(") && strings.Contains(text, ")") {
		score += 0.18
	}
	if strings.Contains(strings.ToLower(text), "cost") || strings.Contains(strings.ToLower(text), "formula") {
		score += 0.20
	}

	matched, total, overlap := questionTermOverlap(question, text)
	if total > 0 && matched >= 2 {
		score += 0.30 * overlap
	}

	if score > 1.0 {
		score = 1.0
	}
	return score
}

func hasDigit(text string) bool {
	for _, r := range text {
		if unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

func valueEvidenceScore(question, text string) float64 {
	score := 0.0
	lowerText := strings.ToLower(text)
	lowerQuestion := strings.ToLower(question)

	if hasDigit(text) {
		score += 0.25
	}

	valueMarkers := []string{"%", "€", "$", " ms", " sec", " second", " hz", "khz", "mhz", "ghz", " kb", " mb", " gb", " tb", "km/h", "m/s", "bps", "mbps", "gbps", "version", "year", "date"}
	for _, marker := range valueMarkers {
		if strings.Contains(lowerText, marker) {
			score += 0.30
			break
		}
	}

	if strings.Contains(lowerQuestion, "programming language") {
		languageMarkers := []string{"go", "golang", "java", "python", "c++", "c#", "javascript", "typescript", "rust", "ruby", "php", "swift", "kotlin", "scala", "matlab", "sql"}
		for _, marker := range languageMarkers {
			if strings.Contains(lowerText, marker) {
				score += 0.35
				break
			}
		}
	}

	matched, total, overlap := questionTermOverlap(question, text)
	if total > 0 && matched >= 2 {
		score += 0.45 * overlap
	}

	if score > 1.0 {
		score = 1.0
	}
	return score
}

// ─────────────────────────────────────────────
// P17-E2: narrow unsupported-question validation
// ─────────────────────────────────────────────
//
// P17-E failed because it applied a broad final support gate to almost every
// question. P17-E2 keeps all P17-B2 routing/section protection unchanged and
// only checks questions that explicitly ask for precise facts that are easy to
// hallucinate from related context, such as prices, inventors, implementation
// languages, maximum/supported speeds, dates/versions, or 5G-specific facts.
// Normal conceptual questions are not gated by E2.

func p17E2NormalizedText(text string) string {
	return normalizeProfileText(text)
}

func p17E2RiskType(question string) string {
	q := strings.ToLower(strings.TrimSpace(question))

	if strings.Contains(q, "programming language") || strings.Contains(q, "implementation language") || strings.Contains(q, "implemented in") || strings.Contains(q, "written in") {
		return "programming_language"
	}
	if strings.Contains(q, "who invented") || strings.Contains(q, "inventor") || strings.Contains(q, "invented by") || strings.Contains(q, "who created") || strings.Contains(q, "created by") {
		return "inventor"
	}
	if strings.Contains(q, "maximum speed") || strings.Contains(q, "max speed") || strings.Contains(q, "supported speed") || (strings.Contains(q, "speed") && strings.Contains(q, "supported")) || strings.Contains(q, "how fast") {
		return "speed"
	}
	if strings.Contains(q, "price") || strings.Contains(q, "how much") || strings.Contains(q, "license cost") || strings.Contains(q, "licence cost") || strings.Contains(q, "cost of the license") || strings.Contains(q, "cost of the licence") {
		return "price"
	}
	if strings.Contains(q, "which year") || strings.Contains(q, "what year") || strings.Contains(q, "which date") || strings.Contains(q, "what date") || strings.Contains(q, "which version") || strings.Contains(q, "what version") {
		return "date_version"
	}

	// 5G is a useful narrow unsupported-risk marker because the AMCN slides used
	// in this experiment discuss Mobile IP / WLAN mobility, but not 5G mobility
	// management. Do not generalize this to other domains; it is just a technical
	// token-presence check.
	if strings.Contains(q, "5g") || strings.Contains(q, "5 g") {
		return "5g_specific"
	}

	return ""
}

func p17E2ContextBlocks(finalContext string) []string {
	finalContext = strings.TrimSpace(finalContext)
	if finalContext == "" {
		return nil
	}

	parts := strings.Split(finalContext, "\n\n[")
	blocks := make([]string, 0, len(parts))
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if i > 0 {
			p = "[" + p
		}
		blocks = append(blocks, p)
	}
	if len(blocks) == 0 {
		blocks = append(blocks, finalContext)
	}
	return blocks
}

func p17E2ContainsAnyNormalized(text string, terms []string) bool {
	norm := p17E2NormalizedText(text)
	if norm == "" {
		return false
	}
	for _, term := range terms {
		term = strings.ToLower(strings.TrimSpace(term))
		if term == "" {
			continue
		}
		if strings.Contains(norm, " "+term+" ") {
			return true
		}
	}
	return false
}

func p17E2HasSpeedUnit(text string) bool {
	lower := strings.ToLower(text)
	if !hasDigit(text) {
		return false
	}
	markers := []string{
		"km/h", "kmh", "m/s", "mph",
		"bps", "kbps", "mbps", "gbps", "bit/s", "bits/s", "kbit/s", "mbit/s", "gbit/s",
		"maximum speed", "max speed", "supported speed",
	}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func p17E2HasPriceEvidence(text string) bool {
	lower := strings.ToLower(text)
	if !hasDigit(text) {
		return false
	}
	markers := []string{"€", "$", " eur", " euro", " usd", " price", "license fee", "licence fee", "cost"}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func p17E2HasDateVersionEvidence(text string) bool {
	lower := strings.ToLower(text)
	if regexp.MustCompile(`\b(19|20)\d{2}\b`).MatchString(lower) {
		return true
	}
	markers := []string{"version", "date", "year", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func p17E2HasInventorEvidence(text string) bool {
	lower := strings.ToLower(text)
	markers := []string{"invented by", "inventor", "invented", "created by", "developed by", "designed by"}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func p17E2HasProgrammingLanguageEvidence(text string) bool {
	// Keep SQL out of this list. Database slides mention SQL often, but that is
	// not evidence for the implementation/programming language of a DBMS/tool.
	languages := []string{"go", "golang", "java", "python", "javascript", "typescript", "rust", "ruby", "php", "swift", "kotlin", "scala", "matlab", "csharp", "c++"}
	return p17E2ContainsAnyNormalized(text, languages) || strings.Contains(strings.ToLower(text), "c++") || strings.Contains(strings.ToLower(text), "c#")
}

func p17E2HasRiskEvidence(question, finalContext, riskType string) bool {
	if strings.TrimSpace(finalContext) == "" {
		return false
	}

	q := strings.ToLower(question)
	blocks := p17E2ContextBlocks(finalContext)
	if len(blocks) == 0 {
		return false
	}

	switch riskType {
	case "5g_specific":
		return strings.Contains(strings.ToLower(finalContext), "5g") || strings.Contains(strings.ToLower(finalContext), "5 g")

	case "speed":
		// If the user asks for handoff/handover speed, avoid accepting an unrelated
		// throughput value from another wireless standard. Require speed evidence in
		// a block that also talks about handoff/handover.
		requiresHandoffBlock := strings.Contains(q, "handoff") || strings.Contains(q, "handover")
		for _, block := range blocks {
			lowerBlock := strings.ToLower(block)
			if !p17E2HasSpeedUnit(block) {
				continue
			}
			if requiresHandoffBlock && !(strings.Contains(lowerBlock, "handoff") || strings.Contains(lowerBlock, "handover")) {
				continue
			}
			return true
		}
		return false

	case "price":
		for _, block := range blocks {
			if p17E2HasPriceEvidence(block) {
				return true
			}
		}
		return false

	case "date_version":
		for _, block := range blocks {
			if p17E2HasDateVersionEvidence(block) {
				return true
			}
		}
		return false

	case "inventor":
		for _, block := range blocks {
			if p17E2HasInventorEvidence(block) {
				return true
			}
		}
		return false

	case "programming_language":
		for _, block := range blocks {
			if p17E2HasProgrammingLanguageEvidence(block) {
				return true
			}
		}
		return false
	}

	return true
}

func p17E2FinalSupportDecision(question, finalContext string, app *appSetup) (bool, string) {
	if app == nil || !app.useP17E2UnsupportedValidation {
		return true, "p17e2_disabled"
	}

	riskType := p17E2RiskType(question)
	if riskType == "" {
		return true, "not_risky_question"
	}

	if p17E2HasRiskEvidence(question, finalContext, riskType) {
		return true, "risk_evidence_found_" + riskType
	}

	return false, "missing_risk_evidence_" + riskType
}

func chunkQualityScore(question string, doc *ai.Document, text string, rerankScore int, useReranker bool) float64 {
	_, _, textOverlap := questionTermOverlap(question, text)
	metaOverlap := metadataMatchScore(question, doc)

	quality := 0.0

	// Reranker remains the main signal when enabled, but it is no longer the only signal.
	if useReranker {
		quality += 0.55 * (float64(rerankScore) / 10.0)
	} else {
		quality += 0.25
	}

	// The chunk itself must share meaningful terms with the question.
	quality += 0.30 * textOverlap

	// Section/breadcrumb match is very valuable for slide decks.
	quality += 0.15 * metaOverlap

	// Definition and formula questions need more direct lexical grounding.
	if isDefinitionQuestion(question) {
		quality += 0.05 * textOverlap
	}
	if isFormulaQuestion(question) {
		quality += 0.05 * textOverlap
	}

	if quality > 1.0 {
		quality = 1.0
	}
	return quality
}

func shouldKeepCandidateC54(question string, doc *ai.Document, text string, rerankScore int, useReranker bool, alreadyKept int) (bool, string, float64) {
	matchedTerms, totalTerms, textOverlap := questionTermOverlap(question, text)
	metaOverlap := metadataMatchScore(question, doc)
	quality := chunkQualityScore(question, doc, text, rerankScore, useReranker)

	// Generic evidence-type validation. These rules are pattern-based and portable
	// across PDFs; they do not hard-code dataset-specific abbreviations or formulas.
	if isAbbreviationQuestion(question) {
		abbrScore := abbreviationEvidenceScore(question, text)
		if abbrScore >= 0.60 {
			return true, "abbreviation_expansion_evidence", quality
		}
		if useReranker && rerankScore >= 9 && abbrScore >= 0.35 && (textOverlap >= 0.20 || metaOverlap >= 0.20) {
			return true, "abbreviation_partial_evidence", quality
		}
		return false, "abbreviation_without_expansion_evidence", quality
	}

	if isFormulaQuestion(question) {
		formulaScore := formulaEvidenceScore(question, text)
		if formulaScore >= 0.50 && (matchedTerms >= 2 || metaOverlap >= 0.25 || rerankScore >= 8) {
			return true, "formula_evidence", quality
		}
		return false, "formula_question_without_formula_evidence", quality
	}

	if isExactValueQuestion(question) {
		valueScore := valueEvidenceScore(question, text)
		if valueScore >= 0.65 && (matchedTerms >= 2 || metaOverlap >= 0.25 || rerankScore >= 8) {
			return true, "exact_value_evidence", quality
		}
		return false, "exact_value_not_supported", quality
	}

	// Normal C5.3-style behavior for all other questions.
	if useReranker && rerankScore >= 9 && alreadyKept == 0 {
		return true, "strong_top_rerank", quality
	}

	if useReranker && rerankScore >= 8 && (textOverlap >= 0.20 || metaOverlap >= 0.20 || matchedTerms >= 2) {
		return true, "strong_rerank_grounded", quality
	}

	if useReranker && rerankScore >= 5 && (textOverlap >= 0.35 || metaOverlap >= 0.35 || matchedTerms >= 3) {
		return true, "medium_rerank_good_overlap", quality
	}

	if isWhyHowQuestion(question) && quality >= 0.55 && matchedTerms >= 2 {
		return true, "why_how_grounded", quality
	}

	if isDefinitionQuestion(question) && quality >= 0.60 && (matchedTerms >= 2 || metaOverlap >= 0.30) {
		return true, "definition_grounded", quality
	}

	if !useReranker && quality >= 0.50 && (textOverlap >= 0.35 || metaOverlap >= 0.35 || matchedTerms >= 3) {
		return true, "no_reranker_quality_pass", quality
	}

	if totalTerms == 0 {
		return true, "no_question_terms", quality
	}

	return false, "low_question_alignment", quality
}

var contextWhitespaceRE = regexp.MustCompile(`\s+`)

func normalizeContextForDedup(text string) string {
	text = strings.ToLower(text)
	text = contextWhitespaceRE.ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

func extractDocumentText(doc *ai.Document) string {
	if doc == nil {
		return ""
	}

	var sb strings.Builder
	for _, part := range doc.Content {
		sb.WriteString(part.Text)
	}
	return sb.String()
}

func extractContextText(doc *ai.Document) string {
	if doc == nil {
		return ""
	}

	if doc.Metadata != nil {
		if pt, ok := doc.Metadata["parentText"].(string); ok && strings.TrimSpace(pt) != "" {
			return pt
		}
	}

	return extractDocumentText(doc)
}

func tokenSet(text string) map[string]bool {
	out := map[string]bool{}
	for _, tok := range strings.Fields(text) {
		if len(tok) < 3 {
			continue
		}
		out[tok] = true
	}
	return out
}

func jaccardSimilarity(a, b string) float64 {
	as := tokenSet(a)
	bs := tokenSet(b)
	if len(as) == 0 || len(bs) == 0 {
		return 0
	}

	intersection := 0
	for tok := range as {
		if bs[tok] {
			intersection++
		}
	}

	union := len(as) + len(bs) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func containmentSimilarity(shorter, longer string) float64 {
	ss := tokenSet(shorter)
	ls := tokenSet(longer)
	if len(ss) == 0 || len(ls) == 0 {
		return 0
	}

	contained := 0
	for tok := range ss {
		if ls[tok] {
			contained++
		}
	}
	return float64(contained) / float64(len(ss))
}

func isNearDuplicateContext(candidate string, existing []string) (bool, string) {
	candidate = normalizeContextForDedup(candidate)
	if candidate == "" {
		return true, "empty_context"
	}

	for _, prev := range existing {
		prev = normalizeContextForDedup(prev)
		if prev == "" {
			continue
		}

		if candidate == prev {
			return true, "exact_duplicate"
		}

		// Keep the near-duplicate filter conservative. This removes repeated parentText/slide
		// variants but avoids deleting two genuinely different chunks from the same section.
		jac := jaccardSimilarity(candidate, prev)
		if jac >= 0.92 {
			return true, fmt.Sprintf("near_duplicate_jaccard_%.2f", jac)
		}

		shorter := candidate
		longer := prev
		if len(shorter) > len(longer) {
			shorter, longer = longer, shorter
		}

		if len(shorter) >= 250 {
			cont := containmentSimilarity(shorter, longer)
			if cont >= 0.90 {
				return true, fmt.Sprintf("near_duplicate_containment_%.2f", cont)
			}
		}
	}

	return false, ""
}

type appSetup struct {
	ctx                           context.Context
	genk                          *genkit.Genkit
	llm                           ai.Model
	chunkSize                     int
	chunkOverlap                  int
	useReranker                   bool
	useHyDE                       bool
	useQueryExpansion             bool
	queryExpansionK               int
	queryExpansionRetrieveK       int
	useKeyword                    bool
	fusionMethod                  string
	usePreRerankHeadingBoost      bool
	headingBoostStrong            float64
	headingBoostMedium            float64
	headingBoostWeak              float64
	useP17BExactSectionBoost      bool
	p17BExactSectionBoost         float64
	p17BBreadcrumbBoost           float64
	useP17BSiblingDemotion        bool
	p17BSiblingDemotion           float64
	p17BSiblingMinOverlap         float64
	useFinalHeadingProtection     bool
	finalHeadingMinPriority       int
	useFinalHeadingForceInclude   bool
	finalHeadingForcePriority     int
	finalHeadingForceMax          int
	finalHeadingForceMinChars     int
	useP17E2UnsupportedValidation bool
	allowedTags                   []string
	autoTagRouting                bool
	autoTagMode                   string
	autoTagPreK                   int
	autoTagMinDominant            int
	autoTagTop2Min                int
	autoTagMinScore               float64
	autoTagSingleMargin           float64
	autoTagHighMargin             float64
	autoTagTop2Share              float64
	autoTagProfileWeight          float64
	autoTagMaxSelected            int
	autoTagProfileLimit           int
	finalContextK                 int
	rerankMinScore                int
	maxContextChars               int
	store                         VectorStore
	hydePrompt                    ai.Prompt
	rerankPrompt                  ai.Prompt
	qaPrompt                      ai.Prompt
	useConversationalResolution   bool
	ollamaAddr                    string
	uniAPIBaseURL                 string
	uniAPIKey                     string
	conversationModel             string
	conversationHistoryMessages   int
}

func setup() *appSetup {
	_ = godotenv.Overload()

	llmModel := os.Getenv("LLM_MODEL")
	if llmModel == "" {
		llmModel = "googleai/gemini-2.5-flash"
	}

	embeddingModel := os.Getenv("EMBEDDING_MODEL")
	if embeddingModel == "" {
		embeddingModel = "gemma3:27b"
	}

	embeddingDim := atoiOrDefault(os.Getenv("EMBEDDING_DIM"), 768)

	chunkSize := atoiOrDefault(os.Getenv("CHUNK_SIZE"), 1500)
	chunkOverlap := atoiOrDefault(os.Getenv("CHUNK_OVERLAP"), 200)

	finalContextK := atoiOrDefault(os.Getenv("FINAL_CONTEXT_K"), 5)
	if finalContextK <= 0 {
		finalContextK = 5
	}

	rerankMinScore := atoiOrDefault(os.Getenv("RERANK_MIN_SCORE"), 2)
	if rerankMinScore < 0 {
		rerankMinScore = 0
	}
	if rerankMinScore > 10 {
		rerankMinScore = 10
	}

	// 0 means no character cap, preserving C5.1 behavior by default.
	maxContextChars := atoiOrDefault(os.Getenv("MAX_CONTEXT_CHARS"), 0)

	useReranker := strings.ToLower(os.Getenv("USE_RERANKER")) == "true"
	useHyDE := strings.ToLower(os.Getenv("USE_HYDE")) == "true"
	useQueryExpansion := strings.ToLower(os.Getenv("USE_QUERY_EXPANSION")) == "true"
	queryExpansionK := atoiOrDefault(os.Getenv("QUERY_EXPANSION_K"), 3)
	if queryExpansionK < 1 {
		queryExpansionK = 1
	}
	if queryExpansionK > 5 {
		queryExpansionK = 5
	}
	queryExpansionRetrieveK := atoiOrDefault(os.Getenv("QUERY_EXPANSION_RETRIEVE_K"), 3)
	if queryExpansionRetrieveK < 1 {
		queryExpansionRetrieveK = 1
	}
	if queryExpansionRetrieveK > 5 {
		queryExpansionRetrieveK = 5
	}
	useKeyword := strings.ToLower(os.Getenv("USE_KEYWORD")) == "true"

	fusionMethod := strings.ToLower(os.Getenv("FUSION_METHOD"))
	if fusionMethod == "" {
		fusionMethod = "simple"
	}
	if fusionMethod != "simple" && fusionMethod != "score" && fusionMethod != "rrf" && fusionMethod != "borda" && fusionMethod != "combmnz" && fusionMethod != "combmnz_kw" && fusionMethod != "combmnz_rank_kw" {
		log.Fatalf("unsupported FUSION_METHOD: %q (use simple, score, rrf, borda, combmnz, combmnz_kw, or combmnz_rank_kw)", fusionMethod)
	}

	// P16-A: metadata-aware heading/section boost before reranking.
	// This keeps retrieval unchanged, then slightly increases fused scores when
	// question terms match section, breadcrumb, title, or heading metadata.
	usePreRerankHeadingBoost := strings.ToLower(strings.TrimSpace(os.Getenv("USE_PRERERANK_HEADING_BOOST"))) == "true"
	headingBoostStrong := atofOrDefault(os.Getenv("HEADING_BOOST_STRONG"), 0.18)
	headingBoostMedium := atofOrDefault(os.Getenv("HEADING_BOOST_MEDIUM"), 0.10)
	headingBoostWeak := atofOrDefault(os.Getenv("HEADING_BOOST_WEAK"), 0.05)
	if headingBoostStrong < 0 {
		headingBoostStrong = 0
	}
	if headingBoostMedium < 0 {
		headingBoostMedium = 0
	}
	if headingBoostWeak < 0 {
		headingBoostWeak = 0
	}

	// P17-B1/B2: exact section-match boost before reranking.
	// This keeps P16-A retrieval/reranking unchanged but gives a stronger score
	// boost to candidates whose section/title exactly contains a core question phrase
	// such as "storage hierarchy" or "link layer handoff".
	useP17BExactSectionBoost := strings.ToLower(strings.TrimSpace(os.Getenv("USE_P17B_EXACT_SECTION_BOOST"))) == "true"
	p17BExactSectionBoost := atofOrDefault(os.Getenv("P17B_EXACT_SECTION_BOOST"), 0.75)
	p17BBreadcrumbBoost := atofOrDefault(os.Getenv("P17B_BREADCRUMB_BOOST"), 0.35)
	if p17BExactSectionBoost < 0 {
		p17BExactSectionBoost = 0
	}
	if p17BBreadcrumbBoost < 0 {
		p17BBreadcrumbBoost = 0
	}

	// P17-B2: sibling-section demotion.
	// If one candidate directly matches a core question phrase, related sibling sections
	// that only partially overlap with that phrase are demoted. Example: for
	// "storage hierarchy", demote "Storage Models" and "PAX Storage Model".
	useP17BSiblingDemotion := strings.ToLower(strings.TrimSpace(os.Getenv("USE_P17B_SIBLING_DEMOTION"))) == "true"
	p17BSiblingDemotion := atofOrDefault(os.Getenv("P17B_SIBLING_DEMOTION"), 0.80)
	if p17BSiblingDemotion < 0 {
		p17BSiblingDemotion = 0
	}
	if p17BSiblingDemotion > 0.95 {
		p17BSiblingDemotion = 0.95
	}
	p17BSiblingMinOverlap := atofOrDefault(os.Getenv("P17B_SIBLING_MIN_OVERLAP"), 0.34)
	if p17BSiblingMinOverlap <= 0 || p17BSiblingMinOverlap > 1 {
		p17BSiblingMinOverlap = 0.34
	}

	// P16-B: final-context heading protection.
	// After reranking, exact/strong section-heading matches can be protected and moved
	// ahead of semantically related sibling sections before final C5.4 context selection.
	useFinalHeadingProtection := strings.ToLower(strings.TrimSpace(os.Getenv("USE_FINAL_HEADING_PROTECTION"))) == "true"
	finalHeadingMinPriority := atoiOrDefault(os.Getenv("FINAL_HEADING_MIN_PRIORITY"), 2)
	if finalHeadingMinPriority < 1 {
		finalHeadingMinPriority = 1
	}
	if finalHeadingMinPriority > 3 {
		finalHeadingMinPriority = 3
	}

	// P16-C: protected inclusion, not only reordering.
	// If an exact/strong heading candidate exists, force it into the final context before normal C5.4 selection.
	useFinalHeadingForceInclude := strings.ToLower(strings.TrimSpace(os.Getenv("USE_FINAL_HEADING_FORCE_INCLUDE"))) == "true"
	finalHeadingForcePriority := atoiOrDefault(os.Getenv("FINAL_HEADING_FORCE_PRIORITY"), 3)
	if finalHeadingForcePriority < 1 {
		finalHeadingForcePriority = 1
	}
	if finalHeadingForcePriority > 3 {
		finalHeadingForcePriority = 3
	}
	finalHeadingForceMax := atoiOrDefault(os.Getenv("FINAL_HEADING_FORCE_MAX"), 1)
	if finalHeadingForceMax < 1 {
		finalHeadingForceMax = 1
	}
	if finalHeadingForceMax > 3 {
		finalHeadingForceMax = 3
	}
	finalHeadingForceMinChars := atoiOrDefault(os.Getenv("FINAL_HEADING_FORCE_MIN_CHARS"), 80)
	if finalHeadingForceMinChars < 0 {
		finalHeadingForceMinChars = 0
	}

	// P17-E2: conservative unsupported-question validation.
	// Unlike P17-E, this does not gate normal conceptual questions.
	useP17E2UnsupportedValidation := strings.ToLower(strings.TrimSpace(os.Getenv("USE_P17E2_UNSUPPORTED_VALIDATION"))) == "true"

	// P20-A: conversational query resolution.
	// Enabled by default. Set USE_CONVERSATIONAL_RESOLUTION=false to reproduce
	// the previous standalone-only pipeline exactly.
	useConversationalResolution := strings.ToLower(strings.TrimSpace(os.Getenv("USE_CONVERSATIONAL_RESOLUTION"))) != "false"
	conversationHistoryMessages := atoiOrDefault(os.Getenv("CONVERSATION_HISTORY_MESSAGES"), 4)
	if conversationHistoryMessages < 1 {
		conversationHistoryMessages = 1
	}
	if conversationHistoryMessages > 8 {
		conversationHistoryMessages = 8
	}

	var allowedTags []string
	if raw := strings.TrimSpace(os.Getenv("ALLOWED_TAGS")); raw != "" {
		allowedTags = normalizeAllowedTags(strings.Split(raw, ","))
	}

	// P15-D: conservative metadata-aware automatic tag routing.
	// If ALLOWED_TAGS/input.Tag is empty, the QA flow first performs an
	// unfiltered vector retrieval and combines rank-weighted semantic tag votes
	// with database-generated tag profiles. Unlike P15-D, P15-D only forces
	// a single tag when confidence is high; otherwise it keeps top-2 or falls
	// back to all tags to avoid catastrophic wrong-tag filtering.
	autoTagRouting := strings.ToLower(strings.TrimSpace(os.Getenv("AUTO_TAG_ROUTING"))) == "true"
	autoTagMode := strings.ToLower(strings.TrimSpace(os.Getenv("AUTO_TAG_MODE")))
	if autoTagMode == "" {
		autoTagMode = "profile"
	}
	autoTagPreK := atoiOrDefault(os.Getenv("AUTO_TAG_PRE_K"), 12)
	if autoTagPreK < 1 {
		autoTagPreK = 12
	}
	autoTagMinDominant := atoiOrDefault(os.Getenv("AUTO_TAG_MIN_DOMINANT"), 4)
	if autoTagMinDominant < 1 {
		autoTagMinDominant = 4
	}
	autoTagTop2Min := atoiOrDefault(os.Getenv("AUTO_TAG_TOP2_MIN"), 6)
	if autoTagTop2Min < 1 {
		autoTagTop2Min = 6
	}
	autoTagMinScore := atofOrDefault(os.Getenv("AUTO_TAG_MIN_SCORE"), 3.0)
	if autoTagMinScore < 0 {
		autoTagMinScore = 0
	}
	autoTagSingleMargin := atofOrDefault(os.Getenv("AUTO_TAG_SINGLE_MARGIN"), 1.25)
	if autoTagSingleMargin < 1.0 {
		autoTagSingleMargin = 1.0
	}

	// P15-D conservative routing thresholds.
	// AUTO_TAG_HIGH_MARGIN controls when we trust a single tag. A value of 1.60
	// means the top tag must score at least 60% higher than the second tag.
	autoTagHighMargin := atofOrDefault(os.Getenv("AUTO_TAG_HIGH_MARGIN"), 1.60)
	if autoTagHighMargin < 1.0 {
		autoTagHighMargin = 1.60
	}

	// If the top tag is not dominant enough, P15-D may keep the top two tags,
	// but only when they jointly explain most routing evidence.
	autoTagTop2Share := atofOrDefault(os.Getenv("AUTO_TAG_TOP2_SHARE"), 0.70)
	if autoTagTop2Share < 0.0 || autoTagTop2Share > 1.0 {
		autoTagTop2Share = 0.70
	}

	autoTagProfileWeight := atofOrDefault(os.Getenv("AUTO_TAG_PROFILE_WEIGHT"), 1.5)
	if autoTagProfileWeight < 0 {
		autoTagProfileWeight = 0
	}
	autoTagMaxSelected := atoiOrDefault(os.Getenv("AUTO_TAG_MAX_SELECTED"), 2)
	if autoTagMaxSelected < 1 {
		autoTagMaxSelected = 1
	}
	autoTagProfileLimit := atoiOrDefault(os.Getenv("AUTO_TAG_PROFILE_LIMIT_PER_TAG"), 80)
	if autoTagProfileLimit < 1 {
		autoTagProfileLimit = 80
	}

	ctx := context.Background()

	ollamaAddr := os.Getenv("OLLAMA_HOST")
	if ollamaAddr == "" {
		ollamaAddr = "http://localhost:11434"
	}

	ollamaPlugin := &ollama.Ollama{
		ServerAddress: ollamaAddr,
		Timeout:       120,
	}

	// The university API supports both chat generation and embeddings.
	// Ollama is retained only for explicitly selected legacy embedding experiments.
	uniAPIBaseURL := strings.TrimRight(strings.TrimSpace(os.Getenv("UNI_API_BASE_URL")), "/")
	if uniAPIBaseURL == "" {
		uniAPIBaseURL = "https://dbhage.prakinf.tu-ilmenau.de/api/v1"
	}
	uniAPIKey := strings.TrimSpace(os.Getenv("UNI_API_KEY"))
	useUniversity := strings.HasPrefix(llmModel, "university/")
	embeddingProvider := strings.ToLower(strings.TrimSpace(os.Getenv("EMBEDDING_PROVIDER")))
	if embeddingProvider == "" {
		// Existing university embedding IDs default to the university endpoint.
		// Other models remain on Ollama unless EMBEDDING_PROVIDER is specified.
		if embeddingModel == "bge-m3" || embeddingModel == "nomic-embed-text-v1.5" {
			embeddingProvider = "university"
		} else {
			embeddingProvider = "ollama"
		}
	}
	if embeddingProvider != "university" && embeddingProvider != "ollama" {
		log.Fatalf("invalid EMBEDDING_PROVIDER %q (use university or ollama)", embeddingProvider)
	}
	if (useUniversity || embeddingProvider == "university") && uniAPIKey == "" {
		log.Fatal("UNI_API_KEY is required for university generation or embeddings")
	}
	if embeddingProvider == "university" {
		switch embeddingModel {
		case "bge-m3":
			if embeddingDim != 1024 {
				log.Fatal("bge-m3 requires EMBEDDING_DIM=1024")
			}
		case "nomic-embed-text-v1.5":
			if embeddingDim != 768 {
				log.Fatal("nomic-embed-text-v1.5 requires EMBEDDING_DIM=768")
			}
		}
	}

	plugins := []api.Plugin{
		ollamaPlugin,
		&evaluators.GenkitEval{Metrics: []evaluators.MetricConfig{
			{MetricType: evaluators.EvaluatorRegex},
		}},
	}
	if strings.HasPrefix(llmModel, "googleai/") {
		plugins = append(plugins, &googlegenai.GoogleAI{})
	}
	var universityPlugin *compat_oai.OpenAICompatible
	if useUniversity || embeddingProvider == "university" {
		universityPlugin = &compat_oai.OpenAICompatible{
			Provider: "university",
			APIKey:   uniAPIKey,
			BaseURL:  uniAPIBaseURL,
		}
		plugins = append(plugins, universityPlugin)
	}
	genk := genkit.Init(ctx,
		genkit.WithPromptDir("./prompts"),
		genkit.WithPlugins(plugins...),
	)

	var llm ai.Model
	if strings.HasPrefix(llmModel, "googleai/") {
		modelName := strings.TrimPrefix(llmModel, "googleai/")
		llm = googlegenai.GoogleAIModel(genk, modelName)
	} else if strings.HasPrefix(llmModel, "ollama/") {
		modelName := strings.TrimPrefix(llmModel, "ollama/")
		if ollamaPlugin.DefineModel(genk, ollama.ModelDefinition{
			Name: modelName,
			Type: "chat",
		}, nil) == nil {
			log.Fatalf("failed to define ollama model %v", modelName)
		}
		llm = ollama.Model(genk, modelName)
	} else if strings.HasPrefix(llmModel, "university/") {
		llm = genkit.LookupModel(genk, llmModel)

		if llm == nil {
			log.Fatalf("failed to resolve university model: %s", llmModel)
		}
	} else {
		log.Fatalf("unsupported LLM_MODEL provider: %q (use university/, googleai/ or ollama/ prefix)", llmModel)
	}

	// P20-A conversational resolution supports native Ollama or university chat completions.
	conversationModel := ""
	if strings.HasPrefix(llmModel, "ollama/") {
		conversationModel = strings.TrimPrefix(llmModel, "ollama/")
	} else if useUniversity {
		conversationModel = strings.TrimPrefix(llmModel, "university/")
	}

	phaseName := strings.TrimSpace(os.Getenv("EXPERIMENT_PHASE"))
	if phaseName == "" {
		if useQueryExpansion && useReranker && fusionMethod == "rrf" && rerankMinScore >= 3 {
			phaseName = "P4F question expansion + strict rerank filtering"
		} else if useHyDE && useQueryExpansion {
			phaseName = "P4E HyDE + question expansion"
		} else if useHyDE {
			phaseName = "P4A HyDE"
		} else if useQueryExpansion {
			phaseName = "P4C/P4D question expansion"
		} else {
			phaseName = "P4B no-HyDE baseline"
		}
	}

	log.Printf("PHASE: %s", phaseName)
	log.Printf("LLM: %s", llmModel)
	log.Printf("EMBEDDING_MODEL: %s", embeddingModel)
	log.Printf("EMBEDDING_PROVIDER: %s", embeddingProvider)
	log.Printf("EMBEDDING_DIM: %d", embeddingDim)
	log.Printf("USE_RERANKER: %v", useReranker)
	log.Printf("USE_HYDE: %v", useHyDE)
	log.Printf("USE_QUERY_EXPANSION: %v", useQueryExpansion)
	log.Printf("QUERY_EXPANSION_K: %d", queryExpansionK)
	log.Printf("QUERY_EXPANSION_RETRIEVE_K: %d", queryExpansionRetrieveK)
	log.Printf("USE_KEYWORD: %v", useKeyword)
	log.Printf("FUSION_METHOD: %s", fusionMethod)
	log.Printf("USE_PRERERANK_HEADING_BOOST: %v", usePreRerankHeadingBoost)
	log.Printf("HEADING_BOOST_STRONG: %.2f", headingBoostStrong)
	log.Printf("HEADING_BOOST_MEDIUM: %.2f", headingBoostMedium)
	log.Printf("HEADING_BOOST_WEAK: %.2f", headingBoostWeak)
	log.Printf("USE_P17B_EXACT_SECTION_BOOST: %v", useP17BExactSectionBoost)
	log.Printf("P17B_EXACT_SECTION_BOOST: %.2f", p17BExactSectionBoost)
	log.Printf("P17B_BREADCRUMB_BOOST: %.2f", p17BBreadcrumbBoost)
	log.Printf("USE_P17B_SIBLING_DEMOTION: %v", useP17BSiblingDemotion)
	log.Printf("P17B_SIBLING_DEMOTION: %.2f", p17BSiblingDemotion)
	log.Printf("P17B_SIBLING_MIN_OVERLAP: %.2f", p17BSiblingMinOverlap)
	log.Printf("USE_FINAL_HEADING_PROTECTION: %v", useFinalHeadingProtection)
	log.Printf("FINAL_HEADING_MIN_PRIORITY: %d", finalHeadingMinPriority)
	log.Printf("USE_FINAL_HEADING_FORCE_INCLUDE: %v", useFinalHeadingForceInclude)
	log.Printf("FINAL_HEADING_FORCE_PRIORITY: %d", finalHeadingForcePriority)
	log.Printf("FINAL_HEADING_FORCE_MAX: %d", finalHeadingForceMax)
	log.Printf("FINAL_HEADING_FORCE_MIN_CHARS: %d", finalHeadingForceMinChars)
	log.Printf("USE_P17E2_UNSUPPORTED_VALIDATION: %v", useP17E2UnsupportedValidation)
	log.Printf("USE_CONVERSATIONAL_RESOLUTION: %v", useConversationalResolution)
	log.Printf("CONVERSATION_HISTORY_MESSAGES: %d", conversationHistoryMessages)
	log.Printf("ALLOWED_TAGS: %v", allowedTags)
	log.Printf("AUTO_TAG_ROUTING: %v", autoTagRouting)
	log.Printf("AUTO_TAG_MODE: %s", autoTagMode)
	log.Printf("AUTO_TAG_PRE_K: %d", autoTagPreK)
	log.Printf("AUTO_TAG_MIN_DOMINANT: %d", autoTagMinDominant)
	log.Printf("AUTO_TAG_TOP2_MIN: %d", autoTagTop2Min)
	log.Printf("AUTO_TAG_MIN_SCORE: %.2f", autoTagMinScore)
	log.Printf("AUTO_TAG_SINGLE_MARGIN: %.2f", autoTagSingleMargin)
	log.Printf("AUTO_TAG_HIGH_MARGIN: %.2f", autoTagHighMargin)
	log.Printf("AUTO_TAG_TOP2_SHARE: %.2f", autoTagTop2Share)
	log.Printf("AUTO_TAG_PROFILE_WEIGHT: %.2f", autoTagProfileWeight)
	log.Printf("AUTO_TAG_MAX_SELECTED: %d", autoTagMaxSelected)
	log.Printf("AUTO_TAG_PROFILE_LIMIT_PER_TAG: %d", autoTagProfileLimit)
	log.Printf("FINAL_CONTEXT_K: %d", finalContextK)
	log.Printf("RERANK_MIN_SCORE: %d", rerankMinScore)
	log.Printf("MAX_CONTEXT_CHARS: %d", maxContextChars)

	var emb ai.Embedder
	if embeddingProvider == "university" {
		// The compat_oai API returns an ai.Embedder usable directly by pgstore.go.
		emb = universityPlugin.DefineEmbedder("university", embeddingModel, nil)
	} else {
		emb = ollamaPlugin.DefineEmbedder(genk, ollamaAddr, embeddingModel, nil)
	}
	if emb == nil {
		log.Fatalf("failed to define %s embedder %q", embeddingProvider, embeddingModel)
	}

	registerEvaluators(genk, llmModel)

	store, err := newVectorStore(genk, emb, embeddingDim)
	if err != nil {
		log.Fatal(err)
	}

	hydePrompt := genkit.LookupPrompt(genk, "hyde")
	if hydePrompt == nil {
		log.Fatal("prompt 'hyde' not found in ./prompts/")
	}
	rerankPrompt := genkit.LookupPrompt(genk, "rerank")
	if rerankPrompt == nil {
		log.Fatal("prompt 'rerank' not found in ./prompts/")
	}
	qaPrompt := genkit.LookupPrompt(genk, "qa")
	if qaPrompt == nil {
		log.Fatal("prompt 'qa' not found in ./prompts/")
	}

	return &appSetup{
		ctx:                           ctx,
		genk:                          genk,
		llm:                           llm,
		chunkSize:                     chunkSize,
		chunkOverlap:                  chunkOverlap,
		useReranker:                   useReranker,
		useHyDE:                       useHyDE,
		useQueryExpansion:             useQueryExpansion,
		queryExpansionK:               queryExpansionK,
		queryExpansionRetrieveK:       queryExpansionRetrieveK,
		useKeyword:                    useKeyword,
		fusionMethod:                  fusionMethod,
		usePreRerankHeadingBoost:      usePreRerankHeadingBoost,
		headingBoostStrong:            headingBoostStrong,
		headingBoostMedium:            headingBoostMedium,
		headingBoostWeak:              headingBoostWeak,
		useP17BExactSectionBoost:      useP17BExactSectionBoost,
		p17BExactSectionBoost:         p17BExactSectionBoost,
		p17BBreadcrumbBoost:           p17BBreadcrumbBoost,
		useP17BSiblingDemotion:        useP17BSiblingDemotion,
		p17BSiblingDemotion:           p17BSiblingDemotion,
		p17BSiblingMinOverlap:         p17BSiblingMinOverlap,
		useFinalHeadingProtection:     useFinalHeadingProtection,
		finalHeadingMinPriority:       finalHeadingMinPriority,
		useFinalHeadingForceInclude:   useFinalHeadingForceInclude,
		finalHeadingForcePriority:     finalHeadingForcePriority,
		finalHeadingForceMax:          finalHeadingForceMax,
		finalHeadingForceMinChars:     finalHeadingForceMinChars,
		useP17E2UnsupportedValidation: useP17E2UnsupportedValidation,
		allowedTags:                   allowedTags,
		autoTagRouting:                autoTagRouting,
		autoTagMode:                   autoTagMode,
		autoTagPreK:                   autoTagPreK,
		autoTagMinDominant:            autoTagMinDominant,
		autoTagTop2Min:                autoTagTop2Min,
		autoTagMinScore:               autoTagMinScore,
		autoTagSingleMargin:           autoTagSingleMargin,
		autoTagHighMargin:             autoTagHighMargin,
		autoTagTop2Share:              autoTagTop2Share,
		autoTagProfileWeight:          autoTagProfileWeight,
		autoTagMaxSelected:            autoTagMaxSelected,
		autoTagProfileLimit:           autoTagProfileLimit,
		finalContextK:                 finalContextK,
		rerankMinScore:                rerankMinScore,
		maxContextChars:               maxContextChars,
		store:                         store,
		hydePrompt:                    hydePrompt,
		rerankPrompt:                  rerankPrompt,
		qaPrompt:                      qaPrompt,
		useConversationalResolution:   useConversationalResolution,
		ollamaAddr:                    ollamaAddr,
		uniAPIBaseURL:                 uniAPIBaseURL,
		uniAPIKey:                     uniAPIKey,
		conversationModel:             conversationModel,
		conversationHistoryMessages:   conversationHistoryMessages,
	}
}

func (app *appSetup) defineIndexPDFFlow() *core.Flow[indexInput, any, struct{}] {
	return genkit.DefineFlow(
		app.genk, "indexPDFDocument",
		func(ctx context.Context, input indexInput) (any, error) {
			pages, err := genkit.Run(ctx, "extract", func() ([]pageContent, error) {
				return readPDFPages(input.Path)
			})
			if err != nil {
				return nil, err
			}

			docs, err := genkit.Run(ctx, "chunk", func() ([]*ai.Document, error) {
				paras := parseIntoParagraphs(pages)
				chunks := buildChunks(paras, app.chunkSize, app.chunkOverlap)

				var docs []*ai.Document
				for _, c := range chunks {
					meta := map[string]any{
						"pages":      c.pages,
						"parentText": c.parentText,
						"embedText":  c.embedText,
					}

					if c.section != "" {
						meta["section"] = c.section
					}
					if c.breadcrumb != "" {
						meta["breadcrumb"] = c.breadcrumb
					}
					if input.Tag != "" {
						meta["tag"] = strings.ToLower(strings.TrimSpace(input.Tag))
					}
					if input.Datum != "" {
						meta["datum"] = input.Datum
					}
					if input.Filename != "" {
						meta["filename"] = input.Filename
					}

					docs = append(docs, ai.DocumentFromText(c.text, meta))
				}
				return docs, nil
			})
			if err != nil {
				return nil, err
			}

			if err := app.store.Index(ctx, docs); err != nil {
				return nil, err
			}

			return map[string]interface{}{
				"success":          true,
				"documentsIndexed": len(docs),
			}, nil
		},
	)
}

func (app *appSetup) defineIndexWebFlow() *core.Flow[indexURLInput, any, struct{}] {
	return genkit.DefineFlow(
		app.genk, "indexWebDocument",
		func(ctx context.Context, input indexURLInput) (any, error) {
			htmlContent, err := genkit.Run(ctx, "fetch", func() (string, error) {
				return fetchWebPage(input.URL)
			})
			if err != nil {
				return nil, err
			}

			docs, err := genkit.Run(ctx, "chunk", func() ([]*ai.Document, error) {
				paras, err := htmlToStructuredText(strings.NewReader(htmlContent))
				if err != nil {
					return nil, err
				}

				chunks := buildWebChunks(paras, app.chunkSize, app.chunkOverlap)

				var docs []*ai.Document
				for _, c := range chunks {
					meta := map[string]any{
						"parentText": c.parentText,
						"embedText":  c.embedText,
						"url":        input.URL,
					}

					if c.section != "" {
						meta["section"] = c.section
					}
					if c.breadcrumb != "" {
						meta["breadcrumb"] = c.breadcrumb
					}
					if input.Tag != "" {
						meta["tag"] = strings.ToLower(strings.TrimSpace(input.Tag))
					}
					if input.Datum != "" {
						meta["datum"] = input.Datum
					}

					docs = append(docs, ai.DocumentFromText(c.text, meta))
				}
				return docs, nil
			})
			if err != nil {
				return nil, err
			}

			if err := app.store.Index(ctx, docs); err != nil {
				return nil, err
			}

			return map[string]interface{}{
				"success":          true,
				"documentsIndexed": len(docs),
				"url":              input.URL,
			}, nil
		},
	)
}

func (app *appSetup) defineCrawlWebFlow() *core.Flow[crawlInput, crawlResult, struct{}] {
	return genkit.DefineFlow(
		app.genk, "crawlWeb",
		func(ctx context.Context, input crawlInput) (crawlResult, error) {
			indexPage := buildAndIndexPage(app.chunkSize, app.chunkOverlap, app.store, strings.ToLower(strings.TrimSpace(input.Tag)), input.Datum)
			return runCrawl(ctx, input, "./crawl-data", indexPage)
		},
	)
}

type fusedDocEntry struct {
	doc        *ai.Document
	text       string
	dedupKey   string
	sourceKey  string
	score      float64
	bestRank   int
	firstOrder int
	sources    map[string]bool
}

func mergeNormalizeText(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))

	replacer := strings.NewReplacer(
		"\u00a0", " ",
		"\u200b", "",
		"\u200c", "",
		"\u200d", "",
		"\ufeff", "",
		"\t", " ",
		"\r", " ",
		"\n", " ",
	)
	s = replacer.Replace(s)

	var b strings.Builder
	lastWasSpace := false

	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			lastWasSpace = false
			continue
		}

		if !lastWasSpace {
			b.WriteRune(' ')
			lastWasSpace = true
		}
	}

	return strings.Join(strings.Fields(b.String()), " ")
}

func mergeTokenSet(s string) map[string]bool {
	tokens := make(map[string]bool)
	for _, t := range strings.Fields(s) {
		if len(t) >= 3 {
			tokens[t] = true
		}
	}
	return tokens
}

func mergeJaccardSimilarity(a, b string) float64 {
	ta := mergeTokenSet(a)
	tb := mergeTokenSet(b)

	if len(ta) == 0 || len(tb) == 0 {
		return 0
	}

	intersection := 0
	for t := range ta {
		if tb[t] {
			intersection++
		}
	}

	union := len(ta) + len(tb) - intersection
	if union == 0 {
		return 0
	}

	return float64(intersection) / float64(union)
}

func mergeContainmentSimilarity(a, b string) float64 {
	ta := mergeTokenSet(a)
	tb := mergeTokenSet(b)

	if len(ta) == 0 || len(tb) == 0 {
		return 0
	}

	intersection := 0
	for t := range ta {
		if tb[t] {
			intersection++
		}
	}

	smaller := len(ta)
	if len(tb) < smaller {
		smaller = len(tb)
	}
	if smaller == 0 {
		return 0
	}

	return float64(intersection) / float64(smaller)
}

func mergeSourceKey(doc *ai.Document) string {
	pages := strings.TrimSpace(docMetadataString(doc, "pages"))
	section := strings.TrimSpace(docMetadataString(doc, "section"))
	breadcrumb := strings.TrimSpace(docMetadataString(doc, "breadcrumb"))
	url := strings.TrimSpace(docMetadataString(doc, "url"))
	tag := strings.TrimSpace(docMetadataString(doc, "tag"))

	return strings.ToLower(strings.Join([]string{
		tag,
		url,
		pages,
		section,
		breadcrumb,
	}, "|"))
}

func makeMergeDedupKey(doc *ai.Document, txt string) string {
	normalized := mergeNormalizeText(txt)

	prefix := normalized
	if len(prefix) > 700 {
		prefix = prefix[:700]
	}

	return mergeSourceKey(doc) + "|" + prefix
}

func findNearDuplicateMergeEntry(entries []*fusedDocEntry, docMap map[string]*fusedDocEntry, doc *ai.Document, txt string, dedupKey string, sourceKey string) *fusedDocEntry {
	if existing, exists := docMap[dedupKey]; exists {
		return existing
	}

	normalizedText := mergeNormalizeText(txt)

	for _, existing := range entries {
		if existing == nil {
			continue
		}

		sameSourceLocation := sourceKey != "" && existing.sourceKey == sourceKey

		existingNormalized := mergeNormalizeText(existing.text)
		jaccard := mergeJaccardSimilarity(normalizedText, existingNormalized)
		containment := mergeContainmentSimilarity(normalizedText, existingNormalized)

		if sameSourceLocation && (jaccard >= 0.72 || containment >= 0.82) {
			return existing
		}

		if jaccard >= 0.88 || containment >= 0.94 {
			return existing
		}
	}

	return nil
}

func mergeRetrievedDocs(app *appSetup, resp1 *ai.RetrieverResponse, resp2 *ai.RetrieverResponse, kwDocs []*ai.Document, allowedTags []string, secondaryLabel string) ([]fusedDocEntry, int, int) {
	if strings.TrimSpace(secondaryLabel) == "" {
		secondaryLabel = "vector-secondary"
	}
	docMap := make(map[string]*fusedDocEntry)
	var docEntries []*fusedDocEntry
	orderCounter := 0

	registerOrMergeEntry := func(doc *ai.Document, score float64, rank int, sourceName string) {
		txt := extractDocumentText(doc)
		if strings.TrimSpace(txt) == "" {
			return
		}

		sourceKey := mergeSourceKey(doc)
		dedupKey := makeMergeDedupKey(doc, txt)

		if existing := findNearDuplicateMergeEntry(docEntries, docMap, doc, txt, dedupKey, sourceKey); existing != nil {
			existing.score += score
			existing.sources[sourceName] = true

			if rank < existing.bestRank {
				existing.bestRank = rank
				existing.doc = doc
				existing.text = txt
				existing.dedupKey = dedupKey
				existing.sourceKey = sourceKey
			}

			debugf(
				"[mergeDedup] source=%s rank=%d addedScore=%.6f accumulated=%.6f pages=%q section=%q text=%q",
				sourceName,
				rank,
				score,
				existing.score,
				docMetadataString(doc, "pages"),
				docMetadataString(doc, "section"),
				previewText(txt, 100),
			)

			return
		}

		orderCounter++

		entry := &fusedDocEntry{
			doc:        doc,
			text:       txt,
			dedupKey:   dedupKey,
			sourceKey:  sourceKey,
			score:      score,
			bestRank:   rank,
			firstOrder: orderCounter,
			sources:    map[string]bool{sourceName: true},
		}

		docMap[dedupKey] = entry
		docEntries = append(docEntries, entry)
	}

	addSimple := func(docs []*ai.Document, sourceName string) {
		for rank, doc := range docs {
			score := 1.0 / float64(rank+1)
			registerOrMergeEntry(doc, score, rank+1, sourceName)
		}
	}

	addScored := func(docs []*ai.Document, weight float64, sourceName string) {
		for rank, doc := range docs {
			baseScore := 1.0 / float64(rank+1)

			if doc.Metadata != nil {
				if v, ok := doc.Metadata["retrievalScore"].(float64); ok {
					baseScore = v
				}
			}

			score := weight * baseScore
			registerOrMergeEntry(doc, score, rank+1, sourceName)
		}
	}

	addRRF := func(docs []*ai.Document, weight float64, sourceName string) {
		const rrfK = 30.0

		for rank, doc := range docs {
			txt := extractDocumentText(doc)
			if strings.TrimSpace(txt) == "" {
				continue
			}

			score := weight * (1.0 / (rrfK + float64(rank+1)))

			beforeCount := len(docEntries)
			registerOrMergeEntry(doc, score, rank+1, sourceName)

			if len(docEntries) == beforeCount {
				debugf(
					"[rrfAdd:dedup] source=%s rank=%d partial=%.6f pages=%q section=%q text=%q",
					sourceName,
					rank+1,
					score,
					docMetadataString(doc, "pages"),
					docMetadataString(doc, "section"),
					previewText(txt, 100),
				)
			} else {
				debugf(
					"[rrfAdd:new] source=%s rank=%d partial=%.6f pages=%q section=%q text=%q",
					sourceName,
					rank+1,
					score,
					docMetadataString(doc, "pages"),
					docMetadataString(doc, "section"),
					previewText(txt, 100),
				)
			}
		}
	}

	// P11-S3: Normalized Borda fusion.
	// This is rank-based like RRF, but sharper for short retrieval lists.
	// Each retriever contributes scores in the same 0..1 range, so keyword top-3
	// is not unfairly penalized compared with vector/HyDE top-5.
	addBorda := func(docs []*ai.Document, weight float64, sourceName string) {
		listLen := len(docs)
		if listLen == 0 {
			return
		}

		for rank, doc := range docs {
			txt := extractDocumentText(doc)
			if strings.TrimSpace(txt) == "" {
				continue
			}

			// rank is zero-based. Rank 1 receives 1.00. Last item receives 1/listLen.
			score := weight * (float64(listLen-rank) / float64(listLen))

			beforeCount := len(docEntries)
			registerOrMergeEntry(doc, score, rank+1, sourceName)

			if len(docEntries) == beforeCount {
				debugf(
					"[bordaAdd:dedup] source=%s rank=%d listLen=%d partial=%.6f pages=%q section=%q text=%q",
					sourceName,
					rank+1,
					listLen,
					score,
					docMetadataString(doc, "pages"),
					docMetadataString(doc, "section"),
					previewText(txt, 100),
				)
			} else {
				debugf(
					"[bordaAdd:new] source=%s rank=%d listLen=%d partial=%.6f pages=%q section=%q text=%q",
					sourceName,
					rank+1,
					listLen,
					score,
					docMetadataString(doc, "pages"),
					docMetadataString(doc, "section"),
					previewText(txt, 100),
				)
			}
		}
	}

	// P11-S4/S5/S6: CombMNZ-family fusion.
	//
	// S4/S5 used score-normalized CombMNZ. S6 repairs that by using
	// rank-normalized CombMNZ with keyword protection. This avoids min-max
	// exaggeration when vector/HyDE scores are tightly clustered.
	//
	// For S6, use:
	//   FUSION_METHOD=combmnz_rank_kw
	//
	// Rank-normalized scoring:
	//   rank 1 in top-5 = 1.00
	//   rank 2 in top-5 = 0.80
	//   rank 5 in top-5 = 0.20
	//   rank 1 in top-3 = 1.00
	//   rank 3 in top-3 = 0.33
	//
	// CombMNZ:
	//   CombSUM = sum of normalized source scores
	//   CombMNZ = CombSUM * number of retrievers that returned the candidate
	addCombMNZ := func(docs []*ai.Document, weight float64, sourceName string) {
		listLen := len(docs)
		if listLen == 0 {
			return
		}

		// Defensive check for S4/S5 raw-score modes. We only use raw-score
		// normalization if every document in this source list has a usable raw score.
		// Otherwise the whole source falls back to rank normalization. This prevents
		// mixed normalization where some candidates use real scores and others use
		// rank fallback values.
		allHaveRawScore := true
		rawScores := make([]float64, listLen)
		minScore := 0.0
		maxScore := 0.0

		for i, doc := range docs {
			if doc == nil || doc.Metadata == nil {
				allHaveRawScore = false
				break
			}

			raw, ok := doc.Metadata["retrievalScore"].(float64)
			if !ok {
				raw, ok = doc.Metadata["score"].(float64)
			}
			if !ok {
				allHaveRawScore = false
				break
			}

			rawScores[i] = raw
			if i == 0 || raw < minScore {
				minScore = raw
			}
			if i == 0 || raw > maxScore {
				maxScore = raw
			}
		}

		for rank, doc := range docs {
			txt := extractDocumentText(doc)
			if strings.TrimSpace(txt) == "" {
				continue
			}

			rankNormScore := float64(listLen-rank) / float64(listLen)
			normScore := rankNormScore
			normMode := "rank"

			if app.fusionMethod == "combmnz" || app.fusionMethod == "combmnz_kw" {
				if allHaveRawScore && maxScore > minScore {
					// S4/S5 legacy mode: min-max normalized raw scores.
					// Kept for reproducibility of previous experiments.
					normScore = (rawScores[rank] - minScore) / (maxScore - minScore)
					normMode = "minmax"
				} else {
					normScore = rankNormScore
					normMode = "rank_fallback"
				}

				// Avoid completely losing the last candidate when min-max normalization
				// maps it to zero. This is useful with very short lists such as keyword top-3.
				if normScore < 0.05 {
					normScore = 0.05
				}
			}

			// S6 mode always uses rank-normalized scores for all retrievers.
			// This avoids comparing incompatible raw score distributions and prevents
			// min-max from over-amplifying tiny cosine-score differences.
			if app.fusionMethod == "combmnz_rank_kw" {
				normScore = rankNormScore
				normMode = "rank_only"
			}

			score := weight * normScore
			beforeCount := len(docEntries)
			registerOrMergeEntry(doc, score, rank+1, sourceName)

			if len(docEntries) == beforeCount {
				debugf(
					"[combmnzAdd:dedup] mode=%s source=%s rank=%d listLen=%d norm=%.6f partial=%.6f pages=%q section=%q text=%q",
					normMode,
					sourceName,
					rank+1,
					listLen,
					normScore,
					score,
					docMetadataString(doc, "pages"),
					docMetadataString(doc, "section"),
					previewText(txt, 100),
				)
			} else {
				debugf(
					"[combmnzAdd:new] mode=%s source=%s rank=%d listLen=%d norm=%.6f partial=%.6f pages=%q section=%q text=%q",
					normMode,
					sourceName,
					rank+1,
					listLen,
					normScore,
					score,
					docMetadataString(doc, "pages"),
					docMetadataString(doc, "section"),
					previewText(txt, 100),
				)
			}
		}
	}

	nKw := len(kwDocs)
	n2 := 0
	if resp2 != nil {
		n2 = len(resp2.Documents)
	}

	const equalFusionWeight = 1.0

	switch app.fusionMethod {
	case "simple":
		if len(kwDocs) > 0 {
			addSimple(kwDocs, "keyword")
		}
		if resp1 != nil {
			addSimple(resp1.Documents, "vector-question")
		}
		if resp2 != nil {
			addSimple(resp2.Documents, secondaryLabel)
		}

	case "score":
		if resp1 != nil {
			addScored(resp1.Documents, equalFusionWeight, "vector-question")
		}
		if resp2 != nil {
			addScored(resp2.Documents, equalFusionWeight, secondaryLabel)
		}
		if len(kwDocs) > 0 {
			addScored(kwDocs, equalFusionWeight, "keyword")
		}

	case "rrf":
		if resp1 != nil {
			addRRF(resp1.Documents, equalFusionWeight, "vector-question")
		}
		if resp2 != nil {
			addRRF(resp2.Documents, equalFusionWeight, secondaryLabel)
		}
		if len(kwDocs) > 0 {
			addRRF(kwDocs, equalFusionWeight, "keyword")
		}

	case "borda":
		if resp1 != nil {
			addBorda(resp1.Documents, equalFusionWeight, "vector-question")
		}
		if resp2 != nil {
			addBorda(resp2.Documents, equalFusionWeight, secondaryLabel)
		}
		if len(kwDocs) > 0 {
			addBorda(kwDocs, equalFusionWeight, "keyword")
		}

	case "combmnz", "combmnz_kw", "combmnz_rank_kw":
		if resp1 != nil {
			addCombMNZ(resp1.Documents, equalFusionWeight, "vector-question")
		}
		if resp2 != nil {
			addCombMNZ(resp2.Documents, equalFusionWeight, secondaryLabel)
		}
		if len(kwDocs) > 0 {
			addCombMNZ(kwDocs, equalFusionWeight, "keyword")
		}

	default:
		debugf("[docQA] unknown fusion method %q, falling back to simple", app.fusionMethod)
		if len(kwDocs) > 0 {
			addSimple(kwDocs, "keyword")
		}
		if resp1 != nil {
			addSimple(resp1.Documents, "vector-question")
		}
		if resp2 != nil {
			addSimple(resp2.Documents, secondaryLabel)
		}
	}

	merged := make([]fusedDocEntry, 0, len(docEntries))
	for _, entry := range docEntries {
		if entry != nil {
			if app.fusionMethod == "combmnz" || app.fusionMethod == "combmnz_kw" || app.fusionMethod == "combmnz_rank_kw" {
				combSum := entry.score
				entry.score = combSum * float64(len(entry.sources))
			}
			merged = append(merged, *entry)
		}
	}

	sortFusedDocEntries(merged)

	if app.fusionMethod == "rrf" || app.fusionMethod == "borda" || app.fusionMethod == "combmnz" || app.fusionMethod == "combmnz_kw" || app.fusionMethod == "combmnz_rank_kw" {
		for i, entry := range merged {
			debugf(
				"[fusionFinal:%s] rank=%d score=%.6f sourceCount=%d bestRank=%d pages=%q section=%q breadcrumb=%q tag=%q text=%q",
				app.fusionMethod,
				i+1,
				entry.score,
				len(entry.sources),
				entry.bestRank,
				docMetadataString(entry.doc, "pages"),
				docMetadataString(entry.doc, "section"),
				docMetadataString(entry.doc, "breadcrumb"),
				docMetadataString(entry.doc, "tag"),
				previewText(entry.text, 120),
			)
		}
	}

	if len(allowedTags) > 0 {
		filtered := make([]fusedDocEntry, 0, len(merged))
		for _, entry := range merged {
			if len(FilterByTag([]*ai.Document{entry.doc}, allowedTags)) > 0 {
				filtered = append(filtered, entry)
			}
		}
		if len(filtered) != len(merged) {
			debugf("[docQA] post-fusion tag filter removed %d chunks", len(merged)-len(filtered))
		}
		merged = filtered
	}

	const activeMergeCap = 6

	// P11-S5/S6: keyword-protected CombMNZ.
	// Keep the CombMNZ ranking, but ensure at least two keyword-supported
	// candidates survive into the pre-reranker pool when such candidates exist.
	// S6 uses rank-normalized source scores before CombMNZ. This directly targets
	// acronym / exact-terminology failures such as HA, CoA, CN, and other
	// slide-heading or formula lookup questions.
	if len(merged) > activeMergeCap {
		if app.fusionMethod == "combmnz_kw" || app.fusionMethod == "combmnz_rank_kw" {
			selected := append([]fusedDocEntry{}, merged[:activeMergeCap]...)

			keywordCount := 0
			for _, entry := range selected {
				if entry.sources["keyword"] {
					keywordCount++
				}
			}

			const minKeywordProtected = 2
			if keywordCount < minKeywordProtected {
				for _, candidate := range merged[activeMergeCap:] {
					if keywordCount >= minKeywordProtected {
						break
					}
					if !candidate.sources["keyword"] {
						continue
					}

					// Replace the weakest selected non-keyword-only candidate from the bottom.
					replaceIdx := -1
					for i := len(selected) - 1; i >= 0; i-- {
						if !selected[i].sources["keyword"] {
							replaceIdx = i
							break
						}
					}

					if replaceIdx == -1 {
						break
					}

					debugf(
						"[combmnz_kw:protect] replacing rank=%d score=%.6f pages=%q section=%q with keyword score=%.6f pages=%q section=%q",
						replaceIdx+1,
						selected[replaceIdx].score,
						docMetadataString(selected[replaceIdx].doc, "pages"),
						docMetadataString(selected[replaceIdx].doc, "section"),
						candidate.score,
						docMetadataString(candidate.doc, "pages"),
						docMetadataString(candidate.doc, "section"),
					)

					selected[replaceIdx] = candidate
					keywordCount++
				}
			}

			// Re-sort selected candidates after protection so the reranker prompt receives
			// them in a stable, score-consistent order.
			sort.SliceStable(selected, func(i, j int) bool {
				if selected[i].score != selected[j].score {
					return selected[i].score > selected[j].score
				}
				if len(selected[i].sources) != len(selected[j].sources) {
					return len(selected[i].sources) > len(selected[j].sources)
				}
				if selected[i].bestRank != selected[j].bestRank {
					return selected[i].bestRank < selected[j].bestRank
				}
				if selected[i].firstOrder != selected[j].firstOrder {
					return selected[i].firstOrder < selected[j].firstOrder
				}
				return selected[i].dedupKey < selected[j].dedupKey
			})

			merged = selected
		} else {
			merged = merged[:activeMergeCap]
		}
	}

	return merged, n2, nKw
}

type autoTagCount struct {
	tag   string
	count int
}

type autoTagScore struct {
	tag   string
	score float64
	count int
}

type tagProfileProvider interface {
	TagProfiles(ctx context.Context, limitPerTag int) (map[string]string, error)
}

func sortedAutoTagCounts(tagCounts map[string]int) []autoTagCount {
	ranked := make([]autoTagCount, 0, len(tagCounts))
	for tag, count := range tagCounts {
		if strings.TrimSpace(tag) == "" || count <= 0 {
			continue
		}
		ranked = append(ranked, autoTagCount{tag: tag, count: count})
	}

	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].count != ranked[j].count {
			return ranked[i].count > ranked[j].count
		}
		return ranked[i].tag < ranked[j].tag
	})

	return ranked
}

func sortedAutoTagScores(tagScores map[string]float64, tagCounts map[string]int) []autoTagScore {
	ranked := make([]autoTagScore, 0, len(tagScores))
	for tag, score := range tagScores {
		if strings.TrimSpace(tag) == "" || score <= 0 {
			continue
		}
		ranked = append(ranked, autoTagScore{tag: tag, score: score, count: tagCounts[tag]})
	}

	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].score != ranked[j].score {
			return ranked[i].score > ranked[j].score
		}
		if ranked[i].count != ranked[j].count {
			return ranked[i].count > ranked[j].count
		}
		return ranked[i].tag < ranked[j].tag
	})

	return ranked
}

func autoTagTokens(text string) []string {
	raw := questionTokenRE.FindAllString(text, -1)
	seen := map[string]bool{}
	out := make([]string, 0, len(raw))

	for _, tok := range raw {
		trimmed := strings.TrimSpace(tok)
		lower := strings.ToLower(trimmed)
		if lower == "" || contextStopWords[lower] || seen[lower] {
			continue
		}

		// Keep normal technical terms with >=3 characters and also short
		// acronym/number tokens such as HA, IP, CoA, 3D, 802, 11.
		if len(lower) < 3 {
			hasUpper := false
			hasDigit := false
			for _, r := range trimmed {
				if unicode.IsUpper(r) {
					hasUpper = true
				}
				if unicode.IsDigit(r) {
					hasDigit = true
				}
			}
			if !hasUpper && !hasDigit {
				continue
			}
		}

		seen[lower] = true
		out = append(out, lower)
	}

	return out
}

func normalizeProfileText(text string) string {
	tokens := questionTokenRE.FindAllString(strings.ToLower(text), -1)
	if len(tokens) == 0 {
		return ""
	}
	return " " + strings.Join(tokens, " ") + " "
}

func profileMatchScore(question string, profiles map[string]string) map[string]float64 {
	terms := autoTagTokens(question)
	scores := make(map[string]float64)
	if len(terms) == 0 || len(profiles) == 0 {
		return scores
	}

	for tag, profile := range profiles {
		normalized := normalizeProfileText(profile)
		if normalized == "" {
			continue
		}

		var score float64
		for _, term := range terms {
			needle := " " + term + " "
			if strings.Contains(normalized, needle) {
				// Acronyms and numbers are highly informative for routing.
				if len(term) <= 3 {
					score += 1.5
				} else {
					score += 1.0
				}
			}
		}

		if score > 0 {
			scores[strings.ToLower(strings.TrimSpace(tag))] = score
		}
	}

	return scores
}

func (app *appSetup) autoSelectTags(ctx context.Context, question string) ([]string, map[string]int, map[string]float64, string, error) {
	preK := app.autoTagPreK
	if preK < 1 {
		preK = 12
	}

	resp, err := app.store.Retriever().Retrieve(ctx, &ai.RetrieverRequest{
		Query: ai.DocumentFromText(question, nil),
		Options: map[string]any{
			"k":    preK,
			"tags": []string{}, // explicitly unfiltered
		},
	})
	if err != nil {
		return nil, nil, nil, "error", err
	}

	tagCounts := make(map[string]int)
	semanticScores := make(map[string]float64)
	for rank, doc := range resp.Documents {
		tag := strings.ToLower(strings.TrimSpace(docMetadataString(doc, "tag")))
		if tag == "" {
			continue
		}
		tagCounts[tag]++

		// Rank-weighted semantic evidence: rank 1 gets preK points,
		// rank 2 gets preK-1, ..., rank preK gets 1.
		weight := float64(preK - rank)
		if weight < 1 {
			weight = 1
		}
		semanticScores[tag] += weight
	}

	finalScores := make(map[string]float64)
	for tag, score := range semanticScores {
		finalScores[tag] += score
	}

	if app.autoTagMode == "profile" || app.autoTagMode == "p15d" || app.autoTagMode == "metadata" {
		if provider, ok := app.store.(tagProfileProvider); ok {
			profiles, err := provider.TagProfiles(ctx, app.autoTagProfileLimit)
			if err != nil {
				log.Printf("[AUTO-TAG:P15D] profile load failed query=%q err=%v", question, err)
			} else {
				profileScores := profileMatchScore(question, profiles)
				for tag, score := range profileScores {
					finalScores[tag] += app.autoTagProfileWeight * score
				}
			}
		} else {
			log.Printf("[AUTO-TAG:P15D] store does not provide tag profiles; using semantic scores only")
		}
	}

	// Keep P15-B fallback available for ablation/reproducibility.
	if app.autoTagMode == "count" || app.autoTagMode == "p15b" {
		rankedCounts := sortedAutoTagCounts(tagCounts)
		if len(rankedCounts) == 0 {
			return nil, tagCounts, finalScores, "low", nil
		}
		if rankedCounts[0].count >= app.autoTagMinDominant {
			return []string{rankedCounts[0].tag}, tagCounts, finalScores, "count-high", nil
		}
		if len(rankedCounts) >= 2 && rankedCounts[0].count+rankedCounts[1].count >= app.autoTagTop2Min {
			return []string{rankedCounts[0].tag, rankedCounts[1].tag}, tagCounts, finalScores, "count-medium", nil
		}
		return nil, tagCounts, finalScores, "low", nil
	}

	ranked := sortedAutoTagScores(finalScores, tagCounts)
	if len(ranked) == 0 || ranked[0].score < app.autoTagMinScore {
		return nil, tagCounts, finalScores, "low", nil
	}

	// P15-D conservative confidence policy:
	// 1) High confidence: select only the top tag.
	// 2) Medium confidence: select top two tags if together they dominate evidence.
	// 3) Low confidence: return no tag filter, so the final retrieval can search all tags.
	totalScore := 0.0
	for _, r := range ranked {
		if r.score > 0 {
			totalScore += r.score
		}
	}
	if totalScore <= 0 {
		return nil, tagCounts, finalScores, "low", nil
	}

	top := ranked[0]
	if len(ranked) == 1 {
		return []string{top.tag}, tagCounts, finalScores, "high-single-candidate", nil
	}

	second := ranked[1]
	topShare := top.score / totalScore
	top2Share := (top.score + second.score) / totalScore

	// Strong single-tag dominance. This avoids the P15-D behavior where a weak
	// winner was still forced and caused wrong-domain failures.
	if second.score <= 0 || top.score >= second.score*app.autoTagHighMargin || topShare >= 0.65 {
		return []string{top.tag}, tagCounts, finalScores, "high", nil
	}

	maxSelected := app.autoTagMaxSelected
	if maxSelected < 1 {
		maxSelected = 1
	}

	// Medium confidence: keep top two only if they jointly dominate the evidence.
	// This is safer than forcing one tag, but still narrows retrieval when the
	// router is reasonably sure the answer is in one of two domains.
	if maxSelected > 1 && top2Share >= app.autoTagTop2Share && second.score >= app.autoTagMinScore*0.5 {
		return []string{top.tag, second.tag}, tagCounts, finalScores, "medium-top2", nil
	}

	// Low confidence: do not apply any tag filter.
	return nil, tagCounts, finalScores, "low-fallback-all-tags", nil
}

func trimConversationHistory(history []conversationTurn, maxMessages int) []conversationTurn {
	if maxMessages <= 0 || len(history) == 0 {
		return nil
	}

	cleaned := make([]conversationTurn, 0, len(history))
	for _, turn := range history {
		role := strings.ToLower(strings.TrimSpace(turn.Role))
		content := strings.TrimSpace(turn.Content)
		if content == "" {
			continue
		}
		if role != "user" && role != "assistant" {
			continue
		}
		cleaned = append(cleaned, conversationTurn{
			Role:    role,
			Content: content,
		})
	}

	if len(cleaned) > maxMessages {
		cleaned = cleaned[len(cleaned)-maxMessages:]
	}
	return cleaned
}

func formatConversationHistory(history []conversationTurn) string {
	if len(history) == 0 {
		return ""
	}

	var b strings.Builder
	for _, turn := range history {
		label := "User"
		if strings.EqualFold(turn.Role, "assistant") {
			label = "Assistant"
		}
		b.WriteString(label)
		b.WriteString(": ")
		b.WriteString(strings.TrimSpace(turn.Content))
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}

func (app *appSetup) resolveConversation(
	ctx context.Context,
	history []conversationTurn,
	question string,
) (conversationResolution, error) {
	fallback := conversationResolution{
		FollowUp:           false,
		StandaloneQuestion: question,
	}

	if app == nil || !app.useConversationalResolution {
		return fallback, nil
	}
	if strings.TrimSpace(app.conversationModel) == "" {
		return fallback, fmt.Errorf("conversational resolution requires a university/ or ollama/ LLM_MODEL")
	}

	history = trimConversationHistory(history, app.conversationHistoryMessages)
	if len(history) == 0 {
		return fallback, nil
	}

	historyText := formatConversationHistory(history)
	if historyText == "" {
		return fallback, nil
	}

	type chatMessage struct {
		Role     string `json:"role"`
		Content  string `json:"content"`
		Thinking string `json:"thinking,omitempty"`
	}
	type ollamaChatRequest struct {
		Model    string         `json:"model"`
		Messages []chatMessage  `json:"messages"`
		Think    bool           `json:"think"`
		Stream   bool           `json:"stream"`
		Options  map[string]any `json:"options,omitempty"`
	}
	type universityChatRequest struct {
		Model     string        `json:"model"`
		Messages  []chatMessage `json:"messages"`
		Stream    bool          `json:"stream"`
		MaxTokens int           `json:"max_tokens"`
	}
	type chatResponse struct {
		Message chatMessage `json:"message"`
		Choices []struct {
			Message      chatMessage `json:"message"`
			FinishReason string      `json:"finish_reason"`
		} `json:"choices"`
	}

	systemPrompt := strings.TrimSpace(`
You rewrite conversational user questions into standalone questions for document retrieval.

Rules:
- Do not answer the user's question.
- Use only the supplied recent conversation to resolve references, pronouns, or omitted subjects.
- If the newest question is already understandable on its own, return it unchanged.
- If it depends on previous conversation, rewrite only enough to make it standalone.
- Preserve the user's meaning and language.
- Do not add facts or explanations.
- Return exactly one standalone question.
- Do not output JSON, Markdown, labels, commentary, or quotation marks.
`)

	userPrompt := fmt.Sprintf(
		"Recent conversation:\n%s\n\nNewest user question:\n%s",
		historyText,
		question,
	)

	useUniversity := strings.HasPrefix(os.Getenv("LLM_MODEL"), "university/")
	// Qwen thinking models can exhaust a small completion budget before
	// emitting user-visible content. Request a concise non-thinking rewrite.
	if useUniversity {
		userPrompt += "\n/no_think"
	}
	messages := []chatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
	var payload any
	var endpoint string
	if useUniversity {
		payload = universityChatRequest{
			Model:     app.conversationModel,
			Messages:  messages,
			Stream:    false,
			MaxTokens: 600,
		}
		endpoint = strings.TrimRight(app.uniAPIBaseURL, "/") + "/chat/completions"
	} else {
		payload = ollamaChatRequest{
			Model:    app.conversationModel,
			Messages: messages,
			Think:    false,
			Stream:   false,
			Options:  map[string]any{"temperature": 0, "num_predict": 80},
		}
		endpoint = strings.TrimRight(app.ollamaAddr, "/") + "/api/chat"
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fallback, fmt.Errorf("marshal conversational resolver request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return fallback, fmt.Errorf("create conversational resolver request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if useUniversity {
		req.Header.Set("Authorization", "Bearer "+app.uniAPIKey)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fallback, fmt.Errorf("conversational resolver request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fallback, fmt.Errorf("conversational resolver returned HTTP %d: %s", resp.StatusCode, previewText(string(raw), 400))
	}

	var decoded chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return fallback, fmt.Errorf("decode conversational resolver response: %w", err)
	}
	var standalone string
	if useUniversity {
		if len(decoded.Choices) == 0 {
			return fallback, fmt.Errorf("university conversational resolver returned no choices")
		}
		standalone = strings.TrimSpace(decoded.Choices[0].Message.Content)
	} else {
		standalone = strings.TrimSpace(decoded.Message.Content)
	}
	if standalone == "" {
		if useUniversity && len(decoded.Choices) > 0 {
			return fallback, fmt.Errorf("conversational resolver returned empty content (finish_reason=%q)", decoded.Choices[0].FinishReason)
		}
		return fallback, fmt.Errorf("conversational resolver returned empty content")
	}

	standalone = strings.TrimSpace(strings.Trim(standalone, "`\"'"))
	standalone = strings.Join(strings.Fields(standalone), " ")
	if standalone == "" {
		return fallback, fmt.Errorf("conversational resolver returned only formatting characters")
	}

	normalize := func(v string) string {
		v = strings.ToLower(strings.TrimSpace(v))
		v = strings.TrimRight(v, " ?!.;:")
		return strings.Join(strings.Fields(v), " ")
	}

	return conversationResolution{
		FollowUp:           normalize(standalone) != normalize(question),
		StandaloneQuestion: standalone,
	}, nil
}

// ─────────────────────────────────────────────
// P21-A: conversational-message bypass
// ─────────────────────────────────────────────
func normalizeConversationalMessage(text string) string {
	text = strings.ToLower(strings.TrimSpace(text))
	text = strings.Trim(text, " \t\r\n.!?,;:'\\\"")
	text = strings.Join(strings.Fields(text), " ")
	return text
}

func conversationalReply(question string) (string, bool) {
	q := normalizeConversationalMessage(question)
	if q == "" {
		return "", false
	}

	switch q {
	case "thanks", "thank you", "thank you very much", "thanks a lot",
		"thank you for your help", "thank you for answering",
		"thank you for answering everything", "thanks for your help",
		"thanks for answering", "thanks for answering everything",
		"than you", "than you for answering", "than you for your help",
		"thankyou", "thx":
		return "You're welcome! Glad I could help.", true

	case "danke", "danke schön", "dankeschön", "vielen dank",
		"vielen dank für deine hilfe", "danke für deine hilfe":
		return "Gern geschehen!", true

	case "hi", "hello", "hey", "hello there":
		return "Hello! How can I help you with your documents?", true

	case "hallo", "guten tag":
		return "Hallo! Was möchtest du aus deinen Dokumenten wissen?", true

	case "ok", "okay", "got it", "understood", "makes sense", "great":
		return "Got it.", true

	case "bye", "goodbye", "see you", "see you later":
		return "Goodbye!", true

	case "tschüss", "tschüs", "bis später":
		return "Tschüss!", true
	}

	return "", false
}

// ─────────────────────────────────────────────
// P21-B: grounded inline citations + unsupported follow-up wording
// ─────────────────────────────────────────────

var evidenceCitationRE = regexp.MustCompile(`(?i)\[E([0-9]+)\]`)

type citationEvidence struct {
	ID          int
	SourceIndex int
	Text        string
}

func splitEvidenceFragments(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	// First preserve paragraph / bullet structure where possible.
	rawBlocks := regexp.MustCompile(`\n\s*\n`).Split(text, -1)
	fragments := make([]string, 0)

	const targetChars = 420
	const hardMaxChars = 620

	appendFragment := func(fragment string) {
		fragment = strings.TrimSpace(fragment)
		if fragment == "" {
			return
		}
		fragments = append(fragments, fragment)
	}

	for _, rawBlock := range rawBlocks {
		block := strings.TrimSpace(rawBlock)
		if block == "" {
			continue
		}

		// Short blocks are already good citation units.
		if len([]rune(block)) <= hardMaxChars {
			appendFragment(block)
			continue
		}

		// For long paragraphs, split conservatively on sentence-like boundaries.
		parts := regexp.MustCompile(`([.!?])\s+`).Split(block, -1)
		var current strings.Builder

		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}

			if current.Len() > 0 && current.Len()+1+len(part) > targetChars {
				appendFragment(current.String())
				current.Reset()
			}

			if current.Len() > 0 {
				current.WriteString(" ")
			}
			current.WriteString(part)

			// Extremely long sentence / OCR block: hard split by words.
			if current.Len() > hardMaxChars {
				words := strings.Fields(current.String())
				current.Reset()

				var wordBuf strings.Builder
				for _, word := range words {
					if wordBuf.Len() > 0 && wordBuf.Len()+1+len(word) > targetChars {
						appendFragment(wordBuf.String())
						wordBuf.Reset()
					}
					if wordBuf.Len() > 0 {
						wordBuf.WriteString(" ")
					}
					wordBuf.WriteString(word)
				}
				if wordBuf.Len() > 0 {
					current.WriteString(wordBuf.String())
				}
			}
		}

		appendFragment(current.String())
	}

	// Merge tiny adjacent fragments so the model gets enough local meaning.
	if len(fragments) <= 1 {
		return fragments
	}

	merged := make([]string, 0, len(fragments))
	for _, fragment := range fragments {
		if len([]rune(fragment)) < 90 && len(merged) > 0 {
			prev := merged[len(merged)-1]
			if len([]rune(prev))+1+len([]rune(fragment)) <= hardMaxChars {
				merged[len(merged)-1] = strings.TrimSpace(prev + " " + fragment)
				continue
			}
		}
		merged = append(merged, fragment)
	}

	return merged
}

func isLikelyConversationalFollowUp(originalQuestion, resolvedQuestion string) bool {
	original := strings.ToLower(strings.TrimSpace(originalQuestion))
	resolved := strings.ToLower(strings.TrimSpace(resolvedQuestion))

	if original == "" || resolved == "" {
		return false
	}

	// P20 rewrites ambiguous follow-ups such as "How is it different from X?"
	// into standalone questions. If the resolved question differs materially,
	// treat this as a conversational follow-up even when the user has a typo.
	normalize := func(v string) string {
		v = strings.TrimSpace(v)
		v = strings.TrimRight(v, " ?!.;:")
		return strings.Join(strings.Fields(v), " ")
	}

	if normalize(original) != normalize(resolved) {
		return true
	}

	// Conservative fallback for common pronoun-based follow-up forms.
	followUpMarkers := []string{
		"how is it ",
		"how it is ",
		"what about it",
		"what about this",
		"what about that",
		"and this",
		"and that",
		"compared to",
		"different from",
		"difference from",
		"unterschied zu",
		"wie unterscheidet",
	}

	padded := " " + original + " "
	for _, marker := range followUpMarkers {
		if strings.Contains(padded, marker) {
			return true
		}
	}

	return false
}

func isLikelyComparisonFollowUp(originalQuestion, resolvedQuestion string) bool {
	if !isLikelyConversationalFollowUp(originalQuestion, resolvedQuestion) {
		return false
	}

	q := strings.ToLower(strings.TrimSpace(originalQuestion))
	padded := " " + q + " "

	comparisonMarkers := []string{
		"different",
		"difference",
		"differ",
		"compare",
		"comparison",
		"compared",
		"versus",
		" vs ",
		"unterschied",
		"anders",
		"vergleichen",
		"vergleich",

		// Common misspellings seen in conversational input.
		"dirrerent",
		"diferent",
		"differrent",
	}

	for _, marker := range comparisonMarkers {
		if strings.Contains(padded, marker) {
			return true
		}
	}

	// "How is it ... from X?" is almost always a comparison follow-up, even if
	// the comparison adjective itself is misspelled.
	if (strings.Contains(padded, " how is it ") || strings.Contains(padded, " how it is ")) &&
		strings.Contains(padded, " from ") {
		return true
	}

	return false
}

func unsupportedAnswerFor(originalQuestion, resolvedQuestion string) string {
	if !isLikelyConversationalFollowUp(originalQuestion, resolvedQuestion) {
		return "NOT IN DOCUMENT"
	}

	q := strings.ToLower(strings.TrimSpace(originalQuestion))
	padded := " " + q + " "

	isGerman := strings.Contains(padded, " wie ") ||
		strings.Contains(padded, " unterschied") ||
		strings.Contains(padded, " vergleichen") ||
		strings.Contains(padded, " anders ")

	if isLikelyComparisonFollowUp(originalQuestion, resolvedQuestion) {
		if isGerman {
			return "Die indexierten Dokumente enthalten keinen ausreichend belegten Vergleich zwischen diesen beiden Themen."
		}
		return "The indexed documents do not provide enough supported information to compare these two topics."
	}

	if isGerman {
		return "Die indexierten Dokumente enthalten nicht genügend belegte Informationen, um diese Folgefrage zu beantworten."
	}

	return "The indexed documents do not provide enough supported information to answer this follow-up question."
}

func canonicalSourceKey(src qaSource, fallbackIndex int) string {
	if strings.TrimSpace(src.URL) != "" {
		return "url:" + strings.ToLower(strings.TrimSpace(src.URL))
	}

	parts := []string{
		"pdf",
		strings.ToLower(strings.TrimSpace(src.Tag)),
		strings.TrimSpace(src.Pages),
		strings.ToLower(strings.TrimSpace(src.Section)),
		strings.ToLower(strings.TrimSpace(src.Breadcrumb)),
	}

	key := strings.Join(parts, "|")
	if strings.Trim(key, "|") == "pdf" {
		return fmt.Sprintf("fallback:%d", fallbackIndex)
	}
	return key
}

func collapseDuplicateVisibleCitations(answer string) string {
	// Go's regexp engine (RE2) does not support backreferences such as \1.
	// Detect each citation cluster and deduplicate IDs in Go code instead.
	clusterRE := regexp.MustCompile(`(?:\[\d+\]\s*){2,}`)
	idRE := regexp.MustCompile(`\[(\d+)\]`)

	return clusterRE.ReplaceAllStringFunc(answer, func(cluster string) string {
		matches := idRE.FindAllStringSubmatch(cluster, -1)
		seen := map[string]bool{}
		out := make([]string, 0, len(matches))

		for _, match := range matches {
			if len(match) < 2 {
				continue
			}

			id := match[1]
			if seen[id] {
				continue
			}

			seen[id] = true
			out = append(out, "["+id+"]")
		}

		return strings.Join(out, "")
	})
}

func auditCitationCoverage(answer string) (substantive int, uncited int) {
	lines := strings.Split(answer, "\n")
	citationRE := regexp.MustCompile(`\[\d+\]`)
	wordRE := regexp.MustCompile(`[A-Za-zÀ-ÿ0-9]+`)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		line = strings.TrimSpace(strings.TrimLeft(line, "#-*0123456789. "))

		units := regexp.MustCompile(`(?m)(?:[.!?](?:\s+|$)|;\s+)`).Split(line, -1)
		for _, unit := range units {
			unit = strings.TrimSpace(unit)
			if unit == "" {
				continue
			}

			words := wordRE.FindAllString(unit, -1)
			if len(words) < 5 {
				continue
			}

			lower := strings.ToLower(unit)
			if strings.HasPrefix(lower, "in summary") ||
				strings.HasPrefix(lower, "summary") ||
				strings.HasPrefix(lower, "therefore") ||
				strings.HasPrefix(lower, "overall") {
				continue
			}

			substantive++
			if !citationRE.MatchString(unit) {
				uncited++
			}
		}
	}

	return substantive, uncited
}

func validateAndNormalizeEvidenceCitations(
	answer string,
	sources []qaSource,
	evidence []citationEvidence,
) (string, []qaSource) {
	answer = strings.TrimSpace(answer)
	if answer == "" || len(sources) == 0 || len(evidence) == 0 {
		return answer, make([]qaSource, 0)
	}

	evidenceByID := make(map[int]citationEvidence, len(evidence))
	for _, item := range evidence {
		evidenceByID[item.ID] = item
	}

	matches := evidenceCitationRE.FindAllStringSubmatch(answer, -1)
	if len(matches) == 0 {
		log.Printf("[P21-E:citations] answer contained no evidence citations; hiding source cards")
		return answer, make([]qaSource, 0)
	}

	groupIndexByKey := map[string]int{}
	groupRepresentativeSource := make([]qaSource, 0)
	sourceIndexToGroup := map[int]int{}

	for sourceIndex, src := range sources {
		key := canonicalSourceKey(src, sourceIndex)
		groupIndex, ok := groupIndexByKey[key]
		if !ok {
			groupIndex = len(groupRepresentativeSource)
			groupIndexByKey[key] = groupIndex
			groupRepresentativeSource = append(groupRepresentativeSource, src)
		}
		sourceIndexToGroup[sourceIndex] = groupIndex
	}

	selectedGroups := make([]int, 0)
	selectedGroupSet := map[int]bool{}

	for _, match := range matches {
		if len(match) < 2 {
			continue
		}

		evidenceID, err := strconv.Atoi(match[1])
		if err != nil {
			continue
		}

		item, ok := evidenceByID[evidenceID]
		if !ok || item.SourceIndex < 0 || item.SourceIndex >= len(sources) {
			continue
		}

		groupIndex, ok := sourceIndexToGroup[item.SourceIndex]
		if !ok || selectedGroupSet[groupIndex] {
			continue
		}

		selectedGroupSet[groupIndex] = true
		selectedGroups = append(selectedGroups, groupIndex)
	}

	if len(selectedGroups) == 0 {
		cleaned := evidenceCitationRE.ReplaceAllString(answer, "")
		cleaned = regexp.MustCompile(`[ \t]+([,.;:!?])`).ReplaceAllString(cleaned, "$1")
		cleaned = regexp.MustCompile(`[ \t]{2,}`).ReplaceAllString(cleaned, " ")
		log.Printf("[P21-E:citations] model produced only invalid evidence IDs; hiding source cards")
		return strings.TrimSpace(cleaned), make([]qaSource, 0)
	}

	displayNumberByGroup := map[int]int{}
	finalSources := make([]qaSource, 0, len(selectedGroups))
	for displayIndex, groupIndex := range selectedGroups {
		displayNumberByGroup[groupIndex] = displayIndex + 1
		finalSources = append(finalSources, groupRepresentativeSource[groupIndex])
	}

	usedEvidence := map[int]bool{}

	normalized := evidenceCitationRE.ReplaceAllStringFunc(answer, func(token string) string {
		match := evidenceCitationRE.FindStringSubmatch(token)
		if len(match) < 2 {
			return ""
		}

		evidenceID, err := strconv.Atoi(match[1])
		if err != nil {
			return ""
		}

		item, ok := evidenceByID[evidenceID]
		if !ok {
			return ""
		}

		groupIndex, ok := sourceIndexToGroup[item.SourceIndex]
		if !ok {
			return ""
		}

		displayNumber, ok := displayNumberByGroup[groupIndex]
		if !ok {
			return ""
		}

		usedEvidence[evidenceID] = true
		return fmt.Sprintf("[%d]", displayNumber)
	})

	normalized = collapseDuplicateVisibleCitations(normalized)
	normalized = regexp.MustCompile(`[ \t]+([,.;:!?])`).ReplaceAllString(normalized, "$1")
	normalized = regexp.MustCompile(`[ \t]{2,}`).ReplaceAllString(normalized, " ")
	normalized = strings.TrimSpace(normalized)

	substantive, uncited := auditCitationCoverage(normalized)
	coverage := 1.0
	if substantive > 0 {
		coverage = float64(substantive-uncited) / float64(substantive)
	}

	log.Printf(
		"[P21-E:citations] evidenceAvailable=%d evidenceUsed=%d contextSources=%d visibleSourceGroups=%d substantiveUnits=%d uncitedUnits=%d coverage=%.2f",
		len(evidence),
		len(usedEvidence),
		len(sources),
		len(finalSources),
		substantive,
		uncited,
		coverage,
	)

	return normalized, finalSources
}

func (app *appSetup) defineQAFlow() *core.Flow[qaInput, qaResponse, string] {
	return genkit.DefineStreamingFlow(
		app.genk, "docQA",
		func(ctx context.Context, input qaInput, sendChunk core.StreamCallback[string]) (qaResponse, error) {
			flowStart := time.Now()

			question := strings.TrimSpace(input.Question)
			if question == "" {
				return qaResponse{Answer: "NOT IN DOCUMENT", Sources: make([]qaSource, 0)}, nil
			}

			// P21-A: conversational messages bypass retrieval and never get sources.
			if reply, ok := conversationalReply(question); ok {
				sendChunk(ctx, reply)

				totalMs := float64(time.Since(flowStart).Milliseconds())
				log.Printf("[P21-A:conversation] bypassRetrieval=true question=%q reply=%q", question, reply)

				return qaResponse{
					Answer:          reply,
					Sources:         make([]qaSource, 0),
					TotalMs:         totalMs,
					OtherOverheadMs: totalMs,
				}, nil
			}

			// P20-A: resolve conversational follow-ups into a standalone retrieval question.
			// The original user wording is preserved for final answer generation.
			retrievalQuestion := question
			if app.useConversationalResolution && len(input.History) > 0 {
				resolved, err := app.resolveConversation(ctx, input.History, question)
				if err != nil {
					log.Printf("[P20-A:followup] resolver failed; fallback to original question=%q err=%v", question, err)
				} else {
					retrievalQuestion = strings.TrimSpace(resolved.StandaloneQuestion)
					if retrievalQuestion == "" {
						retrievalQuestion = question
					}
					log.Printf(
						"[P20-A:followup] followUp=%v original=%q retrieval=%q historyMessages=%d",
						resolved.FollowUp,
						question,
						retrievalQuestion,
						len(trimConversationHistory(input.History, app.conversationHistoryMessages)),
					)
				}
			}

			allowedTags := append([]string{}, app.allowedTags...)
			if strings.TrimSpace(input.Tag) != "" {
				allowedTags = []string{input.Tag}
			}
			allowedTags = normalizeAllowedTags(allowedTags)

			// P15-D conservative automatic tag routing.
			// Manual tags still have priority:
			// 1) input.Tag from the request
			// 2) ALLOWED_TAGS from .env
			// Auto-routing only runs when both are empty.
			if app.autoTagRouting && len(allowedTags) == 0 {
				selectedTags, tagCounts, tagScores, confidence, err := app.autoSelectTags(ctx, retrievalQuestion)
				if err != nil {
					log.Printf("[AUTO-TAG:P15D] failed query=%q err=%v", retrievalQuestion, err)
				} else {
					allowedTags = normalizeAllowedTags(selectedTags)
					if len(allowedTags) == 0 {
						log.Printf("[AUTO-TAG:P15D] query=%q mode=%s confidence=%s counts=%v scores=%v selected=[] fallback=all-tags", retrievalQuestion, app.autoTagMode, confidence, tagCounts, tagScores)
					} else {
						log.Printf("[AUTO-TAG:P15D] query=%q mode=%s confidence=%s counts=%v scores=%v selected=%v", retrievalQuestion, app.autoTagMode, confidence, tagCounts, tagScores, allowedTags)
					}
				}
			} else if app.autoTagRouting {
				log.Printf("[AUTO-TAG] skipped because manual/env tags are already set: %v", allowedTags)
			}

			debugf("[docQA] allowedTags=%v", allowedTags)

			var hydeMs float64
			var retrievalMs float64
			var rerankMs float64
			var answerGenerationMs float64

			// ── Step 1: Optional HyDE or Phase 4C question expansion ──
			hydeText := retrievalQuestion
			secondaryQueries := []string{}
			secondaryLabel := "vector-secondary"
			if app.useHyDE {
				hydeStart := time.Now()
				hydeResp, err := app.hydePrompt.Execute(ctx,
					ai.WithModel(app.llm),
					ai.WithInput(map[string]any{"question": retrievalQuestion}),
				)
				hydeMs = float64(time.Since(hydeStart).Milliseconds())

				if err == nil && hydeResp.Text() != "" {
					hydeText = hydeResp.Text()
				}

				if len(hydeText) > 100 {
					log.Printf("[docQA] hyde=%q", hydeText[:100])
				} else {
					log.Printf("[docQA] hyde=%q", hydeText)
				}
				if hydeText != retrievalQuestion {
					secondaryQueries = append(secondaryQueries, hydeText)
					secondaryLabel = "vector-hyde"
				}
			} else if app.useQueryExpansion {
				expansionStart := time.Now()
				secondaryQueries = buildQuestionExpansions(retrievalQuestion, app.queryExpansionK)
				hydeMs = float64(time.Since(expansionStart).Milliseconds())
				secondaryLabel = "vector-question-expansion"
				log.Printf("[docQA] question expansions=%v", secondaryQueries)
			} else {
				hydeMs = 0
				debugf("[docQA] hyde/question expansion disabled")
			}

			// ── Step 2: Retrieval with metadata pre-filtering ──
			retrievalStart := time.Now()

			resp1, err := app.store.Retriever().Retrieve(ctx, &ai.RetrieverRequest{
				Query: ai.DocumentFromText(retrievalQuestion, nil),
				Options: map[string]any{
					"k":    5,
					"tags": allowedTags,
				},
			})
			if err != nil {
				return qaResponse{}, err
			}

			var resp2 *ai.RetrieverResponse
			if len(secondaryQueries) > 0 {
				var secondaryDocs []*ai.Document
				for _, secondaryQuery := range secondaryQueries {
					if strings.TrimSpace(secondaryQuery) == "" {
						continue
					}
					secondaryResp, err := app.store.Retriever().Retrieve(ctx, &ai.RetrieverRequest{
						Query: ai.DocumentFromText(secondaryQuery, nil),
						Options: map[string]any{
							"k":    app.queryExpansionRetrieveK,
							"tags": allowedTags,
						},
					})
					if err != nil {
						return qaResponse{}, err
					}
					secondaryDocs = append(secondaryDocs, secondaryResp.Documents...)
				}
				if len(secondaryDocs) > 0 {
					resp2 = &ai.RetrieverResponse{Documents: secondaryDocs}
				}
			}

			var kwDocs []*ai.Document
			if app.useKeyword {
				debugf("[docQA] keyword search for: %v", retrievalQuestion)
				kwDocs, err = app.store.KeywordSearch(ctx, retrievalQuestion, 3, allowedTags)
				if err != nil {
					debugf("[docQA] keyword search failed: %v", err)
					kwDocs = nil
				}
			} else {
				debugf("[docQA] keyword disabled")
			}

			retrievalMs = float64(time.Since(retrievalStart).Milliseconds())

			if len(allowedTags) > 0 {
				AuditContamination(resp1.Documents, allowedTags, "vector-question")
				if resp2 != nil {
					AuditContamination(resp2.Documents, allowedTags, secondaryLabel)
				}
				if len(kwDocs) > 0 {
					AuditContamination(kwDocs, allowedTags, "keyword")
				}
			}

			// ── Step 2b: Fusion method: simple, score, RRF, Borda, or CombMNZ ──
			merged, n2, nKw := mergeRetrievedDocs(app, resp1, resp2, kwDocs, allowedTags, secondaryLabel)

			// P16-A: apply metadata heading/section/breadcrumb boost after fusion and before reranking.
			// This is the clean experiment: retrieval and chunking remain unchanged, only pre-rerank candidate order changes.
			if app.usePreRerankHeadingBoost {
				merged = applyPreRerankHeadingBoost(retrievalQuestion, merged, app)
			}

			// P17-B1/B2: stronger exact section/title phrase boost before reranking.
			// B2 additionally demotes sibling sections and protects exact matches in final context.
			if app.useP17BExactSectionBoost {
				merged = applyP17BExactSectionMatchBoost(retrievalQuestion, merged, app)
			}

			// P17-B2: demote sibling sections that partially overlap with an exact section match.
			// This targets cases such as Storage Hierarchy vs Storage Models/PAX Storage Model.
			if app.useP17BSiblingDemotion {
				merged = applyP17BSiblingSectionDemotion(retrievalQuestion, merged, app)
			}

			debugf("[docQA] retrieved %d docs (question) + %d docs (secondary) + %d docs (keyword), %d unique after %s fusion",
				len(resp1.Documents), n2, nKw, len(merged), app.fusionMethod)

			// ── Step 3: Optional LLM reranking ──
			rerankStart := time.Now()

			type rankedDoc struct {
				doc   *ai.Document
				score int
			}

			var candidates []rankedDoc

			if app.useReranker {
				rerankChunks := make([]map[string]any, 0, len(merged))
				for i, entry := range merged {
					text := entry.text
					if len(text) > 500 {
						text = text[:500]
					}
					rerankChunks = append(rerankChunks, map[string]any{
						"num":  i + 1,
						"text": text,
					})
				}

				rerankResp, err := app.rerankPrompt.Execute(ctx,
					ai.WithModel(app.llm),
					ai.WithInput(map[string]any{
						"question": retrievalQuestion,
						"chunks":   rerankChunks,
					}),
				)

				rerankMs = float64(time.Since(rerankStart).Milliseconds())

				if err == nil && rerankResp.Text() != "" {
					scores := parseRerankScores(rerankResp.Text(), len(merged))
					log.Printf("[docQA] rerank scores: %v", scores)

					for i, entry := range merged {
						candidates = append(candidates, rankedDoc{doc: entry.doc, score: scores[i]})
					}

					sort.SliceStable(candidates, func(i, j int) bool {
						return candidates[i].score > candidates[j].score
					})

					aboveMinScore := 0
					for _, c := range candidates {
						if c.score >= app.rerankMinScore {
							aboveMinScore++
						}
					}
					log.Printf("[docQA] strict rerank filter: before=%d after=%d minScore=%d", len(candidates), aboveMinScore, app.rerankMinScore)
				} else {
					debugf("[docQA] rerank failed, using original order (err=%v)", err)
					for _, entry := range merged {
						candidates = append(candidates, rankedDoc{doc: entry.doc, score: 5})
					}
				}
			} else {
				rerankMs = 0
				debugf("[docQA] reranker OFF — using merged retrieval order")
				for _, entry := range merged {
					candidates = append(candidates, rankedDoc{doc: entry.doc, score: 5})
				}
			}

			// P17-B2: identify sibling conflicts after reranking as well.
			// Pre-rerank score demotion alone is not enough because the LLM reranker can still
			// move a sibling section above the exact section. This map is used below to push
			// conflicting siblings behind protected/neutral candidates and to skip them after
			// an exact heading match has already been added to final context.
			p17BSiblingConflictReasons := make(map[*ai.Document]string)
			if app.useP17BSiblingDemotion && len(candidates) > 1 {
				targetsByTag := map[string][]string{}
				seenTargets := map[string]map[string]bool{}
				for _, rd := range candidates {
					tag := strings.ToLower(strings.TrimSpace(docMetadataString(rd.doc, "tag")))
					if tag == "" {
						continue
					}
					ok, phrase := p17BDirectHeadingPhraseForDoc(retrievalQuestion, rd.doc)
					if !ok || phrase == "" {
						continue
					}
					if seenTargets[tag] == nil {
						seenTargets[tag] = map[string]bool{}
					}
					if seenTargets[tag][phrase] {
						continue
					}
					seenTargets[tag][phrase] = true
					targetsByTag[tag] = append(targetsByTag[tag], phrase)
				}

				conflictCount := 0
				for _, rd := range candidates {
					tag := strings.ToLower(strings.TrimSpace(docMetadataString(rd.doc, "tag")))
					targets := targetsByTag[tag]
					if len(targets) == 0 {
						continue
					}
					conflict, reason := p17BConflictingSiblingReasonForDoc(retrievalQuestion, rd.doc, targets, app.p17BSiblingMinOverlap)
					if !conflict {
						continue
					}
					p17BSiblingConflictReasons[rd.doc] = reason
					conflictCount++
					debugf(
						"[P17-B2:postRerankSiblingConflict] reason=%s rerank=%d pages=%q section=%q breadcrumb=%q tag=%q text=%q",
						reason,
						rd.score,
						docMetadataString(rd.doc, "pages"),
						docMetadataString(rd.doc, "section"),
						docMetadataString(rd.doc, "breadcrumb"),
						docMetadataString(rd.doc, "tag"),
						previewText(extractContextText(rd.doc), 120),
					)
				}
				log.Printf("[P17-B2:postRerankSiblingConflict] enabled=%v candidates=%d targetTags=%d conflicts=%d", app.useP17BSiblingDemotion, len(candidates), len(targetsByTag), conflictCount)
			}

			// P16-B: final-context heading protection.
			// This runs after reranking but before C5.4 final context selection.
			// Exact/strong heading matches are moved ahead of related sibling sections so the answer generator
			// sees the most heading-aligned evidence first.
			protectedPriorities := make(map[*ai.Document]int)
			protectedReasons := make(map[*ai.Document]string)
			if app.useFinalHeadingProtection && len(candidates) > 1 {
				protectedCount := 0
				strongProtected := 0
				for _, rd := range candidates {
					priority, reason := p16FinalHeadingProtectionPriority(retrievalQuestion, rd.doc)
					if priority >= app.finalHeadingMinPriority {
						protectedPriorities[rd.doc] = priority
						protectedReasons[rd.doc] = reason
						protectedCount++
						if priority >= 3 {
							strongProtected++
						}
						debugf(
							"[P16-B:finalHeadingProtection] priority=%d reason=%s rerank=%d pages=%q section=%q breadcrumb=%q tag=%q text=%q",
							priority,
							reason,
							rd.score,
							docMetadataString(rd.doc, "pages"),
							docMetadataString(rd.doc, "section"),
							docMetadataString(rd.doc, "breadcrumb"),
							docMetadataString(rd.doc, "tag"),
							previewText(extractContextText(rd.doc), 120),
						)
					}
				}

				if protectedCount > 0 || len(p17BSiblingConflictReasons) > 0 {
					sort.SliceStable(candidates, func(i, j int) bool {
						pi := protectedPriorities[candidates[i].doc]
						pj := protectedPriorities[candidates[j].doc]
						if pi != pj {
							return pi > pj
						}

						ci := p17BSiblingConflictReasons[candidates[i].doc] != ""
						cj := p17BSiblingConflictReasons[candidates[j].doc] != ""
						if ci != cj {
							return !ci && cj
						}

						if candidates[i].score != candidates[j].score {
							return candidates[i].score > candidates[j].score
						}
						return i < j
					})
				}

				log.Printf("[P16-B:finalHeadingProtection] enabled=%v candidates=%d protected=%d strong=%d minPriority=%d", app.useFinalHeadingProtection, len(candidates), protectedCount, strongProtected, app.finalHeadingMinPriority)
			}

			// ── Step 3b: P4F strict rerank filtering + C5.4 evidence validation + final-context selection ──
			maxResults := app.finalContextK
			minScore := app.rerankMinScore
			maxContextChars := app.maxContextChars

			var selectedNormalizedContexts []string
			var contextText strings.Builder
			sources := make([]qaSource, 0)
			evidence := make([]citationEvidence, 0)
			kept := 0
			skippedDuplicates := 0
			skippedBelowScore := 0
			skippedCharCap := 0
			skippedLowQuality := 0
			qualityBypassFallbacks := 0

			debugf(
				"[finalContext:C5.4] candidatesBeforeDedup=%d finalContextK=%d rerankMinScore=%d maxContextChars=%d useReranker=%v",
				len(candidates),
				maxResults,
				minScore,
				maxContextChars,
				app.useReranker,
			)

			addDoc := func(doc *ai.Document, rerankScore int, candidateRank int, fallback bool, keepReason string, quality float64) bool {
				text := extractContextText(doc)
				normalizedText := normalizeContextForDedup(text)
				textHash := stableTextHash(text)
				matchedTerms, totalTerms, overlapRatio := questionTermOverlap(retrievalQuestion, text)
				metaOverlap := metadataMatchScore(retrievalQuestion, doc)

				isDup, reason := isNearDuplicateContext(normalizedText, selectedNormalizedContexts)
				if isDup {
					skippedDuplicates++
					debugf(
						"[finalContext:C5.4] selected=false candidateRank=%d rerank=%d reason=%s keepReason=%s quality=%.2f textHash=%s fallback=%v qOverlap=%d/%d(%.2f) metaOverlap=%.2f chars=%d pages=%q section=%q breadcrumb=%q tag=%q text=%q",
						candidateRank,
						rerankScore,
						reason,
						keepReason,
						quality,
						textHash,
						fallback,
						matchedTerms,
						totalTerms,
						overlapRatio,
						metaOverlap,
						len(text),
						docMetadataString(doc, "pages"),
						docMetadataString(doc, "section"),
						docMetadataString(doc, "breadcrumb"),
						docMetadataString(doc, "tag"),
						previewText(text, 120),
					)
					return false
				}

				if maxContextChars > 0 && contextText.Len()+len(text) > maxContextChars {
					skippedCharCap++
					debugf(
						"[finalContext:C5.4] selected=false candidateRank=%d rerank=%d reason=max_context_chars keepReason=%s quality=%.2f textHash=%s fallback=%v qOverlap=%d/%d(%.2f) metaOverlap=%.2f currentChars=%d candidateChars=%d maxContextChars=%d pages=%q section=%q breadcrumb=%q tag=%q text=%q",
						candidateRank,
						rerankScore,
						keepReason,
						quality,
						textHash,
						fallback,
						matchedTerms,
						totalTerms,
						overlapRatio,
						metaOverlap,
						contextText.Len(),
						len(text),
						maxContextChars,
						docMetadataString(doc, "pages"),
						docMetadataString(doc, "section"),
						docMetadataString(doc, "breadcrumb"),
						docMetadataString(doc, "tag"),
						previewText(text, 120),
					)
					return false
				}

				selectedNormalizedContexts = append(selectedNormalizedContexts, normalizedText)
				kept++

				var src qaSource
				if doc.Metadata != nil {
					if v, ok := doc.Metadata["pages"].(string); ok {
						src.Pages = v
					}
					if v, ok := doc.Metadata["section"].(string); ok {
						src.Section = v
					}
					if v, ok := doc.Metadata["breadcrumb"].(string); ok {
						src.Breadcrumb = v
					}
					if v, ok := doc.Metadata["tag"].(string); ok {
						src.Tag = v
					}
					if v, ok := doc.Metadata["datum"].(string); ok {
						src.Datum = v
					}
					if v, ok := doc.Metadata["url"].(string); ok {
						src.URL = v
					}
				}

				sourceIndex := len(sources)
				sourceID := sourceIndex + 1
				sources = append(sources, src)

				contextText.WriteString(fmt.Sprintf("SOURCE S%d\n", sourceID))
				if src.URL != "" {
					contextText.WriteString(fmt.Sprintf("URL: %s\n", src.URL))
					if src.Breadcrumb != "" {
						contextText.WriteString(fmt.Sprintf("Section: %s\n", src.Breadcrumb))
					}
				} else {
					if src.Pages != "" {
						contextText.WriteString(fmt.Sprintf("Page: %s\n", src.Pages))
					}
					if src.Section != "" {
						contextText.WriteString(fmt.Sprintf("Section: %s\n", src.Section))
					}
					if src.Breadcrumb != "" && src.Breadcrumb != src.Section {
						contextText.WriteString(fmt.Sprintf("Breadcrumb: %s\n", src.Breadcrumb))
					}
				}
				if src.Tag != "" {
					contextText.WriteString(fmt.Sprintf("Tag: %s\n", src.Tag))
				}
				if src.Datum != "" {
					contextText.WriteString(fmt.Sprintf("Date: %s\n", src.Datum))
				}

				fragments := splitEvidenceFragments(text)
				if len(fragments) == 0 {
					fragments = []string{text}
				}

				for _, fragment := range fragments {
					evidenceID := len(evidence) + 1
					evidence = append(evidence, citationEvidence{
						ID:          evidenceID,
						SourceIndex: sourceIndex,
						Text:        fragment,
					})
					contextText.WriteString(fmt.Sprintf("\n[E%d]\n%s\n", evidenceID, fragment))
				}

				contextText.WriteString("\n")

				debugf(
					"[finalContext:C5.4] selected=true finalRank=%d candidateRank=%d rerank=%d reason=%s quality=%.2f textHash=%s fallback=%v qOverlap=%d/%d(%.2f) metaOverlap=%.2f chars=%d totalContextChars=%d pages=%q section=%q breadcrumb=%q tag=%q text=%q",
					kept,
					candidateRank,
					rerankScore,
					keepReason,
					quality,
					textHash,
					fallback,
					matchedTerms,
					totalTerms,
					overlapRatio,
					metaOverlap,
					len(text),
					contextText.Len(),
					docMetadataString(doc, "pages"),
					docMetadataString(doc, "section"),
					docMetadataString(doc, "breadcrumb"),
					docMetadataString(doc, "tag"),
					previewText(text, 120),
				)

				return true
			}

			// P16-C: protected inclusion of exact/strong heading matches.
			// Unlike P16-B, this does not merely reorder candidates. It guarantees that the strongest
			// exact heading candidate is included in the final context before related sibling sections.
			if app.useFinalHeadingForceInclude && len(candidates) > 0 && kept < maxResults {
				type p16cProtectedCandidate struct {
					rd       rankedDoc
					rank     int
					priority int
					reason   string
					strength float64
					chars    int
				}

				protectedCandidates := make([]p16cProtectedCandidate, 0)
				for i, rd := range candidates {
					priority := protectedPriorities[rd.doc]
					reason := protectedReasons[rd.doc]
					if priority == 0 {
						priority, reason = p16FinalHeadingProtectionPriority(retrievalQuestion, rd.doc)
					}
					if priority < app.finalHeadingForcePriority {
						continue
					}

					text := strings.TrimSpace(extractContextText(rd.doc))
					if app.finalHeadingForceMinChars > 0 && len(text) < app.finalHeadingForceMinChars {
						debugf(
							"[P16-C:forceInclude] selected=false candidateRank=%d priority=%d reason=too_short chars=%d minChars=%d section=%q breadcrumb=%q tag=%q text=%q",
							i+1,
							priority,
							len(text),
							app.finalHeadingForceMinChars,
							docMetadataString(rd.doc, "section"),
							docMetadataString(rd.doc, "breadcrumb"),
							docMetadataString(rd.doc, "tag"),
							previewText(text, 120),
						)
						continue
					}

					protectedCandidates = append(protectedCandidates, p16cProtectedCandidate{
						rd:       rd,
						rank:     i + 1,
						priority: priority,
						reason:   reason,
						strength: p16ProtectedEvidenceStrength(retrievalQuestion, rd.doc),
						chars:    len(text),
					})
				}

				if len(protectedCandidates) > 0 {
					sort.SliceStable(protectedCandidates, func(i, j int) bool {
						if protectedCandidates[i].priority != protectedCandidates[j].priority {
							return protectedCandidates[i].priority > protectedCandidates[j].priority
						}
						if protectedCandidates[i].strength != protectedCandidates[j].strength {
							return protectedCandidates[i].strength > protectedCandidates[j].strength
						}
						if protectedCandidates[i].rd.score != protectedCandidates[j].rd.score {
							return protectedCandidates[i].rd.score > protectedCandidates[j].rd.score
						}
						return protectedCandidates[i].rank < protectedCandidates[j].rank
					})

					forced := 0
					for _, pc := range protectedCandidates {
						if kept >= maxResults || forced >= app.finalHeadingForceMax {
							break
						}
						quality := 0.75
						if pc.priority >= 3 {
							quality = 0.90
						}
						if addDoc(pc.rd.doc, pc.rd.score, pc.rank, false, "p16c_force_include_"+pc.reason, quality) {
							forced++
							log.Printf(
								"[P16-C:forceInclude] selected=true forcedRank=%d candidateRank=%d priority=%d reason=%s strength=%.2f rerank=%d chars=%d pages=%q section=%q breadcrumb=%q tag=%q",
								forced,
								pc.rank,
								pc.priority,
								pc.reason,
								pc.strength,
								pc.rd.score,
								pc.chars,
								docMetadataString(pc.rd.doc, "pages"),
								docMetadataString(pc.rd.doc, "section"),
								docMetadataString(pc.rd.doc, "breadcrumb"),
								docMetadataString(pc.rd.doc, "tag"),
							)
						}
					}
					log.Printf("[P16-C:forceInclude] enabled=%v candidates=%d eligible=%d forced=%d priorityThreshold=%d maxForced=%d minChars=%d", app.useFinalHeadingForceInclude, len(candidates), len(protectedCandidates), forced, app.finalHeadingForcePriority, app.finalHeadingForceMax, app.finalHeadingForceMinChars)
				} else {
					log.Printf("[P16-C:forceInclude] enabled=%v candidates=%d eligible=0 forced=0 priorityThreshold=%d maxForced=%d minChars=%d", app.useFinalHeadingForceInclude, len(candidates), app.finalHeadingForcePriority, app.finalHeadingForceMax, app.finalHeadingForceMinChars)
				}
			}

			// First pass: strict question-aware filtering.
			for i, rd := range candidates {
				if kept >= maxResults {
					break
				}

				text := extractContextText(rd.doc)
				matchedTerms, totalTerms, overlapRatio := questionTermOverlap(retrievalQuestion, text)
				metaOverlap := metadataMatchScore(retrievalQuestion, rd.doc)

				headingPriority := protectedPriorities[rd.doc]
				headingReason := protectedReasons[rd.doc]
				isStrongHeadingProtected := app.useFinalHeadingProtection && headingPriority >= 3
				p17BSiblingConflictReason := p17BSiblingConflictReasons[rd.doc]

				// P17-B2: once an exact/strong heading match has already been selected,
				// do not let partially matching sibling sections enter the final context.
				// Example: after Storage Hierarchy is included, skip Storage Models/PAX.
				if p17BSiblingConflictReason != "" && kept > 0 && !isStrongHeadingProtected {
					skippedLowQuality++
					debugf(
						"[finalContext:P17-B2] selected=false candidateRank=%d rerank=%d reason=p17b_sibling_conflict_%s qOverlap=%d/%d(%.2f) metaOverlap=%.2f pages=%q section=%q breadcrumb=%q tag=%q text=%q",
						i+1,
						rd.score,
						p17BSiblingConflictReason,
						matchedTerms,
						totalTerms,
						overlapRatio,
						metaOverlap,
						docMetadataString(rd.doc, "pages"),
						docMetadataString(rd.doc, "section"),
						docMetadataString(rd.doc, "breadcrumb"),
						docMetadataString(rd.doc, "tag"),
						previewText(text, 120),
					)
					continue
				}

				if app.useReranker && rd.score < minScore && !isStrongHeadingProtected {
					skippedBelowScore++
					debugf(
						"[finalContext:C5.4] selected=false candidateRank=%d rerank=%d reason=below_min_score qOverlap=%d/%d(%.2f) metaOverlap=%.2f pages=%q section=%q breadcrumb=%q tag=%q",
						i+1,
						rd.score,
						matchedTerms,
						totalTerms,
						overlapRatio,
						metaOverlap,
						docMetadataString(rd.doc, "pages"),
						docMetadataString(rd.doc, "section"),
						docMetadataString(rd.doc, "breadcrumb"),
						docMetadataString(rd.doc, "tag"),
					)
					continue
				}

				keep, reason, quality := shouldKeepCandidateC54(retrievalQuestion, rd.doc, text, rd.score, app.useReranker, kept)
				if !keep && isStrongHeadingProtected {
					keep = true
					reason = "p16b_heading_protected_" + headingReason
					if quality < 0.60 {
						quality = 0.60
					}
				}
				if !keep {
					skippedLowQuality++
					debugf(
						"[finalContext:C5.4] selected=false candidateRank=%d rerank=%d reason=%s quality=%.2f qOverlap=%d/%d(%.2f) metaOverlap=%.2f pages=%q section=%q breadcrumb=%q tag=%q text=%q",
						i+1,
						rd.score,
						reason,
						quality,
						matchedTerms,
						totalTerms,
						overlapRatio,
						metaOverlap,
						docMetadataString(rd.doc, "pages"),
						docMetadataString(rd.doc, "section"),
						docMetadataString(rd.doc, "breadcrumb"),
						docMetadataString(rd.doc, "tag"),
						previewText(text, 120),
					)
					continue
				}

				addDoc(rd.doc, rd.score, i+1, false, reason, quality)
			}

			// Second pass: conservative backfill.
			strictEvidenceQuestion := isAbbreviationQuestion(retrievalQuestion) || isExactValueQuestion(retrievalQuestion)
			if kept < 3 && !strictEvidenceQuestion {
				for i, rd := range candidates {
					if kept >= 3 || kept >= maxResults {
						break
					}
					if app.useReranker && rd.score < minScore {
						continue
					}

					text := extractContextText(rd.doc)
					quality := chunkQualityScore(retrievalQuestion, rd.doc, text, rd.score, app.useReranker)
					if app.useReranker && rd.score >= 7 {
						qualityBypassFallbacks++
						addDoc(rd.doc, rd.score, i+1, true, "backfill_high_rerank", quality)
					}
				}
			}

			if app.useReranker && kept == 0 && len(candidates) > 0 && !strictEvidenceQuestion {
				debugln("[docQA] reranker/quality filter removed everything — using top-3 fallback")
				for i, rd := range candidates {
					if i >= 3 {
						break
					}
					text := extractContextText(rd.doc)
					quality := chunkQualityScore(retrievalQuestion, rd.doc, text, rd.score, app.useReranker)
					addDoc(rd.doc, rd.score, i+1, true, "emergency_top3_fallback", quality)
				}
			}

			debugf(
				"[finalContext:C5.4] candidatesBeforeDedup=%d selectedAfterDedup=%d skippedDuplicates=%d skippedBelowScore=%d skippedLowQuality=%d skippedCharCap=%d qualityBackfills=%d finalContextChars=%d",
				len(candidates),
				kept,
				skippedDuplicates,
				skippedBelowScore,
				skippedLowQuality,
				skippedCharCap,
				qualityBypassFallbacks,
				contextText.Len(),
			)

			if kept == 0 && strictEvidenceQuestion {
				debugf("[finalContext:C5.4] strict evidence question has no supported final context; answer should be NOT IN DOCUMENT abbreviation=%v exactValue=%v", isAbbreviationQuestion(retrievalQuestion), isExactValueQuestion(retrievalQuestion))
			}

			debugf("[docQA] kept=%d docs for final answer; skippedDuplicates=%d skippedLowQuality=%d", kept, skippedDuplicates, skippedLowQuality)

			// P17-E2: narrow unsupported-question gate. It only activates for precise
			// fact/value questions that are likely to be hallucinated from related context.
			// Normal conceptual questions keep the P17-B2 behavior.
			if kept > 0 && app.useP17E2UnsupportedValidation {
				supported, reason := p17E2FinalSupportDecision(retrievalQuestion, contextText.String(), app)
				log.Printf("[P17-E2:finalSupport] supported=%v reason=%s", supported, reason)
				if !supported {
					noResult := unsupportedAnswerFor(question, retrievalQuestion)
					sendChunk(ctx, noResult)

					totalMs := float64(time.Since(flowStart).Milliseconds())
					otherOverheadMs := totalMs - hydeMs - retrievalMs - rerankMs - answerGenerationMs
					if otherOverheadMs < 0 {
						otherOverheadMs = 0
					}

					return qaResponse{
						Answer:             noResult,
						Sources:            make([]qaSource, 0),
						TotalMs:            totalMs,
						HydeMs:             hydeMs,
						RetrievalMs:        retrievalMs,
						RerankMs:           rerankMs,
						AnswerGenerationMs: answerGenerationMs,
						OtherOverheadMs:    otherOverheadMs,
					}, nil
				}
			}

			// ── Step 4: Final answer generation ──
			if kept == 0 {
				noResult := unsupportedAnswerFor(question, retrievalQuestion)
				sendChunk(ctx, noResult)

				totalMs := float64(time.Since(flowStart).Milliseconds())
				otherOverheadMs := totalMs - hydeMs - retrievalMs - rerankMs - answerGenerationMs
				if otherOverheadMs < 0 {
					otherOverheadMs = 0
				}

				return qaResponse{
					Answer:             noResult,
					Sources:            make([]qaSource, 0),
					TotalMs:            totalMs,
					HydeMs:             hydeMs,
					RetrievalMs:        retrievalMs,
					RerankMs:           rerankMs,
					AnswerGenerationMs: answerGenerationMs,
					OtherOverheadMs:    otherOverheadMs,
				}, nil
			}

			answerStart := time.Now()

			// P20-A: pass both the original user question and the standalone
			// conversational interpretation to the final QA prompt. The original
			// wording is preserved for a natural response, while retrievalQuestion
			// resolves ambiguous references such as "it", "that", or ordinal terms.
			resolvedQuestion := strings.TrimSpace(retrievalQuestion)
			if resolvedQuestion == "" {
				resolvedQuestion = question
			}
			if !strings.EqualFold(strings.TrimSpace(question), resolvedQuestion) {
				log.Printf("[P20-A:answer] original=%q resolved=%q", question, resolvedQuestion)
			}

			stream := app.qaPrompt.ExecuteStream(ctx,
				ai.WithModel(app.llm),
				ai.WithInput(map[string]any{
					"context":           contextText.String(),
					"question":          question,
					"resolved_question": resolvedQuestion,
				}),
			)

			for result, err := range stream {
				if err != nil {
					return qaResponse{}, err
				}

				if result.Done {
					answerGenerationMs = float64(time.Since(answerStart).Milliseconds())
					totalMs := float64(time.Since(flowStart).Milliseconds())
					otherOverheadMs := totalMs - hydeMs - retrievalMs - rerankMs - answerGenerationMs
					if otherOverheadMs < 0 {
						otherOverheadMs = 0
					}

					finalAnswer := strings.TrimSpace(result.Response.Text())
					finalSources := make([]qaSource, 0)

					if strings.EqualFold(finalAnswer, "NOT IN DOCUMENT") {
						finalAnswer = unsupportedAnswerFor(question, resolvedQuestion)
						log.Printf("[P21-E:notInDocument] modelRejectedContext=true sourcesHidden=%d followUpComparison=%v", len(sources), isLikelyComparisonFollowUp(question, resolvedQuestion))
					} else {
						finalAnswer, finalSources = validateAndNormalizeEvidenceCitations(finalAnswer, sources, evidence)
					}

					return qaResponse{
						Answer:             finalAnswer,
						Sources:            finalSources,
						TotalMs:            totalMs,
						HydeMs:             hydeMs,
						RetrievalMs:        retrievalMs,
						RerankMs:           rerankMs,
						AnswerGenerationMs: answerGenerationMs,
						OtherOverheadMs:    otherOverheadMs,
					}, nil
				}

				sendChunk(ctx, result.Chunk.Text())
			}

			return qaResponse{}, nil
		},
	)
}

func main() {
	app := setup()
	defer app.store.Close()

	indexPDFFlow := app.defineIndexPDFFlow()
	indexWebFlow := app.defineIndexWebFlow()
	crawlWebFlow := app.defineCrawlWebFlow()
	qaFlow := app.defineQAFlow()

	mux := http.NewServeMux()
	setupRoutes(mux, routeConfig{
		ctx:             app.ctx,
		store:           app.store,
		qaHandler:       genkit.Handler(qaFlow),
		indexPDFHandler: genkit.Handler(indexPDFFlow),
		indexWebHandler: genkit.Handler(indexWebFlow),
		crawlWebHandler: genkit.Handler(crawlWebFlow),
		runIndexPDF:     indexPDFFlow.Run,
	})

	log.Println("Starting server on http://localhost:3400")
	log.Println("Web-UI: http://localhost:3400")
	log.Println("Index PDF: POST http://localhost:3400/addPDFDocument")
	log.Println("Index Web: POST http://localhost:3400/addWebDocument")
	log.Println("Crawl Web: POST http://localhost:3400/crawlWeb")
	log.Println("QA flow:   POST http://localhost:3400/askPDF")
	log.Println("Keyword:   GET  http://localhost:3400/api/keyword-search?q=...&k=20&tags=ads")

	listenAddr := os.Getenv("LISTEN_ADDR")
	if listenAddr == "" {
		listenAddr = "127.0.0.1:3400"
	}
	log.Fatal(server.Start(app.ctx, listenAddr, mux))
}

func atoiOrDefault(str string, dval int) int {
	v, err := strconv.Atoi(str)
	if err != nil {
		return dval
	}
	return v
}

func atofOrDefault(str string, dval float64) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(str), 64)
	if err != nil {
		return dval
	}
	return v
}
