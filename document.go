package main

import (
	"fmt"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

type pageContent struct {
	pageNum  int
	slideNum int
	text     string
}

type paragraph struct {
	text       string
	pageNum    int
	slideNum   int
	heading    bool
	breadcrumb string
}

type chunk struct {
	text       string // focused clean chunk text for storage/display
	embedText  string // contextual enriched text used for embedding
	parentText string // for S1, same as text; for S2, expanded context

	pages      string
	section    string
	breadcrumb string
	url        string

	paraIndices []int
}

type indexedParaForChunk struct {
	paragraph
	idx int
}

func readPDFPages(path string) ([]pageContent, error) {
	out, err := exec.Command("pdftotext", "-layout", path, "-").Output()
	if err != nil {
		return nil, fmt.Errorf(
			"pdftotext failed for %q: %w\n"+
				"Make sure poppler is installed:\n"+
				"  macOS:  brew install poppler\n"+
				"  Ubuntu: apt install poppler-utils\n"+
				"  Alpine: apk add poppler-utils",
			path, err,
		)
	}

	rawPages := strings.Split(string(out), "\x0C")

	var pages []pageContent
	for i, text := range rawPages {
		trimmed := strings.TrimSpace(text)
		if trimmed == "" {
			continue
		}

		pages = append(pages, pageContent{
			pageNum:  i + 1,
			slideNum: extractLogicalSlideNumber(trimmed),
			text:     trimmed,
		})
	}

	if len(pages) == 0 {
		return pages, nil
	}

	boilerplate := buildAdaptiveBoilerplateSet(pages)
	pages = stripAdaptiveBoilerplate(pages, boilerplate)
	pages = deduplicateAnimatedSlidesGeneric(pages, boilerplate)

	return pages, nil
}

func buildAdaptiveBoilerplateSet(pages []pageContent) map[string]bool {
	lineFreq := make(map[string]int)

	for _, p := range pages {
		seenOnPage := make(map[string]bool)

		for _, line := range nonEmptyLines(p.text) {
			norm := normalizeBoilerplateCandidate(line)
			if norm == "" {
				continue
			}

			if looksLikeContentLine(norm) {
				continue
			}

			if !seenOnPage[norm] {
				lineFreq[norm]++
				seenOnPage[norm] = true
			}
		}
	}

	threshold := int(float64(len(pages)) * 0.35)
	if threshold < 3 {
		threshold = 3
	}

	out := make(map[string]bool)
	for line, freq := range lineFreq {
		if freq >= threshold {
			out[line] = true
		}
	}

	return out
}

func normalizeBoilerplateCandidate(line string) string {
	line = strings.TrimSpace(line)
	line = strings.Join(strings.Fields(line), " ")

	if line == "" {
		return ""
	}
	if len(line) > 120 {
		return ""
	}

	line = regexp.MustCompile(`(?i)\bpage\s+\d+\s*(/|of)\s*\d+\b`).ReplaceAllString(line, "page N of N")
	line = regexp.MustCompile(`(?i)\bslide\s+\d+\s*(/|of)\s*\d+\b`).ReplaceAllString(line, "slide N of N")
	line = regexp.MustCompile(`(?i)\bV\d+(?:\.\d+)*\s*:\s*\d+\s*/\s*\d+\b`).ReplaceAllString(line, "version slide N of N")
	line = regexp.MustCompile(`\b\d+\s*/\s*\d+\b`).ReplaceAllString(line, "N/N")

	return strings.ToLower(line)
}

func looksLikeContentLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	lower := strings.ToLower(trimmed)

	if trimmed == "" {
		return false
	}
	if strings.HasPrefix(trimmed, "•") || strings.HasPrefix(trimmed, "-") || strings.HasPrefix(trimmed, "*") || strings.HasPrefix(trimmed, "–") {
		return true
	}
	if isNumberedSection(trimmed) {
		return true
	}
	if strings.HasSuffix(trimmed, ":") && countLetters(trimmed) >= 4 {
		return true
	}

	contentWords := []string{
		"definition", "example", "advantage", "disadvantage", "problem", "solution",
		"summary", "overview", "architecture", "algorithm", "model", "method",
		"approach", "structure", "comparison", "application",
		"theorem", "proof", "lemma", "corollary", "proposition",
		"figure", "table", "equation", "formula", "notation",
		"introduction", "conclusion", "abstract", "related work",
		"evaluation", "experiment", "result", "discussion", "analysis",
		"dataset", "baseline", "metric", "performance", "implementation",
		"retrieval", "embedding", "encoding", "attention", "transformer",
		"network", "layer", "training", "inference", "loss", "gradient",
	}

	for _, w := range contentWords {
		if strings.Contains(lower, w) {
			return true
		}
	}

	return false
}

func stripAdaptiveBoilerplate(pages []pageContent, boilerplate map[string]bool) []pageContent {
	out := make([]pageContent, 0, len(pages))

	for _, p := range pages {
		var kept []string
		for _, line := range strings.Split(p.text, "\n") {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				kept = append(kept, "")
				continue
			}
			if looksLikeRepeatedFooter(trimmed) {
				continue
			}
			norm := normalizeBoilerplateCandidate(trimmed)
			if boilerplate != nil && boilerplate[norm] {
				continue
			}
			if hasTooMuchSpacing(trimmed) {
				kept = append(kept, linearizeTableLine(trimmed))
				continue
			}
			kept = append(kept, strings.TrimRight(line, " \t"))
		}
		p.text = strings.TrimSpace(strings.Join(kept, "\n"))
		if p.text != "" {
			out = append(out, p)
		}
	}
	return out
}

func linearizeTableLine(line string) string {
	re := regexp.MustCompile(`\s{3,}`)
	cols := re.Split(strings.TrimSpace(line), -1)
	var nonEmpty []string
	for _, c := range cols {
		c = strings.TrimSpace(c)
		if c != "" {
			nonEmpty = append(nonEmpty, c)
		}
	}
	if len(nonEmpty) <= 1 {
		return strings.TrimRight(line, " \t")
	}
	return "| " + strings.Join(nonEmpty, " | ") + " |"
}

func deduplicateAnimatedSlidesGeneric(pages []pageContent, boilerplate map[string]bool) []pageContent {
	if len(pages) < 2 {
		return pages
	}
	pages = deduplicateByLogicalSlideNumberGeneric(pages, boilerplate)
	pages = deduplicateByConsecutiveSimilarityGeneric(pages, boilerplate)
	return pages
}

func deduplicateByLogicalSlideNumberGeneric(pages []pageContent, boilerplate map[string]bool) []pageContent {
	if len(pages) < 2 {
		return pages
	}
	groups := findRepeatedConsecutiveSlideGroups(pages)
	if len(groups) < 2 {
		return pages
	}
	keep := make([]bool, len(pages))
	for i := range keep {
		keep[i] = true
	}
	safeGroups := 0
	for _, group := range groups {
		if len(group) < 2 || !looksLikeAnimationBuild(pages, group, boilerplate) {
			continue
		}
		safeGroups++
		bestIdx := group[len(group)-1]
		bestLen := meaningfulTextLength(pages[bestIdx].text, boilerplate)
		for _, idx := range group {
			l := meaningfulTextLength(pages[idx].text, boilerplate)
			if l > bestLen*12/10 {
				bestIdx = idx
				bestLen = l
			}
		}
		for _, idx := range group {
			if idx != bestIdx {
				keep[idx] = false
			}
		}
	}
	if safeGroups < 2 {
		return pages
	}
	var result []pageContent
	for i, p := range pages {
		if keep[i] {
			result = append(result, p)
		}
	}
	return result
}

func deduplicateByConsecutiveSimilarityGeneric(pages []pageContent, boilerplate map[string]bool) []pageContent {
	if len(pages) < 2 {
		return pages
	}
	const minContainment = 0.88
	const minJaccard = 0.74
	result := []pageContent{pages[0]}
	duplicatePairs := 0
	for i := 1; i < len(pages); i++ {
		last := &result[len(result)-1]
		current := pages[i]
		if last.slideNum > 0 && current.slideNum > 0 && last.slideNum != current.slideNum {
			result = append(result, current)
			continue
		}
		prevTokens := normalizeForOverlap(last.text, boilerplate)
		currTokens := normalizeForOverlap(current.text, boilerplate)
		if len(prevTokens) < 25 || len(currTokens) < 25 {
			result = append(result, current)
			continue
		}
		containment := tokenContainment(prevTokens, currTokens)
		jaccard := tokenJaccard(prevTokens, currTokens)
		if containment >= minContainment && jaccard >= minJaccard {
			duplicatePairs++
			if meaningfulTextLength(current.text, boilerplate) >= meaningfulTextLength(last.text, boilerplate)*9/10 {
				result[len(result)-1] = current
			}
			continue
		}
		result = append(result, current)
	}
	if duplicatePairs < 2 {
		return pages
	}
	return result
}

func findRepeatedConsecutiveSlideGroups(pages []pageContent) [][]int {
	var groups [][]int
	i := 0
	for i < len(pages) {
		if pages[i].slideNum == 0 {
			i++
			continue
		}
		j := i + 1
		for j < len(pages) && pages[j].slideNum == pages[i].slideNum {
			j++
		}
		if j-i >= 2 {
			group := make([]int, 0, j-i)
			for k := i; k < j; k++ {
				group = append(group, k)
			}
			groups = append(groups, group)
		}
		i = j
	}
	return groups
}

func looksLikeAnimationBuild(pages []pageContent, group []int, boilerplate map[string]bool) bool {
	if len(group) < 2 {
		return false
	}
	const minOverlap = 0.72
	for i := 0; i < len(group)-1; i++ {
		prev := normalizeForOverlap(pages[group[i]].text, boilerplate)
		next := normalizeForOverlap(pages[group[i+1]].text, boilerplate)
		if len(prev) < 20 || len(next) < 20 {
			continue
		}
		if tokenContainment(prev, next) < minOverlap {
			return false
		}
	}
	return true
}

func extractLogicalSlideNumber(text string) int {
	lines := nonEmptyLines(text)
	if len(lines) == 0 {
		return 0
	}
	start := len(lines) - 10
	if start < 0 {
		start = 0
	}
	footerLines := lines[start:]
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)\bV\d+(?:\.\d+)*\s*:\s*(\d{1,4})\s*/\s*(\d{1,4})\b`),
		regexp.MustCompile(`(?i)\bSlide\s+(\d{1,4})\s*(?:/|of)\s*(\d{1,4})\b`),
		regexp.MustCompile(`(?i)\bPage\s+(\d{1,4})\s*(?:/|of)\s*(\d{1,4})\b`),
		regexp.MustCompile(`(?i)^\s*(\d{1,4})\s*/\s*(\d{1,4})\s*$`),
	}
	for _, line := range footerLines {
		clean := strings.TrimSpace(line)
		for _, re := range patterns {
			m := re.FindStringSubmatch(clean)
			if len(m) >= 3 {
				current, err1 := strconv.Atoi(m[1])
				total, err2 := strconv.Atoi(m[2])
				if err1 == nil && err2 == nil && current > 0 && total > 0 && current <= total {
					return current
				}
			}
		}
	}
	return 0
}

func parseIntoParagraphs(pages []pageContent) []paragraph {
	var paras []paragraph
	for _, page := range pages {
		cleanPageText := cleanExtractedText(page.text)
		blocks := splitParagraphs(cleanPageText)
		for _, block := range blocks {
			trimmed := strings.TrimSpace(block)
			if trimmed == "" {
				continue
			}
			paras = append(paras, paragraph{
				text:     trimmed,
				pageNum:  page.pageNum,
				slideNum: page.slideNum,
				heading:  isHeading(trimmed),
			})
		}
	}
	suppressListLikeHeadings(paras)
	suppressDiagramLikeHeadings(paras)
	assignBreadcrumbs(paras)
	return paras
}

func cleanExtractedText(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.Map(func(r rune) rune {
		if r == '\n' || r == '\t' {
			return r
		}
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, text)
	lines := strings.Split(text, "\n")
	var kept []string
	for _, line := range lines {
		clean := strings.TrimSpace(line)
		if clean == "" {
			kept = append(kept, "")
			continue
		}
		if looksLikeRepeatedFooter(clean) {
			continue
		}
		if hasTooMuchSpacing(clean) {
			kept = append(kept, linearizeTableLine(clean))
			continue
		}
		kept = append(kept, strings.TrimRight(line, " \t"))
	}
	return strings.TrimSpace(strings.Join(kept, "\n"))
}

func splitParagraphs(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	for strings.Contains(text, "\n\n\n") {
		text = strings.ReplaceAll(text, "\n\n\n", "\n\n")
	}
	raw := strings.Split(text, "\n\n")
	var result []string
	for _, block := range raw {
		trimmed := strings.TrimSpace(block)
		if trimmed == "" || looksLikeRepeatedFooter(trimmed) {
			continue
		}
		result = append(result, splitDenseBlockSemantically(trimmed)...)
	}
	return result
}

func splitDenseBlockSemantically(block string) []string {
	lines := nonEmptyLines(block)
	if len(lines) <= 1 {
		return []string{strings.TrimSpace(block)}
	}
	if len(block) <= 700 && !containsStandaloneHeadingLine(lines) {
		return []string{strings.TrimSpace(block)}
	}
	var out []string
	var current []string
	currentKind := ""
	flush := func() {
		if len(current) == 0 {
			return
		}
		joined := strings.TrimSpace(strings.Join(current, "\n"))
		if joined != "" {
			out = append(out, joined)
		}
		current = nil
		currentKind = ""
	}
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || looksLikeRepeatedFooter(line) {
			continue
		}
		kind := semanticLineKind(line)
		if kind == "heading" {
			flush()
			out = append(out, line)
			continue
		}
		if len(current) == 0 {
			current = append(current, line)
			currentKind = kind
			continue
		}
		if currentKind == "table" && kind == "table" {
			current = append(current, line)
			continue
		}
		if currentKind == "list" && (kind == "list" || kind == "continuation") {
			current = append(current, line)
			continue
		}
		if kind == "continuation" && (currentKind == "normal" || currentKind == "list") {
			current = append(current, line)
			continue
		}
		if len(strings.Join(current, "\n"))+len(line) > 900 {
			flush()
			current = append(current, line)
			currentKind = kind
			continue
		}
		if currentKind != kind && kind != "continuation" {
			flush()
			current = append(current, line)
			currentKind = kind
			continue
		}
		current = append(current, line)
	}
	flush()
	if len(out) == 0 {
		return []string{strings.TrimSpace(block)}
	}
	return out
}

func semanticLineKind(line string) string {
	line = strings.TrimSpace(line)
	if line == "" {
		return "normal"
	}
	if isHeading(line) {
		return "heading"
	}
	if isTableLikeLine(line) || strings.HasPrefix(line, "| ") {
		return "table"
	}
	if isBulletOrNumberedListLine(line) {
		return "list"
	}
	r := []rune(line)
	if len(r) > 0 && (unicode.IsLower(r[0]) || strings.HasPrefix(line, ")") || strings.HasPrefix(line, ",")) {
		return "continuation"
	}
	return "normal"
}

func containsStandaloneHeadingLine(lines []string) bool {
	for _, line := range lines {
		if isHeading(line) {
			return true
		}
	}
	return false
}

func isBulletOrNumberedListLine(line string) bool {
	line = strings.TrimSpace(line)
	if line == "" {
		return false
	}
	prefixes := []string{"•", "-", "*", "–", "—", "▪", "◦"}
	for _, p := range prefixes {
		if strings.HasPrefix(line, p+" ") || strings.HasPrefix(line, p+"\t") {
			return true
		}
	}
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`^\(?[0-9]{1,3}\)?[.)]\s+\S+`),
		regexp.MustCompile(`^[A-Za-z][.)]\s+\S+`),
		regexp.MustCompile(`(?i)^(step|phase)\s+\d+\s*[:.-]\s+\S+`),
	}
	for _, re := range patterns {
		if re.MatchString(line) {
			return true
		}
	}
	return false
}

func assignBreadcrumbs(paras []paragraph) {
	var stack []string
	for i := range paras {
		if !paras[i].heading {
			if len(stack) > 0 {
				paras[i].breadcrumb = strings.Join(stack, " > ")
			}
			continue
		}
		h := normalizeHeadingText(paras[i].text)
		level := pdfHeadingLevel(h)
		if level <= 0 {
			level = 1
		}
		if level > 4 {
			level = 4
		}
		if len(stack) >= level {
			stack = stack[:level-1]
		}
		stack = append(stack, h)
		paras[i].breadcrumb = strings.Join(stack, " > ")
	}
}

func isHeading(text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	lines := nonEmptyLines(text)
	if len(lines) == 0 || len(lines) > 2 {
		return false
	}
	line := normalizeHeadingText(lines[0])
	if line == "" || len(line) > 120 {
		return false
	}
	if isBadHeadingCandidateGeneric(line) {
		return false
	}
	if len(line) < 100 && isNumberedSection(line) {
		return true
	}
	if len(line) <= 90 && looksLikeTitleHeading(line) {
		return true
	}
	if len(line) >= 4 && line == strings.ToUpper(line) && containsLetter(line) && countLetters(line) >= 4 && !mostlyShortTokens(line) {
		return true
	}
	if len(line) < 70 && strings.HasSuffix(line, ":") && countLetters(line) >= 5 && !isPromptOnlyLabel(line) {
		return true
	}
	return false
}

func isBadHeadingCandidateGeneric(line string) bool {
	line = strings.TrimSpace(line)
	if line == "" || looksLikeRepeatedFooter(line) || isFormulaLikeHeading(line) || isBrokenHeadingCandidate(line) || mostlyNumbersSymbolsOrShortTokens(line) || hasTooMuchSpacing(line) || countLetters(line) < 3 || isPromptOnlyLabel(line) || isTableLikeLine(line) || isDiagramLikeLabel(line) {
		return true
	}
	return false
}

func isFormulaLikeHeading(line string) bool {
	t := strings.TrimSpace(line)
	lower := strings.ToLower(t)
	if t == "" {
		return true
	}
	if strings.HasPrefix(t, "■") || strings.HasPrefix(t, "-") || strings.HasPrefix(t, "*") || strings.HasPrefix(t, "•") || strings.HasPrefix(t, "–") {
		return true
	}
	if strings.Contains(t, "=") || strings.Contains(lower, "cost(") {
		return true
	}
	symbols := 0
	for _, r := range t {
		if strings.ContainsRune("=+-*/·×÷≤≥<>∑∏√σπρλ→⇒", r) {
			symbols++
		}
	}
	return symbols >= 2 && len([]rune(t)) < 100
}

func isBrokenHeadingCandidate(line string) bool {
	t := strings.TrimSpace(line)
	if t == "" {
		return true
	}
	if strings.HasSuffix(t, "/") || strings.HasSuffix(t, "\\") || strings.HasSuffix(t, "|") {
		return true
	}
	openRound := strings.Count(t, "(")
	closeRound := strings.Count(t, ")")
	openSquare := strings.Count(t, "[")
	closeSquare := strings.Count(t, "]")
	if openRound != closeRound || openSquare != closeSquare {
		if len([]rune(t)) <= 120 {
			return true
		}
	}
	return false
}

func suppressListLikeHeadings(paras []paragraph) {
	for i := 0; i < len(paras); i++ {
		if !paras[i].heading || !isShortSimpleLabel(paras[i].text) {
			continue
		}
		runStart := i
		runEnd := i
		for runEnd+1 < len(paras) && paras[runEnd+1].pageNum == paras[runStart].pageNum && paras[runEnd+1].heading && isShortSimpleLabel(paras[runEnd+1].text) {
			runEnd++
		}
		if runEnd-runStart+1 >= 3 {
			for j := runStart; j <= runEnd; j++ {
				paras[j].heading = false
			}
		}
		i = runEnd
	}
}

func suppressDiagramLikeHeadings(paras []paragraph) {
	for i := range paras {
		if !paras[i].heading {
			continue
		}
		line := normalizeHeadingText(paras[i].text)
		if isDiagramLikeLabel(line) || isTableLikeLine(line) || isPromptOnlyLabel(line) || isFormulaLikeHeading(line) || isBrokenHeadingCandidate(line) || isSurroundedByShortLabels(paras, i) {
			paras[i].heading = false
		}
	}
}

func isShortSimpleLabel(text string) bool {
	line := normalizeHeadingText(text)
	if line == "" {
		return false
	}
	words := strings.Fields(line)
	if len(words) == 0 || len(words) > 4 {
		return false
	}
	if isNumberedSection(line) || strings.ContainsAny(line, ".:,;!?()[]{}") || mostlyNumbersSymbolsOrShortTokens(line) {
		return false
	}
	return countLetters(line) >= 4
}

func isSurroundedByShortLabels(paras []paragraph, idx int) bool {
	page := paras[idx].pageNum
	shorts := 0
	for j := idx - 2; j <= idx+2; j++ {
		if j < 0 || j >= len(paras) || j == idx || paras[j].pageNum != page {
			continue
		}
		if isShortSimpleLabel(paras[j].text) || isDiagramLikeLabel(normalizeHeadingText(paras[j].text)) {
			shorts++
		}
	}
	return shorts >= 2
}

func isPromptOnlyLabel(line string) bool {
	lower := strings.ToLower(strings.TrimSpace(line))
	labels := []string{"what", "what?", "why", "why?", "where", "where?", "problem", "problem:", "solution", "solution:", "idea", "idea:", "reason", "reason:", "example", "example:", "overview", "summary"}
	for _, x := range labels {
		if lower == x {
			return true
		}
	}
	return strings.HasSuffix(lower, "?") && len(strings.Fields(lower)) <= 4
}

func isTableLikeLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	lower := strings.ToLower(trimmed)
	if hasTooMuchSpacing(trimmed) {
		return true
	}
	tableWords := []string{"col0", "col1", "col2", "col3", "row", "rows", "column", "columns", "header", "footer"}
	hits := 0
	for _, w := range tableWords {
		if strings.Contains(lower, w) {
			hits++
		}
	}
	if hits >= 2 {
		return true
	}
	fields := strings.Fields(trimmed)
	if len(fields) >= 4 {
		short := 0
		for _, f := range fields {
			f = strings.Trim(f, ".,;:()[]{}<>")
			if len([]rune(f)) <= 3 || isNumericLike(f) {
				short++
			}
		}
		if float64(short)/float64(len(fields)) >= 0.75 {
			return true
		}
	}
	return false
}

func isDiagramLikeLabel(line string) bool {
	line = strings.TrimSpace(line)
	if line == "" {
		return true
	}
	fields := strings.Fields(line)
	if len(fields) == 1 {
		token := strings.Trim(fields[0], ".,;:()[]{}<>")
		if len([]rune(token)) <= 3 && token == strings.ToUpper(token) && containsLetter(token) {
			return true
		}
		if regexp.MustCompile(`(?i)^[a-z]{1,4}\d+$`).MatchString(token) {
			return true
		}
	}
	if len(fields) >= 2 {
		nodeLike := 0
		for _, f := range fields {
			clean := strings.Trim(f, ".,;:()[]{}<>")
			if regexp.MustCompile(`(?i)^[a-z]{1,4}\d+$`).MatchString(clean) || (len([]rune(clean)) <= 3 && clean == strings.ToUpper(clean) && containsLetter(clean)) {
				nodeLike++
			}
		}
		if float64(nodeLike)/float64(len(fields)) >= 0.60 {
			return true
		}
	}
	return false
}

func looksLikeTitleHeading(line string) bool {
	words := strings.Fields(line)
	if len(words) == 0 || len(words) > 12 || isDiagramLikeLabel(line) || isTableLikeLine(line) {
		return false
	}
	letterWords := 0
	titleishWords := 0
	for _, w := range words {
		clean := strings.Trim(w, ":-–—()[]{}.,;")
		if clean == "" {
			continue
		}
		if containsLetter(clean) {
			letterWords++
			r := []rune(clean)
			if len(r) > 0 && unicode.IsUpper(r[0]) {
				titleishWords++
			}
		}
	}
	if letterWords == 0 {
		return false
	}
	return float64(titleishWords)/float64(letterWords) >= 0.50
}

func mostlyNumbersSymbolsOrShortTokens(line string) bool {
	tokens := strings.Fields(line)
	if len(tokens) == 0 {
		return true
	}
	shortOrNumeric := 0
	for _, t := range tokens {
		clean := strings.Trim(t, ".,;:()[]{}<>")
		if clean == "" || isNumericLike(clean) || len([]rune(clean)) <= 2 {
			shortOrNumeric++
		}
	}
	return len(tokens) >= 3 && float64(shortOrNumeric)/float64(len(tokens)) >= 0.70
}

func mostlyShortTokens(line string) bool {
	tokens := strings.Fields(line)
	if len(tokens) == 0 {
		return true
	}
	short := 0
	for _, t := range tokens {
		t = strings.Trim(t, ".,;:()[]{}<>")
		if len([]rune(t)) <= 3 {
			short++
		}
	}
	return float64(short)/float64(len(tokens)) >= 0.75
}

func isNumericLike(s string) bool {
	hasDigit := false
	for _, r := range s {
		if unicode.IsDigit(r) {
			hasDigit = true
			continue
		}
		if r == '/' || r == '.' || r == ',' || r == '-' || r == '+' || r == '#' || r == '%' {
			continue
		}
		return false
	}
	return hasDigit
}

func hasTooMuchSpacing(line string) bool {
	return strings.Count(line, "   ") >= 2
}

func normalizeHeadingText(s string) string {
	lines := nonEmptyLines(s)
	if len(lines) == 0 {
		return ""
	}
	line := strings.TrimSpace(lines[0])
	line = strings.Join(strings.Fields(line), " ")
	return line
}

func pdfHeadingLevel(h string) int {
	h = strings.TrimSpace(h)
	re := regexp.MustCompile(`^(\d+(?:\.\d+)*)\s+`)
	m := re.FindStringSubmatch(h)
	if len(m) >= 2 {
		return strings.Count(m[1], ".") + 1
	}
	if strings.HasPrefix(strings.ToLower(h), "chapter ") {
		return 1
	}
	if h == strings.ToUpper(h) && containsLetter(h) && countLetters(h) >= 4 && len(strings.Fields(h)) <= 8 {
		return 1
	}
	words := strings.Fields(h)
	if len(words) <= 6 && looksLikeTitleHeading(h) {
		return 2
	}
	if strings.HasSuffix(h, ":") {
		return 3
	}
	return 2
}

func containsLetter(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

func countLetters(s string) int {
	n := 0
	for _, r := range s {
		if unicode.IsLetter(r) {
			n++
		}
	}
	return n
}

func isNumberedSection(line string) bool {
	line = strings.TrimSpace(line)
	if line == "" {
		return false
	}
	re := regexp.MustCompile(`^\d+(?:\.\d+)*\.?\s+[A-Za-zÄÖÜäöüß].+`)
	return re.MatchString(line)
}

func nonEmptyLines(text string) []string {
	raw := strings.Split(text, "\n")
	var lines []string
	for _, line := range raw {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			lines = append(lines, trimmed)
		}
	}
	return lines
}

func looksLikeRepeatedFooter(text string) bool {
	t := strings.TrimSpace(text)
	if t == "" {
		return false
	}
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)\bV\d+(?:\.\d+)*\s*:\s*\d{1,4}\s*/\s*\d{1,4}\b`),
		regexp.MustCompile(`(?i)\bSlide\s+\d{1,4}\s*(?:/|of)\s*\d{1,4}\b`),
		regexp.MustCompile(`(?i)\bPage\s+\d{1,4}\s*(?:/|of)\s*\d{1,4}\b`),
		regexp.MustCompile(`(?i)^\s*\d{1,4}\s*/\s*\d{1,4}\s*$`),
	}
	for _, re := range patterns {
		if re.MatchString(t) {
			return true
		}
	}
	return false
}

func normalizeForOverlap(text string, boilerplate map[string]bool) []string {
	text = strings.ToLower(text)
	lines := nonEmptyLines(text)
	var kept []string
	for _, line := range lines {
		if looksLikeRepeatedFooter(line) {
			continue
		}
		norm := normalizeBoilerplateCandidate(line)
		if boilerplate != nil && boilerplate[norm] {
			continue
		}
		kept = append(kept, line)
	}
	text = strings.Join(kept, " ")
	text = strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return r
		}
		return ' '
	}, text)
	tokens := strings.Fields(text)
	seen := make(map[string]bool)
	var result []string
	for _, tok := range tokens {
		if len(tok) <= 1 {
			continue
		}
		if !seen[tok] {
			seen[tok] = true
			result = append(result, tok)
		}
	}
	sort.Strings(result)
	return result
}

func tokenContainment(prev, next []string) float64 {
	if len(prev) == 0 || len(next) == 0 {
		return 0
	}
	nextSet := make(map[string]bool, len(next))
	for _, t := range next {
		nextSet[t] = true
	}
	matches := 0
	for _, t := range prev {
		if nextSet[t] {
			matches++
		}
	}
	return float64(matches) / float64(len(prev))
}

func tokenJaccard(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	setA := make(map[string]bool, len(a))
	setB := make(map[string]bool, len(b))
	for _, t := range a {
		setA[t] = true
	}
	for _, t := range b {
		setB[t] = true
	}
	intersection := 0
	for t := range setA {
		if setB[t] {
			intersection++
		}
	}
	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func meaningfulTextLength(text string, boilerplate map[string]bool) int {
	tokens := normalizeForOverlap(text, boilerplate)
	total := 0
	for _, t := range tokens {
		total += len(t)
	}
	return total
}

func buildContextualEmbeddingText(section, breadcrumb, pages string, ps []indexedParaForChunk, bodyText string) string {
	var b strings.Builder
	b.WriteString("Context for retrieval:\n")
	if breadcrumb != "" {
		b.WriteString("Document section path: ")
		b.WriteString(breadcrumb)
		b.WriteString("\n")
	}
	if section != "" {
		b.WriteString("Current section: ")
		b.WriteString(section)
		b.WriteString("\n")
	}
	if pages != "" {
		b.WriteString("Pages: ")
		b.WriteString(pages)
		b.WriteString("\n")
	}
	if len(ps) > 0 {
		firstSlide := ps[0].slideNum
		lastSlide := ps[len(ps)-1].slideNum
		if firstSlide > 0 && lastSlide > 0 {
			if firstSlide == lastSlide {
				b.WriteString("Logical slide: ")
				b.WriteString(strconv.Itoa(firstSlide))
				b.WriteString("\n")
			} else {
				b.WriteString("Logical slides: ")
				b.WriteString(strconv.Itoa(firstSlide))
				b.WriteString("-")
				b.WriteString(strconv.Itoa(lastSlide))
				b.WriteString("\n")
			}
		}
	}
	b.WriteString("\nChunk text:\n")
	b.WriteString(strings.TrimSpace(bodyText))
	return strings.TrimSpace(b.String())
}

// ─────────────────────────────────────────────
// P12-C5: Slide/Page-based retrieval + neighbor parent context
// ─────────────────────────────────────────────
//
// Retrieval unit:
//
//	One PDF page / logical slide becomes one focused chunk.
//
// Final answer context:
//
//	parentText contains previous page + current page + next page.
//
// Why:
//
//	C1 retrieved clean slide-level chunks, but sometimes missed cross-slide
//	explanations. C5 keeps C1's retrieval precision while giving the final
//	answer generator a small local context window.
//
// Important:
// - overlapSize is intentionally ignored in C5.
// - embedText stays focused on the current page only.
// - parentText is expanded to previous/current/next page.
// - pgstore.go and main.go should stay unchanged.
func buildChunks(paras []paragraph, maxSize, overlapSize int) []chunk {
	var chunks []chunk

	if len(paras) == 0 {
		return chunks
	}

	// Web paragraphs do not have physical PDF page numbers.
	// The current PDF chunker is page-based, so if we let web paragraphs
	// pass through it, all web content has pageNum=0 and collapses into
	// one huge chunk. Route page-less paragraphs to the web chunker.
	if isWebParagraphSet(paras) {
		return buildWebChunks(paras, maxSize, overlapSize)
	}

	type pageGroup struct {
		pageNum int
		paras   []indexedParaForChunk
	}

	var groups []pageGroup

	// Group consecutive paragraphs by physical page.
	// readPDFPages already performs conservative animation-slide deduplication.
	for i, p := range paras {
		if len(groups) == 0 || groups[len(groups)-1].pageNum != p.pageNum {
			groups = append(groups, pageGroup{
				pageNum: p.pageNum,
				paras: []indexedParaForChunk{
					{
						paragraph: p,
						idx:       i,
					},
				},
			})
			continue
		}

		groups[len(groups)-1].paras = append(groups[len(groups)-1].paras, indexedParaForChunk{
			paragraph: p,
			idx:       i,
		})
	}

	parasText := func(ps []indexedParaForChunk) string {
		var b strings.Builder

		for _, p := range ps {
			if strings.TrimSpace(p.text) == "" {
				continue
			}

			if b.Len() > 0 {
				b.WriteString("\n\n")
			}

			b.WriteString(strings.TrimSpace(p.text))
		}

		return strings.TrimSpace(b.String())
	}

	pageRange := func(ps []indexedParaForChunk) string {
		if len(ps) == 0 {
			return ""
		}

		start := ps[0].pageNum
		end := ps[len(ps)-1].pageNum

		if end > start {
			return fmt.Sprintf("%d-%d", start, end)
		}

		return fmt.Sprintf("%d", start)
	}

	inferSection := func(ps []indexedParaForChunk) string {
		for _, p := range ps {
			if p.heading {
				h := normalizeHeadingText(p.text)
				if h != "" {
					return h
				}
			}
		}

		for _, p := range ps {
			if p.breadcrumb != "" {
				parts := strings.Split(p.breadcrumb, " > ")
				if len(parts) > 0 {
					return strings.TrimSpace(parts[len(parts)-1])
				}
			}
		}

		return ""
	}

	inferBreadcrumb := func(ps []indexedParaForChunk, section string) string {
		for _, p := range ps {
			if p.breadcrumb != "" {
				return p.breadcrumb
			}
		}

		return section
	}

	displayText := func(section, breadcrumb, bodyText string) string {
		var b strings.Builder

		if breadcrumb != "" {
			b.WriteString("[")
			b.WriteString(breadcrumb)
			b.WriteString("]\n\n")
		}

		if section != "" {
			b.WriteString(section)
			b.WriteString("\n\n")
		}

		b.WriteString(strings.TrimSpace(bodyText))

		return strings.TrimSpace(b.String())
	}

	groupBodyText := func(g pageGroup) string {
		return parasText(g.paras)
	}

	neighborParentText := func(groupIndex int, currentSection, currentBreadcrumb string) string {
		if groupIndex < 0 || groupIndex >= len(groups) {
			return ""
		}

		start := groupIndex - 1
		if start < 0 {
			start = 0
		}

		end := groupIndex + 1
		if end >= len(groups) {
			end = len(groups) - 1
		}

		var b strings.Builder

		if currentBreadcrumb != "" {
			b.WriteString("[")
			b.WriteString(currentBreadcrumb)
			b.WriteString("]\n\n")
		}

		if currentSection != "" {
			b.WriteString(currentSection)
			b.WriteString("\n\n")
		}

		for gi := start; gi <= end; gi++ {
			body := groupBodyText(groups[gi])
			if body == "" {
				continue
			}

			pages := pageRange(groups[gi].paras)

			if b.Len() > 0 {
				b.WriteString("\n\n")
			}

			if gi == groupIndex {
				b.WriteString("Current page")
			} else if gi < groupIndex {
				b.WriteString("Previous page")
			} else {
				b.WriteString("Next page")
			}

			if pages != "" {
				b.WriteString(" ")
				b.WriteString(pages)
			}

			b.WriteString(":\n")
			b.WriteString(body)
		}

		return strings.TrimSpace(b.String())
	}

	for gi, g := range groups {
		if len(g.paras) == 0 {
			continue
		}

		bodyText := parasText(g.paras)
		if bodyText == "" {
			continue
		}

		pages := pageRange(g.paras)
		section := inferSection(g.paras)
		breadcrumb := inferBreadcrumb(g.paras, section)

		// Focused clean text for storage/display/retrieval.
		cleanText := displayText(section, breadcrumb, bodyText)

		// Keep embedding focused on the current page only.
		// This avoids S2-style broad embedding noise.
		contextualText := buildContextualEmbeddingText(
			section,
			breadcrumb,
			pages,
			g.paras,
			bodyText,
		)

		// Expanded local context for final QA only.
		parentText := neighborParentText(gi, section, breadcrumb)
		if parentText == "" {
			parentText = cleanText
		}

		indices := make([]int, len(g.paras))
		for i, p := range g.paras {
			indices[i] = p.idx
		}

		chunks = append(chunks, chunk{
			text:        cleanText,
			embedText:   contextualText,
			parentText:  parentText,
			pages:       pages,
			section:     section,
			breadcrumb:  breadcrumb,
			paraIndices: indices,
		})
	}

	return chunks
}

// isWebParagraphSet detects paragraph sets that came from web extraction.
// PDF paragraphs have physical page numbers starting at 1. Web paragraphs
// normally have pageNum=0, so they must not use the PDF page-based chunker.
func isWebParagraphSet(paras []paragraph) bool {
	if len(paras) == 0 {
		return false
	}

	for _, p := range paras {
		if p.pageNum != 0 {
			return false
		}
	}

	return true
}

// buildWebChunks keeps webpage chunking size/section-based instead of page-based.
// This is intentionally separate from buildChunks(), which is optimized for PDFs.
func buildWebChunks(paras []paragraph, maxSize, overlapSize int) []chunk {
	var chunks []chunk

	if len(paras) == 0 {
		return chunks
	}

	if maxSize <= 0 {
		maxSize = 1500
	}
	if overlapSize < 0 {
		overlapSize = 0
	}
	if overlapSize >= maxSize {
		overlapSize = maxSize / 4
	}

	parasText := func(ps []indexedParaForChunk) string {
		var b strings.Builder

		for _, p := range ps {
			if strings.TrimSpace(p.text) == "" {
				continue
			}

			if b.Len() > 0 {
				b.WriteString("\n\n")
			}

			b.WriteString(strings.TrimSpace(p.text))
		}

		return strings.TrimSpace(b.String())
	}

	inferSection := func(ps []indexedParaForChunk) string {
		for _, p := range ps {
			if p.heading {
				h := normalizeHeadingText(p.text)
				if h != "" {
					return h
				}
			}
		}

		for _, p := range ps {
			if p.breadcrumb != "" {
				parts := strings.Split(p.breadcrumb, " > ")
				if len(parts) > 0 {
					return strings.TrimSpace(parts[len(parts)-1])
				}
			}
		}

		return ""
	}

	inferBreadcrumb := func(ps []indexedParaForChunk, section string) string {
		for _, p := range ps {
			if p.breadcrumb != "" {
				return p.breadcrumb
			}
		}
		return section
	}

	displayText := func(section, breadcrumb, bodyText string) string {
		var b strings.Builder

		if breadcrumb != "" {
			b.WriteString("[")
			b.WriteString(breadcrumb)
			b.WriteString("]\n\n")
		}

		if section != "" {
			b.WriteString(section)
			b.WriteString("\n\n")
		}

		b.WriteString(strings.TrimSpace(bodyText))
		return strings.TrimSpace(b.String())
	}

	flushChunk := func(ps []indexedParaForChunk) {
		if len(ps) == 0 {
			return
		}

		bodyText := parasText(ps)
		if bodyText == "" {
			return
		}

		section := inferSection(ps)
		breadcrumb := inferBreadcrumb(ps, section)
		cleanText := displayText(section, breadcrumb, bodyText)

		contextualText := buildContextualEmbeddingText(
			section,
			breadcrumb,
			"",
			ps,
			bodyText,
		)

		indices := make([]int, len(ps))
		for i, p := range ps {
			indices[i] = p.idx
		}

		chunks = append(chunks, chunk{
			text:        cleanText,
			embedText:   contextualText,
			parentText:  cleanText,
			pages:       "",
			section:     section,
			breadcrumb:  breadcrumb,
			paraIndices: indices,
		})
	}

	var current []indexedParaForChunk
	currentLen := 0

	for i, p := range paras {
		text := strings.TrimSpace(p.text)
		if text == "" {
			continue
		}

		ip := indexedParaForChunk{
			paragraph: p,
			idx:       i,
		}

		addLen := len([]rune(text))
		if currentLen > 0 {
			addLen += 2
		}

		// Start a new chunk if the next paragraph would exceed the target size.
		if len(current) > 0 && currentLen+addLen > maxSize {
			flushChunk(current)

			// Optional overlap: keep the last paragraphs until overlapSize is reached.
			var overlap []indexedParaForChunk
			overlapLen := 0

			if overlapSize > 0 {
				for j := len(current) - 1; j >= 0; j-- {
					t := strings.TrimSpace(current[j].text)
					if t == "" {
						continue
					}

					tLen := len([]rune(t))
					if overlapLen > 0 {
						tLen += 2
					}

					if overlapLen+tLen > overlapSize {
						break
					}

					overlap = append([]indexedParaForChunk{current[j]}, overlap...)
					overlapLen += tLen
				}
			}

			current = overlap
			currentLen = overlapLen
		}

		current = append(current, ip)
		currentLen += addLen
	}

	flushChunk(current)

	return chunks
}
