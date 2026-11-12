package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
	"unicode"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// fetchWebPage downloads the HTML content of a URL.
func fetchWebPage(rawURL string) (string, error) {
	client := &http.Client{Timeout: 30 * time.Second}

	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}

	req.Header.Set("User-Agent", "ask-pdf/1.0 (web indexer)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d from %s", resp.StatusCode, rawURL)
	}

	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if contentType != "" &&
		!strings.Contains(contentType, "text/html") &&
		!strings.Contains(contentType, "application/xhtml+xml") {
		return "", fmt.Errorf("non-HTML content type %q from %s", contentType, rawURL)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("reading response: %w", err)
	}

	log.Printf("URL '%v' processed.\n", rawURL)
	return string(body), nil
}

// skipTags are HTML elements whose content should be completely ignored.
// These are generic structural boilerplate tags.
var skipTags = map[string]bool{
	"script":   true,
	"style":    true,
	"nav":      true,
	"footer":   true,
	"header":   true,
	"aside":    true,
	"form":     true,
	"noscript": true,
	"svg":      true,
	"canvas":   true,
	"iframe":   true,
	"object":   true,
	"embed":    true,
	"button":   true,
	"select":   true,
	"textarea": true,
}

// boilerplatePatterns are generic substrings in class/id/role/aria-label attributes
// that often indicate navigation, ads, cookie banners, and page chrome.
// Keep this generic; do not add website-specific terms.
var boilerplatePatterns = []string{
	"nav",
	"menu",
	"footer",
	"header",
	"sidebar",
	"side-bar",
	"cookie",
	"banner",
	"popup",
	"modal",
	"advertisement",
	"advert",
	"ad-",
	"ads",
	"breadcrumb",
	"breadcrumbs",
	"pagination",
	"pager",
	"search",
	"subscribe",
	"newsletter",
	"social",
	"share",
	"sharing",
	"related",
	"recommend",
	"promo",
	"promotion",
	"toolbar",
	"topbar",
	"navbar",
	"masthead",
	"skip",
	"login",
	"signin",
	"signup",
	"print",
	"download",
	"language",
	"locale",
	"comment",
	"comments",
	"feedback",
}

// headingLevel returns the heading level (1-6) for h1-h6 tags, or 0 for non-headings.
func headingLevel(tagName string) int {
	switch tagName {
	case "h1":
		return 1
	case "h2":
		return 2
	case "h3":
		return 3
	case "h4":
		return 4
	case "h5":
		return 5
	case "h6":
		return 6
	default:
		return 0
	}
}

// isBoilerplate checks if a node's class/id/role/aria-label contains generic boilerplate patterns.
func isBoilerplate(n *html.Node) bool {
	if n.Type != html.ElementNode {
		return false
	}

	for _, attr := range n.Attr {
		key := strings.ToLower(strings.TrimSpace(attr.Key))
		if key != "class" && key != "id" && key != "role" && key != "aria-label" {
			continue
		}

		val := strings.ToLower(attr.Val)
		for _, pat := range boilerplatePatterns {
			if strings.Contains(val, pat) {
				return true
			}
		}
	}

	return false
}

// findMainContent locates the best content root: <main>, largest <article>, or <body>.
func findMainContent(doc *html.Node) *html.Node {
	var body *html.Node
	var mainNode *html.Node
	var bestArticle *html.Node
	var bestArticleLen int

	var find func(*html.Node)
	find = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.DataAtom {
			case atom.Body:
				body = n
			case atom.Main:
				mainNode = n
			case atom.Article:
				textLen := nodeTextLen(n)
				if textLen > bestArticleLen {
					bestArticle = n
					bestArticleLen = textLen
				}
			}
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			find(c)
		}
	}

	find(doc)

	if mainNode != nil && nodeTextLen(mainNode) > 200 {
		return mainNode
	}

	if bestArticle != nil && bestArticleLen > 200 {
		return bestArticle
	}

	if body != nil {
		return body
	}

	return doc
}

// nodeTextLen returns the approximate text length under a node,
// ignoring skipped and boilerplate subtrees.
func nodeTextLen(n *html.Node) int {
	total := 0

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			if skipTags[n.Data] || isBoilerplate(n) {
				return
			}
		}

		if n.Type == html.TextNode {
			total += len(strings.TrimSpace(n.Data))
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	walk(n)
	return total
}

// headingStack tracks the current breadcrumb path through heading levels.
type headingStack struct {
	entries []headingEntry
}

type headingEntry struct {
	level int
	text  string
}

// push adds a heading. It pops any entries at the same or deeper level first.
func (hs *headingStack) push(level int, text string) {
	for len(hs.entries) > 0 && hs.entries[len(hs.entries)-1].level >= level {
		hs.entries = hs.entries[:len(hs.entries)-1]
	}

	hs.entries = append(hs.entries, headingEntry{level: level, text: text})
}

// breadcrumb returns the current path as "A > B > C".
func (hs *headingStack) breadcrumb() string {
	if len(hs.entries) == 0 {
		return ""
	}

	parts := make([]string, len(hs.entries))
	for i, e := range hs.entries {
		parts[i] = e.text
	}

	return strings.Join(parts, " > ")
}

// htmlToStructuredText parses HTML and returns structured paragraphs with
// headings detected from HTML tags, tables formatted, lists rendered,
// boilerplate removed, and low-value paragraphs filtered.
func htmlToStructuredText(r io.Reader) ([]paragraph, error) {
	doc, err := html.Parse(r)
	if err != nil {
		return nil, fmt.Errorf("parsing HTML: %w", err)
	}

	root := findMainContent(doc)
	hs := &headingStack{}
	var paras []paragraph

	seenParas := map[string]bool{}

	appendPara := func(text string, heading bool, breadcrumb string) {
		text = cleanWebExtractedText(text)
		if isLowValueWebText(text) {
			return
		}

		key := paragraphDedupKey(text)
		if key != "" && seenParas[key] {
			return
		}
		if key != "" {
			seenParas[key] = true
		}

		paras = append(paras, paragraph{
			text:       text,
			heading:    heading,
			breadcrumb: breadcrumb,
		})
	}

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			// Skip generic boilerplate subtrees.
			if skipTags[n.Data] || isBoilerplate(n) {
				return
			}

			// Handle headings.
			if level := headingLevel(n.Data); level > 0 {
				text := cleanWebExtractedText(extractPlainText(n))
				if text != "" && !isLowValueWebText(text) {
					hs.push(level, text)
					appendPara(text, true, hs.breadcrumb())
				}
				return
			}

			// Handle code/pre blocks as standalone paragraphs.
			if n.Data == "pre" || n.Data == "code" {
				codeText := cleanWebExtractedText(extractPlainText(n))
				if codeText != "" && !isLowValueWebText(codeText) {
					appendPara(codeText, false, hs.breadcrumb())
				}
				return
			}

			// Handle tables.
			if n.Data == "table" {
				tableText := cleanWebExtractedText(renderTable(n))
				if tableText != "" && !isLowValueWebText(tableText) {
					tableParagraphs := splitTableIfLarge(tableText, 1200)
					bc := hs.breadcrumb()
					for _, tp := range tableParagraphs {
						appendPara(tp, false, bc)
					}
				}
				return
			}

			// Handle lists.
			if n.Data == "ul" || n.Data == "ol" {
				listText := cleanWebExtractedText(renderList(n, 0))
				if listText != "" && !isLowValueWebText(listText) {
					appendPara(listText, false, hs.breadcrumb())
				}
				return
			}

			// Handle paragraph-like block elements.
			// Generic containers such as div/section/article/main are intentionally
			// not treated as paragraph blocks, because returning early on them can
			// skip useful nested paragraphs, lists, tables, and code blocks.
			if isBlockElement(n.Data) {
				text := cleanWebExtractedText(renderInlineContent(n))
				if text != "" && !isLowValueWebText(text) {
					appendPara(text, false, hs.breadcrumb())
					return
				}
			}
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	walk(root)

	return paras, nil
}

// isBlockElement returns true for HTML elements that should produce separate paragraphs.
//
// Important:
// Do not include generic containers like div, section, article, main, figure, or details.
// Those often contain nested useful content. If we treat them as standalone paragraphs
// and return early, we may accidentally skip nested paragraphs, lists, tables, or code blocks.
var blockElements = map[string]bool{
	"p":          true,
	"blockquote": true,
	"figcaption": true,
	"summary":    true,
	"dd":         true,
	"dt":         true,
}

func isBlockElement(tag string) bool {
	return blockElements[tag]
}

// extractPlainText extracts all text content from a node,
// ignoring boilerplate subtrees.
func extractPlainText(n *html.Node) string {
	var sb strings.Builder

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			if skipTags[n.Data] || isBoilerplate(n) {
				return
			}
		}

		if n.Type == html.TextNode {
			sb.WriteString(n.Data)
			sb.WriteString(" ")
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	walk(n)
	return normalizeSpaces(sb.String())
}

// renderInlineContent renders inline content of a block element.
// For B1, links keep only visible anchor text. Raw hrefs are not embedded
// because page URL is already stored as metadata and hrefs add embedding noise.
func renderInlineContent(n *html.Node) string {
	var sb strings.Builder

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			text := collapseWhitespace(n.Data)
			if text != "" {
				sb.WriteString(text)
				sb.WriteString(" ")
			}
			return
		}

		if n.Type == html.ElementNode {
			if skipTags[n.Data] || isBoilerplate(n) {
				return
			}

			// Do not recursively flatten nested block elements here.
			if isBlockElement(n.Data) ||
				headingLevel(n.Data) > 0 ||
				n.Data == "table" ||
				n.Data == "ul" ||
				n.Data == "ol" {
				return
			}

			// Links: keep only visible text, not href.
			if n.Data == "a" {
				linkText := cleanWebExtractedText(extractPlainText(n))
				if linkText != "" && !isLowValueAnchorText(linkText) {
					sb.WriteString(linkText)
					sb.WriteString(" ")
				}
				return
			}

			if n.Data == "br" {
				sb.WriteString("\n")
				return
			}
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walk(c)
	}

	return cleanWebExtractedText(sb.String())
}

// isLowValueAnchorText removes generic navigation-like anchor text.
func isLowValueAnchorText(s string) bool {
	t := strings.ToLower(strings.TrimSpace(s))

	if t == "" {
		return true
	}

	if len([]rune(t)) <= 1 {
		return true
	}

	lowValue := map[string]bool{
		"home":         true,
		"next":         true,
		"previous":     true,
		"prev":         true,
		"back":         true,
		"top":          true,
		"login":        true,
		"sign in":      true,
		"sign up":      true,
		"register":     true,
		"subscribe":    true,
		"share":        true,
		"print":        true,
		"download":     true,
		"menu":         true,
		"search":       true,
		"contact":      true,
		"privacy":      true,
		"terms":        true,
		"cookies":      true,
		"read more":    true,
		"learn more":   true,
		"click here":   true,
		"skip":         true,
		"skip to main": true,
	}

	return lowValue[t]
}

// renderTable converts an HTML table into pipe-formatted text.
func renderTable(tableNode *html.Node) string {
	var rows [][]string

	var walkTable func(*html.Node)
	walkTable = func(n *html.Node) {
		if n.Type == html.ElementNode {
			if skipTags[n.Data] || isBoilerplate(n) {
				return
			}
		}

		if n.Type == html.ElementNode && n.Data == "tr" {
			var cells []string

			for c := n.FirstChild; c != nil; c = c.NextSibling {
				if c.Type == html.ElementNode && (c.Data == "td" || c.Data == "th") {
					cellText := cleanWebExtractedText(renderInlineContent(c))
					cells = append(cells, cellText)
				}
			}

			if len(cells) > 0 {
				rows = append(rows, cells)
			}
			return
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walkTable(c)
		}
	}

	walkTable(tableNode)

	if len(rows) == 0 {
		return ""
	}

	maxCols := 0
	for _, row := range rows {
		if len(row) > maxCols {
			maxCols = len(row)
		}
	}

	for i := range rows {
		for len(rows[i]) < maxCols {
			rows[i] = append(rows[i], "")
		}
	}

	var sb strings.Builder

	for i, row := range rows {
		sb.WriteString("| ")
		sb.WriteString(strings.Join(row, " | "))
		sb.WriteString(" |")

		if i == 0 {
			sb.WriteString("\n| ")
			sep := make([]string, 0, len(row))
			for range row {
				sep = append(sep, "---")
			}
			sb.WriteString(strings.Join(sep, " | "))
			sb.WriteString(" |")
		}

		if i < len(rows)-1 {
			sb.WriteString("\n")
		}
	}

	return cleanWebExtractedText(sb.String())
}

// splitTableIfLarge splits a large table text into smaller parts,
// repeating the header line for each part.
func splitTableIfLarge(tableText string, maxSize int) []string {
	if len(tableText) <= maxSize {
		return []string{tableText}
	}

	lines := strings.Split(tableText, "\n")
	if len(lines) < 3 {
		return []string{tableText}
	}

	header := lines[0] + "\n" + lines[1]
	dataLines := lines[2:]

	var parts []string
	var current strings.Builder
	current.WriteString(header)

	for _, line := range dataLines {
		if current.Len()+len(line)+1 > maxSize && current.Len() > len(header) {
			parts = append(parts, current.String())
			current.Reset()
			current.WriteString(header)
		}

		current.WriteString("\n")
		current.WriteString(line)
	}

	if current.Len() > len(header) {
		parts = append(parts, current.String())
	}

	return parts
}

// renderList renders a ul or ol into formatted text with indentation.
func renderList(n *html.Node, depth int) string {
	if n.Type != html.ElementNode || (n.Data != "ul" && n.Data != "ol") {
		return ""
	}

	ordered := n.Data == "ol"
	indent := strings.Repeat("  ", depth)

	var sb strings.Builder
	itemNum := 0

	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if c.Type != html.ElementNode || c.Data != "li" {
			continue
		}

		if skipTags[c.Data] || isBoilerplate(c) {
			continue
		}

		itemNum++

		var inlineText strings.Builder
		var nestedLists []*html.Node

		for gc := c.FirstChild; gc != nil; gc = gc.NextSibling {
			if gc.Type == html.ElementNode && (gc.Data == "ul" || gc.Data == "ol") {
				nestedLists = append(nestedLists, gc)
				continue
			}

			if gc.Type == html.TextNode {
				inlineText.WriteString(collapseWhitespace(gc.Data))
				inlineText.WriteString(" ")
			} else if gc.Type == html.ElementNode {
				inlineText.WriteString(renderInlineContent(gc))
				inlineText.WriteString(" ")
			}
		}

		text := cleanWebExtractedText(inlineText.String())
		if text != "" && !isLowValueWebText(text) {
			if ordered {
				sb.WriteString(fmt.Sprintf("%s%d. %s\n", indent, itemNum, text))
			} else {
				sb.WriteString(fmt.Sprintf("%s- %s\n", indent, text))
			}
		}

		for _, nested := range nestedLists {
			sb.WriteString(renderList(nested, depth+1))
		}
	}

	return cleanWebExtractedText(sb.String())
}

// cleanWebExtractedText performs generic text cleanup for web extraction.
// Named specifically to avoid conflict with cleanExtractedText in document.go.
func cleanWebExtractedText(s string) string {
	s = strings.ReplaceAll(s, "\u00A0", " ")
	s = strings.ReplaceAll(s, "\u200B", "")
	s = strings.ReplaceAll(s, "\u200C", "")
	s = strings.ReplaceAll(s, "\u200D", "")

	// Remove common bracketed UI/citation fragments generically.
	// Keep this conservative so we do not remove useful technical content.
	s = removeBracketedNoise(s)

	return normalizeSpaces(s)
}

// removeBracketedNoise removes generic low-value bracketed fragments.
// It is intentionally conservative.
func removeBracketedNoise(s string) string {
	var out strings.Builder
	runes := []rune(s)

	for i := 0; i < len(runes); i++ {
		if runes[i] != '[' {
			out.WriteRune(runes[i])
			continue
		}

		j := i + 1
		for j < len(runes) && runes[j] != ']' && j-i <= 120 {
			j++
		}

		if j >= len(runes) || runes[j] != ']' {
			out.WriteRune(runes[i])
			continue
		}

		content := strings.ToLower(strings.TrimSpace(string(runes[i+1 : j])))

		if isGenericBracketNoise(content) {
			i = j
			continue
		}

		out.WriteString(string(runes[i : j+1]))
		i = j
	}

	return out.String()
}

// isGenericBracketNoise identifies generic low-value bracketed content.
// Avoid site-specific markers.
func isGenericBracketNoise(content string) bool {
	if content == "" {
		return true
	}

	// Numeric citation markers: [1], [23], [1, 2], [12-14]
	allCitationChars := true
	for _, r := range content {
		if !(unicode.IsDigit(r) || r == ',' || r == '-' || r == '–' || r == '—' || unicode.IsSpace(r)) {
			allCitationChars = false
			break
		}
	}

	if allCitationChars {
		return true
	}

	lowValue := []string{
		"citation needed",
		"needs update",
		"dead link",
		"permanent dead link",
		"clarification needed",
		"better source needed",
		"verification needed",
	}

	for _, p := range lowValue {
		if strings.Contains(content, p) {
			return true
		}
	}

	return false
}

// isLowValueWebText filters generic navigation, boilerplate, and very low-information text.
// It avoids website-specific rules.
func isLowValueWebText(text string) bool {
	t := strings.TrimSpace(text)
	if t == "" {
		return true
	}

	runes := []rune(t)
	runeLen := len(runes)

	// Very short text is usually a menu label or isolated UI text.
	// Keep code-like/API-like short text if it contains parentheses or operators.
	if runeLen < 25 && !looksTechnical(t) {
		return true
	}

	lower := strings.ToLower(t)

	// Generic navigation or UI phrases.
	badPhrases := []string{
		"accept cookies",
		"cookie settings",
		"privacy policy",
		"terms of service",
		"terms and conditions",
		"all rights reserved",
		"subscribe to",
		"sign up for",
		"log in",
		"sign in",
		"skip to content",
		"back to top",
		"share this",
		"follow us",
		"related articles",
		"recommended articles",
		"advertisement",
	}

	for _, p := range badPhrases {
		if strings.Contains(lower, p) && runeLen < 250 {
			return true
		}
	}

	// Too many raw URLs usually means references, link lists, or navigation.
	if strings.Count(lower, "http://")+strings.Count(lower, "https://") >= 4 {
		return true
	}

	// Low alphabetic/digit ratio usually means symbols/navigation/code fragments.
	// Preserve technical/code text.
	if !looksTechnical(t) {
		alnum := 0
		punct := 0

		for _, r := range runes {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				alnum++
			}
			if unicode.IsPunct(r) || unicode.IsSymbol(r) {
				punct++
			}
		}

		if runeLen > 0 {
			alnumRatio := float64(alnum) / float64(runeLen)
			punctRatio := float64(punct) / float64(runeLen)

			if alnumRatio < 0.35 {
				return true
			}
			if punctRatio > 0.55 {
				return true
			}
		}
	}

	// Menu-like text: many very short segments separated by bars/slashes.
	if isMenuLikeText(t) {
		return true
	}

	return false
}

// looksTechnical prevents useful code/API snippets from being filtered as low-value.
func looksTechnical(s string) bool {
	t := strings.TrimSpace(s)
	lower := strings.ToLower(t)

	technicalMarkers := []string{
		"()",
		"->",
		"=>",
		"==",
		":=",
		"::",
		"{",
		"}",
		"[",
		"]",
		"import ",
		"func ",
		"def ",
		"class ",
		"return ",
		"json.",
		"python -m",
		"$ ",
		">>>",
	}

	for _, marker := range technicalMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}

	return false
}

// isMenuLikeText detects generic navigation/menu fragments.
func isMenuLikeText(s string) bool {
	t := strings.TrimSpace(s)

	if len([]rune(t)) > 300 {
		return false
	}

	separators := strings.Count(t, "|") +
		strings.Count(t, " / ") +
		strings.Count(t, " · ") +
		strings.Count(t, " > ") +
		strings.Count(t, " › ")

	if separators < 3 {
		return false
	}

	fields := strings.FieldsFunc(t, func(r rune) bool {
		return r == '|' || r == '/' || r == '·' || r == '>' || r == '›'
	})

	if len(fields) < 4 {
		return false
	}

	short := 0
	for _, f := range fields {
		if len([]rune(strings.TrimSpace(f))) <= 18 {
			short++
		}
	}

	return float64(short)/float64(len(fields)) > 0.75
}

// paragraphDedupKey creates a generic near-dedup key for extracted paragraphs.
// This handles repeated nav blocks and overlap-like repetition.
func paragraphDedupKey(s string) string {
	s = strings.ToLower(cleanWebExtractedText(s))

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

	// Use first 500 chars so near-identical long paragraphs dedup.
	if len([]rune(key)) > 500 {
		key = string([]rune(key)[:500])
	}

	return key
}

// getAttr returns the value of the named attribute, or empty string.
func getAttr(n *html.Node, name string) string {
	for _, a := range n.Attr {
		if a.Key == name {
			return a.Val
		}
	}

	return ""
}

// collapseWhitespace replaces runs of whitespace with single spaces.
func collapseWhitespace(s string) string {
	var sb strings.Builder
	inSpace := false

	for _, r := range s {
		if unicode.IsSpace(r) {
			if !inSpace {
				sb.WriteByte(' ')
				inSpace = true
			}
		} else {
			sb.WriteRune(r)
			inSpace = false
		}
	}

	return sb.String()
}

// normalizeSpaces trims and collapses whitespace into single spaces,
// while preserving newlines for table/list/code readability.
func normalizeSpaces(s string) string {
	s = strings.TrimSpace(s)

	var sb strings.Builder
	inSpace := false
	lastWasNewline := false

	for _, r := range s {
		if r == '\n' || r == '\r' {
			if !lastWasNewline {
				sb.WriteByte('\n')
			}
			inSpace = false
			lastWasNewline = true
			continue
		}

		if unicode.IsSpace(r) {
			if !inSpace && !lastWasNewline {
				sb.WriteByte(' ')
				inSpace = true
			}
			continue
		}

		sb.WriteRune(r)
		inSpace = false
		lastWasNewline = false
	}

	return strings.TrimSpace(sb.String())
}
