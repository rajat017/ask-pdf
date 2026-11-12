package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// registerEvaluators defines custom evaluators for RAG quality, compliance, and timing.
func registerEvaluators(g *genkit.Genkit, modelName string) {
	defineRAGQualityEvaluator(g, modelName)

	defineTimingEvaluator(g, "totalTime", "total_ms", "Total QA Time", "End-to-end response time in milliseconds.")
	defineTimingEvaluator(g, "hydeTime", "hyde_ms", "HyDE Time", "HyDE generation time in milliseconds.")
	defineTimingEvaluator(g, "retrievalTime", "retrieval_ms", "Retrieval Time", "Combined retrieval time in milliseconds.")
	defineTimingEvaluator(g, "rerankTime", "rerank_ms", "Rerank Time", "LLM reranking time in milliseconds.")
	defineTimingEvaluator(g, "answerGenerationTime", "answer_generation_ms", "Answer Generation Time", "Final answer generation time in milliseconds.")
	defineTimingEvaluator(g, "otherOverheadTime", "other_overhead_ms", "Other Overhead Time", "Remaining pipeline overhead in milliseconds.")
}

func defineTimingEvaluator(g *genkit.Genkit, metricID, fieldName, displayName, definition string) {
	genkit.DefineEvaluator(g, api.NewName("custom", metricID),
		&ai.EvaluatorOptions{
			DisplayName: displayName,
			Definition:  definition,
		},
		func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
			outputMap, err := extractOutputMap(req.Input.Output)
			if err != nil {
				return &ai.EvaluatorCallbackResponse{
					TestCaseId: req.Input.TestCaseId,
					Evaluation: []ai.Score{{
						Id:    metricID,
						Score: 0.0,
						Details: map[string]any{
							"reason": "could not parse output for timing extraction",
							"field":  fieldName,
							"unit":   "ms",
						},
					}},
				}, nil
			}

			val := toFloat64(outputMap[fieldName])

			return &ai.EvaluatorCallbackResponse{
				TestCaseId: req.Input.TestCaseId,
				Evaluation: []ai.Score{{
					Id:    metricID,
					Score: val,
					Details: map[string]any{
						"unit":  "ms",
						"field": fieldName,
					},
				}},
			}, nil
		},
	)
}

func defineRAGQualityEvaluator(g *genkit.Genkit, modelName string) {
	genkit.DefineEvaluator(g, api.NewName("custom", "ragQuality"),
		&ai.EvaluatorOptions{
			DisplayName: "RAG Quality",
			Definition:  "Combined evaluator for answer relevancy, faithfulness, correctness, completeness, overall quality, and response compliance. Score 0.0–1.0.",
		},
		func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
			answer, err := extractAnswer(req.Input.Output)
			if err != nil {
				return nil, err
			}

			answer = normalize(answer)
			question := extractQuestion(req)
			reference := extractReference(req)
			contextStr := formatContext(req.Input.Context)

			if isGarbageAnswer(answer) || isQuestionRepetition(question, answer) {
				return ragQualityResponse(req.Input.TestCaseId, ragQualityJudgment{
					AnswerRelevancy:    0.0,
					Faithfulness:       0.0,
					Correctness:        0.0,
					Completeness:       0.0,
					Overall:            0.0,
					LanguageCompliance: 0.0,
					CitationCompliance: 0.0,
					FormatCompliance:   0.0,
					ComplianceOverall:  0.0,
					Reason:             "answer is empty, garbled, or only repeats the question",
					Answerable:         !isNotInDocument(reference),
				}, ""), nil
			}

			// Deterministic handling for NOT_IN_DOCUMENT / NOT_IN_CONTEXT test cases.
			if isNotInDocument(reference) {
				score, reason := scoreUnanswerableBehavior(answer)

				return ragQualityResponse(req.Input.TestCaseId, ragQualityJudgment{
					AnswerRelevancy:    score,
					Faithfulness:       score,
					Correctness:        score,
					Completeness:       score,
					Overall:            score,
					LanguageCompliance: heuristicLanguageCompliance(question, answer),
					CitationCompliance: 1.0,
					FormatCompliance:   heuristicFormatCompliance(answer),
					ComplianceOverall:  average3(heuristicLanguageCompliance(question, answer), 1.0, heuristicFormatCompliance(answer)),
					Reason:             reason,
					Answerable:         false,
				}, ""), nil
			}

			if strings.TrimSpace(reference) == "" {
				return ragQualityResponse(req.Input.TestCaseId, ragQualityJudgment{
					AnswerRelevancy:    0.0,
					Faithfulness:       0.0,
					Correctness:        0.0,
					Completeness:       0.0,
					Overall:            0.0,
					LanguageCompliance: 0.0,
					CitationCompliance: 0.0,
					FormatCompliance:   0.0,
					ComplianceOverall:  0.0,
					Reason:             "no reference answer available",
					Answerable:         true,
				}, ""), nil
			}

			if strings.TrimSpace(contextStr) == "" {
				return ragQualityResponse(req.Input.TestCaseId, ragQualityJudgment{
					AnswerRelevancy:    0.0,
					Faithfulness:       0.0,
					Correctness:        0.0,
					Completeness:       0.0,
					Overall:            0.0,
					LanguageCompliance: 0.0,
					CitationCompliance: 0.0,
					FormatCompliance:   0.0,
					ComplianceOverall:  0.0,
					Reason:             "no retrieved context available",
					Answerable:         true,
				}, ""), nil
			}

			prompt := fmt.Sprintf(`You are a strict but fair evaluator for a Retrieval-Augmented Generation system.

Evaluate the GENERATED ANSWER using the QUESTION, REFERENCE ANSWER, and RETRIEVED CONTEXT.

QUESTION:
%s

REFERENCE ANSWER:
%s

RETRIEVED CONTEXT:
%s

GENERATED ANSWER:
%s

Score each metric from 0.0 to 1.0.

QUALITY METRICS:
- answer_relevancy: Does the answer address the exact question?
- faithfulness: Are the factual claims supported by the retrieved context?
- correctness: Is the answer semantically correct compared with the reference answer?
- completeness: Does the answer cover the required parts of the reference answer?

COMPLIANCE METRICS:
- language_compliance: Is the answer in the same language as the question?
- citation_compliance: Does the answer use inline citations based only on available source metadata?
- format_compliance: Is the answer concise, readable, and free of irrelevant extra content, metadata dumps, follow-up questions, or corrupted tokens?

STRICT EVALUATION RULES:
- Penalize wrong-neighbor answers. If the answer uses a related but different concept than the one asked, reduce correctness strongly.
- Penalize formula errors strongly. If the question asks for a formula and the answer gives a related but wrong formula, correctness should be low.
- Penalize list errors. If the question asks for a specific list and the answer gives a different list, completeness and correctness should be low.
- Penalize unsupported additions. Claims not supported by retrieved context should lower faithfulness.
- Do not let missing or imperfect citations alone destroy semantic quality. Reduce citation_compliance instead.
- If the answer is semantically correct but too verbose, keep quality scores reasonable and reduce format_compliance.
- If the answer is correct but citation formatting is imperfect, reduce citation_compliance, not correctness.
- overall should normally be the average of answer_relevancy, faithfulness, correctness, and completeness.
- compliance_overall should be the average of language_compliance, citation_compliance, and format_compliance.
- Return valid JSON only. Do not include markdown.

JSON keys to return:
{
  "answer_relevancy": number,
  "faithfulness": number,
  "correctness": number,
  "completeness": number,
  "overall": number,
  "language_compliance": number,
  "citation_compliance": number,
  "format_compliance": number,
  "compliance_overall": number,
  "reason": "short reason",
  "answerable": true,
  "missing_key_points": [],
  "incorrect_claims": [],
  "unsupported_claims": [],
  "citation_issues": [],
  "format_issues": [],
  "language_issue": "",
  "has_contradiction": false,
  "has_unsupported_addition": false,
  "uses_wrong_conceptual_schema": false
}`, question, reference, contextStr, answer)

			resp, err := genkit.Generate(ctx, g,
				ai.WithModelName(modelName),
				ai.WithPrompt(prompt),
			)
			if err != nil {
				return nil, fmt.Errorf("ragQuality judge failed: %w", err)
			}

			rawJudge := resp.Text()
			log.Printf("[ragQuality raw judge response][%s] %s", req.Input.TestCaseId, truncate(rawJudge, 1200))

			judgment, parseOK := parseRAGQualityJudgeResponse(rawJudge)

			if judgment.Overall <= 0 && hasAnyQualityScore(judgment) {
				judgment.Overall = average4(judgment.AnswerRelevancy, judgment.Faithfulness, judgment.Correctness, judgment.Completeness)
			}
			if judgment.ComplianceOverall <= 0 && hasAnyComplianceScore(judgment) {
				judgment.ComplianceOverall = average3(judgment.LanguageCompliance, judgment.CitationCompliance, judgment.FormatCompliance)
			}

			if strings.TrimSpace(judgment.Reason) == "" {
				judgment.Reason = "combined RAG quality judgment"
			}
			judgment.Answerable = true

			if !parseOK || suspiciousAllZero(judgment, answer, reference) {
				fallback := heuristicRAGQuality(question, reference, contextStr, answer)
				fallback.Reason = "heuristic fallback used because ragQuality judge returned suspicious all-zero or unparsable output"
				fallback.Answerable = true
				fallback.FormatIssues = append(fallback.FormatIssues, "raw judge output was suspicious or unparsable")
				return ragQualityResponse(req.Input.TestCaseId, fallback, rawJudge), nil
			}

			return ragQualityResponse(req.Input.TestCaseId, judgment, rawJudge), nil
		},
	)
}

type ragQualityJudgment struct {
	AnswerRelevancy float64 `json:"answer_relevancy"`
	Faithfulness    float64 `json:"faithfulness"`
	Correctness     float64 `json:"correctness"`
	Completeness    float64 `json:"completeness"`
	Overall         float64 `json:"overall"`

	LanguageCompliance float64 `json:"language_compliance"`
	CitationCompliance float64 `json:"citation_compliance"`
	FormatCompliance   float64 `json:"format_compliance"`
	ComplianceOverall  float64 `json:"compliance_overall"`

	Reason                    string   `json:"reason"`
	Answerable                bool     `json:"answerable"`
	MissingKeyPoints          []string `json:"missing_key_points"`
	IncorrectClaims           []string `json:"incorrect_claims"`
	UnsupportedClaims         []string `json:"unsupported_claims"`
	CitationIssues            []string `json:"citation_issues"`
	FormatIssues              []string `json:"format_issues"`
	LanguageIssue             string   `json:"language_issue"`
	HasContradiction          bool     `json:"has_contradiction"`
	HasUnsupportedAddition    bool     `json:"has_unsupported_addition"`
	UsesWrongConceptualSchema bool     `json:"uses_wrong_conceptual_schema"`
}

func ragQualityResponse(testCaseID string, j ragQualityJudgment, rawJudge string) *ai.EvaluatorCallbackResponse {
	commonDetails := map[string]any{
		"reason":                       j.Reason,
		"answerable":                   j.Answerable,
		"missing_key_points":           j.MissingKeyPoints,
		"incorrect_claims":             j.IncorrectClaims,
		"unsupported_claims":           j.UnsupportedClaims,
		"citation_issues":              j.CitationIssues,
		"format_issues":                j.FormatIssues,
		"language_issue":               j.LanguageIssue,
		"has_contradiction":            j.HasContradiction,
		"has_unsupported_addition":     j.HasUnsupportedAddition,
		"uses_wrong_conceptual_schema": j.UsesWrongConceptualSchema,
		"evaluator_mode":               "strict_combined_rag_quality_v2",
	}

	if strings.TrimSpace(rawJudge) != "" {
		commonDetails["raw_judge_response_preview"] = truncate(rawJudge, 800)
	}

	return &ai.EvaluatorCallbackResponse{
		TestCaseId: testCaseID,
		Evaluation: []ai.Score{
			{Id: "answerRelevancy", Score: clampScore(j.AnswerRelevancy), Details: cloneDetails(commonDetails, map[string]any{"metric": "answerRelevancy", "group": "quality"})},
			{Id: "faithfulness", Score: clampScore(j.Faithfulness), Details: cloneDetails(commonDetails, map[string]any{"metric": "faithfulness", "group": "quality"})},
			{Id: "correctness", Score: clampScore(j.Correctness), Details: cloneDetails(commonDetails, map[string]any{"metric": "correctness", "group": "quality"})},
			{Id: "completeness", Score: clampScore(j.Completeness), Details: cloneDetails(commonDetails, map[string]any{"metric": "completeness", "group": "quality"})},
			{Id: "overall", Score: clampScore(j.Overall), Details: cloneDetails(commonDetails, map[string]any{"metric": "overall", "group": "quality"})},
			{Id: "languageCompliance", Score: clampScore(j.LanguageCompliance), Details: cloneDetails(commonDetails, map[string]any{"metric": "languageCompliance", "group": "compliance"})},
			{Id: "citationCompliance", Score: clampScore(j.CitationCompliance), Details: cloneDetails(commonDetails, map[string]any{"metric": "citationCompliance", "group": "compliance"})},
			{Id: "formatCompliance", Score: clampScore(j.FormatCompliance), Details: cloneDetails(commonDetails, map[string]any{"metric": "formatCompliance", "group": "compliance"})},
			{Id: "complianceOverall", Score: clampScore(j.ComplianceOverall), Details: cloneDetails(commonDetails, map[string]any{"metric": "complianceOverall", "group": "compliance"})},
		},
	}
}

func cloneDetails(base map[string]any, extra map[string]any) map[string]any {
	out := make(map[string]any, len(base)+len(extra))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range extra {
		out[k] = v
	}
	return out
}

func parseRAGQualityJudgeResponse(text string) (ragQualityJudgment, bool) {
	text = extractLikelyJSON(text)

	var raw map[string]any
	if err := json.Unmarshal([]byte(text), &raw); err == nil {
		reason := toString(raw["reason"])
		if strings.TrimSpace(reason) == "" {
			reason = "judge returned no reason"
		}

		j := ragQualityJudgment{
			AnswerRelevancy: clampScore(toFloat64(raw["answer_relevancy"])),
			Faithfulness:    clampScore(toFloat64(raw["faithfulness"])),
			Correctness:     clampScore(toFloat64(raw["correctness"])),
			Completeness:    clampScore(toFloat64(raw["completeness"])),
			Overall:         clampScore(toFloat64(raw["overall"])),

			LanguageCompliance: clampScore(toFloat64(raw["language_compliance"])),
			CitationCompliance: clampScore(toFloat64(raw["citation_compliance"])),
			FormatCompliance:   clampScore(toFloat64(raw["format_compliance"])),
			ComplianceOverall:  clampScore(toFloat64(raw["compliance_overall"])),

			Reason:                    reason,
			Answerable:                toBool(raw["answerable"]),
			MissingKeyPoints:          toStringSlice(raw["missing_key_points"]),
			IncorrectClaims:           toStringSlice(raw["incorrect_claims"]),
			UnsupportedClaims:         toStringSlice(raw["unsupported_claims"]),
			CitationIssues:            toStringSlice(raw["citation_issues"]),
			FormatIssues:              toStringSlice(raw["format_issues"]),
			LanguageIssue:             toString(raw["language_issue"]),
			HasContradiction:          toBool(raw["has_contradiction"]),
			HasUnsupportedAddition:    toBool(raw["has_unsupported_addition"]),
			UsesWrongConceptualSchema: toBool(raw["uses_wrong_conceptual_schema"]),
		}

		return j, true
	}

	score := firstScoreInText(text)
	return ragQualityJudgment{
		AnswerRelevancy:    score,
		Faithfulness:       score,
		Correctness:        score,
		Completeness:       score,
		Overall:            score,
		LanguageCompliance: 0.0,
		CitationCompliance: 0.0,
		FormatCompliance:   0.0,
		ComplianceOverall:  0.0,
		Reason:             "could not parse judge response cleanly: " + truncate(text, 300),
		Answerable:         true,
	}, false
}

func suspiciousAllZero(j ragQualityJudgment, answer, reference string) bool {
	if isGarbageAnswer(answer) || isNotInDocument(reference) {
		return false
	}

	return j.AnswerRelevancy == 0 &&
		j.Faithfulness == 0 &&
		j.Correctness == 0 &&
		j.Completeness == 0 &&
		j.Overall == 0 &&
		len(strings.Fields(answer)) >= 3
}

func heuristicRAGQuality(question, reference, contextStr, answer string) ragQualityJudgment {
	refOverlap := lexicalOverlap(reference, answer)
	ctxOverlap := lexicalOverlap(contextStr, answer)
	qOverlap := lexicalOverlap(question, answer)

	quality := 0.20

	switch {
	case refOverlap >= 0.75:
		quality = 0.90
	case refOverlap >= 0.55:
		quality = 0.78
	case refOverlap >= 0.38:
		quality = 0.62
	case refOverlap >= 0.22:
		quality = 0.48
	case ctxOverlap >= 0.35:
		quality = 0.50
	case ctxOverlap >= 0.20:
		quality = 0.40
	}

	answerLower := strings.ToLower(answer)

	if containsInsufficientAnswer(answerLower) && !isNotInDocument(reference) {
		quality = math.Min(quality, 0.30)
	}

	if qOverlap >= 0.85 && len(strings.Fields(answer)) <= len(strings.Fields(question))+12 {
		quality = math.Min(quality, 0.30)
	}

	if containsKeyTerms(reference, answer) && quality < 0.65 {
		quality = 0.65
	}

	langScore := heuristicLanguageCompliance(question, answer)
	citationScore := heuristicCitationCompliance(answer)
	formatScore := heuristicFormatCompliance(answer)

	correctness := quality
	completeness := quality

	if len(strings.Fields(answer)) < 8 && len(strings.Fields(reference)) > 15 {
		completeness = math.Min(completeness, 0.75)
	}

	j := ragQualityJudgment{
		AnswerRelevancy:    clampScore(math.Max(quality, 0.35)),
		Faithfulness:       clampScore(math.Max(math.Min(quality, ctxOverlap+0.30), 0.30)),
		Correctness:        clampScore(correctness),
		Completeness:       clampScore(completeness),
		LanguageCompliance: langScore,
		CitationCompliance: citationScore,
		FormatCompliance:   formatScore,
		ComplianceOverall:  average3(langScore, citationScore, formatScore),
		Answerable:         true,
	}

	j.Overall = average4(j.AnswerRelevancy, j.Faithfulness, j.Correctness, j.Completeness)
	return j
}

func lexicalOverlap(a, b string) float64 {
	aTerms := importantTerms(a)
	bTerms := importantTerms(b)

	if len(aTerms) == 0 || len(bTerms) == 0 {
		return 0
	}

	match := 0
	for t := range aTerms {
		if _, ok := bTerms[t]; ok {
			match++
		}
	}

	return float64(match) / float64(len(aTerms))
}

func importantTerms(s string) map[string]struct{} {
	s = strings.ToLower(s)
	var cleaned strings.Builder

	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' {
			cleaned.WriteRune(r)
		} else {
			cleaned.WriteRune(' ')
		}
	}

	stop := map[string]struct{}{
		"the": {}, "a": {}, "an": {}, "and": {}, "or": {}, "of": {}, "to": {}, "in": {}, "on": {},
		"for": {}, "with": {}, "by": {}, "from": {}, "is": {}, "are": {}, "was": {}, "were": {},
		"be": {}, "been": {}, "it": {}, "this": {}, "that": {}, "what": {}, "why": {}, "how": {},
		"when": {}, "which": {}, "does": {}, "do": {}, "did": {}, "as": {}, "at": {}, "if": {},
		"its": {}, "into": {}, "than": {}, "then": {}, "only": {}, "also": {}, "about": {},
	}

	out := map[string]struct{}{}
	for _, tok := range strings.Fields(cleaned.String()) {
		if len(tok) < 2 {
			continue
		}
		if _, ok := stop[tok]; ok {
			continue
		}
		out[tok] = struct{}{}
	}
	return out
}

func containsKeyTerms(reference, answer string) bool {
	refTerms := importantTerms(reference)
	ansTerms := importantTerms(answer)

	keyCount := 0
	matched := 0

	for term := range refTerms {
		if len(term) >= 3 {
			keyCount++
			if _, ok := ansTerms[term]; ok {
				matched++
			}
		}
	}

	if keyCount == 0 {
		return false
	}

	return float64(matched)/float64(keyCount) >= 0.45
}

func heuristicLanguageCompliance(question, answer string) float64 {
	qGerman := looksGerman(question)
	aGerman := looksGerman(answer)

	if qGerman == aGerman {
		return 1.0
	}

	if !qGerman && strings.Contains(strings.ToLower(answer), "seite") {
		withoutSeite := strings.ReplaceAll(strings.ToLower(answer), "seite", "")
		if !looksGerman(withoutSeite) {
			return 0.9
		}
	}

	return 0.3
}

func looksGerman(s string) bool {
	s = strings.ToLower(s)
	germanMarkers := []string{
		" der ", " die ", " das ", " und ", " ist ", " sind ", " ein ", " eine ",
		" nicht ", " mit ", " für ", " auf ", " im ", " in der ", " zu ",
		"seite", "bereitgestellten", "kontext", "antwort",
	}
	padded := " " + s + " "
	count := 0
	for _, m := range germanMarkers {
		if strings.Contains(padded, m) {
			count++
		}
	}
	return count >= 2
}

func heuristicCitationCompliance(answer string) float64 {
	a := strings.TrimSpace(answer)
	lower := strings.ToLower(a)

	if strings.Contains(lower, "https://example.com") || strings.Contains(lower, "http://example.com") {
		return 0.0
	}

	pageCitation := regexp.MustCompile(`(?i)\(page[:\s]+[^)]*(tag[:\s]+[^)]*)?\)`)
	seiteCitation := regexp.MustCompile(`(?i)\(seite[:\s]+[^)]*(tag[:\s]+[^)]*)?\)`)
	urlCitation := regexp.MustCompile(`(?i)\((source|url)[:\s]+https?://[^)]+\)`)
	bracketPage := regexp.MustCompile(`(?i)\[page[:\s]+[^\]]+\]`)
	bracketURL := regexp.MustCompile(`(?i)\[url[:\s]+https?://[^\]]+\]`)

	switch {
	case pageCitation.MatchString(a):
		return 1.0
	case urlCitation.MatchString(a):
		return 1.0
	case bracketPage.MatchString(a):
		return 0.8
	case bracketURL.MatchString(a):
		return 0.8
	case seiteCitation.MatchString(a):
		return 0.7
	case strings.Contains(lower, "page") || strings.Contains(lower, "seite") || strings.Contains(lower, "url:") || strings.Contains(lower, "source:"):
		return 0.5
	default:
		return 0.2
	}
}

func heuristicFormatCompliance(answer string) float64 {
	a := strings.TrimSpace(strings.ToLower(answer))
	score := 1.0

	if strings.HasPrefix(a, "here is") || strings.HasPrefix(a, "here's") ||
		strings.HasPrefix(a, "based on") || strings.HasPrefix(a, "according to") {
		score -= 0.25
	}

	if strings.Contains(a, "retrieved context") || strings.Contains(a, "provided context snippet") {
		score -= 0.25
	}

	if strings.Contains(a, "<unused") || strings.Contains(a, "<unk>") || strings.Contains(a, "<pad>") {
		score -= 0.8
	}

	if len(strings.Fields(answer)) > 140 {
		score -= 0.20
	}

	return clampScore(score)
}

func hasAnyQualityScore(j ragQualityJudgment) bool {
	return j.AnswerRelevancy > 0 || j.Faithfulness > 0 || j.Correctness > 0 || j.Completeness > 0
}

func hasAnyComplianceScore(j ragQualityJudgment) bool {
	return j.LanguageCompliance > 0 || j.CitationCompliance > 0 || j.FormatCompliance > 0
}

func average4(a, b, c, d float64) float64 {
	return clampScore((clampScore(a) + clampScore(b) + clampScore(c) + clampScore(d)) / 4.0)
}

func average3(a, b, c float64) float64 {
	return clampScore((clampScore(a) + clampScore(b) + clampScore(c)) / 3.0)
}

func extractAnswer(output any) (string, error) {
	if output == nil {
		return "", errors.New("output is nil")
	}
	switch v := output.(type) {
	case string:
		return v, nil
	case map[string]any:
		if a, ok := v["answer"].(string); ok {
			return a, nil
		}
		b, _ := json.Marshal(v)
		return string(b), nil
	default:
		b, _ := json.Marshal(v)
		return string(b), nil
	}
}

func extractOutputMap(output any) (map[string]any, error) {
	if output == nil {
		return nil, errors.New("output is nil")
	}

	switch v := output.(type) {
	case map[string]any:
		return v, nil
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			return nil, err
		}
		return m, nil
	}
}

func extractQuestion(req *ai.EvaluatorCallbackRequest) string {
	if req == nil {
		return ""
	}

	b, err := json.Marshal(req.Input.Input)
	if err == nil {
		var m map[string]any
		if json.Unmarshal(b, &m) == nil {
			if q, ok := m["question"].(string); ok {
				return normalize(q)
			}
			if q, ok := m["query"].(string); ok {
				return normalize(q)
			}
			if q, ok := m["input"].(string); ok {
				return normalize(q)
			}
		}
	}

	return normalize(fmt.Sprintf("%v", req.Input.Input))
}

func extractReference(req *ai.EvaluatorCallbackRequest) string {
	if req == nil {
		return ""
	}

	b, err := json.Marshal(req.Input)
	if err == nil {
		var generic map[string]any
		if json.Unmarshal(b, &generic) == nil {
			if ref, ok := generic["reference"].(string); ok {
				return normalize(ref)
			}
			if ref, ok := generic["expectedOutput"].(string); ok {
				return normalize(ref)
			}
			if ref, ok := generic["expected"].(string); ok {
				return normalize(ref)
			}
			if inp, ok := generic["input"].(map[string]any); ok {
				if ref, ok := inp["reference"].(string); ok {
					return normalize(ref)
				}
				if ref, ok := inp["expectedOutput"].(string); ok {
					return normalize(ref)
				}
				if ref, ok := inp["expected"].(string); ok {
					return normalize(ref)
				}
			}
		}
	}
	return ""
}

func formatContext(ctx []any) string {
	if len(ctx) == 0 {
		return ""
	}

	var parts []string
	seen := map[string]struct{}{}

	for _, c := range ctx {
		var s string
		switch v := c.(type) {
		case string:
			s = v
		case map[string]any:
			if text, ok := v["text"].(string); ok {
				s = text
			} else if content, ok := v["content"].(string); ok {
				s = content
			} else {
				b, _ := json.Marshal(v)
				s = string(b)
			}
		default:
			b, _ := json.Marshal(v)
			s = string(b)
		}

		s = normalizeWhitespace(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		parts = append(parts, s)
	}

	const maxParts = 12
	if len(parts) > maxParts {
		parts = parts[:maxParts]
	}

	return strings.Join(parts, "\n---\n")
}

func extractLikelyJSON(text string) string {
	text = strings.TrimSpace(text)

	if strings.HasPrefix(text, "```") {
		text = strings.TrimPrefix(text, "```json")
		text = strings.TrimPrefix(text, "```")
		text = strings.TrimSuffix(text, "```")
		text = strings.TrimSpace(text)
	}

	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start >= 0 && end > start {
		return text[start : end+1]
	}
	return text
}

func firstScoreInText(text string) float64 {
	for _, word := range strings.Fields(text) {
		word = strings.Trim(word, " \t\r\n.,;:()[]{}\"'")
		if f, err := strconv.ParseFloat(word, 64); err == nil && f >= 0 && f <= 1 {
			return f
		}
	}
	return 0.0
}

func toStringSlice(v any) []string {
	items, ok := v.([]any)
	if !ok {
		return nil
	}

	out := make([]string, 0, len(items))
	for _, item := range items {
		s := normalize(toString(item))
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func toBool(v any) bool {
	switch b := v.(type) {
	case bool:
		return b
	case string:
		return strings.EqualFold(strings.TrimSpace(b), "true")
	case float64:
		return b != 0
	default:
		return false
	}
}

func toString(v any) string {
	switch s := v.(type) {
	case string:
		return s
	case json.Number:
		return s.String()
	case float64:
		return strconv.FormatFloat(s, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(s), 'f', -1, 64)
	case int:
		return strconv.Itoa(s)
	case int64:
		return strconv.FormatInt(s, 10)
	default:
		if v == nil {
			return ""
		}
		b, _ := json.Marshal(v)
		return string(b)
	}
}

func toFloat64(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case json.Number:
		f, _ := n.Float64()
		return f
	case string:
		f, _ := strconv.ParseFloat(strings.TrimSpace(n), 64)
		return f
	default:
		return 0
	}
}

func clampScore(s float64) float64 {
	if math.IsNaN(s) || math.IsInf(s, 0) {
		return 0
	}
	if s < 0 {
		return 0
	}
	if s > 1 {
		return 1
	}
	return s
}

func normalize(s string) string {
	return normalizeWhitespace(strings.TrimSpace(s))
}

func normalizeWhitespace(s string) string {
	return strings.Join(strings.Fields(strings.ReplaceAll(s, "\u00A0", " ")), " ")
}

func isGarbageAnswer(s string) bool {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return true
	}
	if s == "{}" || s == "[]" || s == "null" {
		return true
	}
	badMarkers := []string{"<unused", "<pad>", "<unk>"}
	for _, m := range badMarkers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

func isQuestionRepetition(question, answer string) bool {
	q := normalize(strings.ToLower(question))
	a := normalize(strings.ToLower(answer))
	if q == "" || a == "" {
		return false
	}
	if a == q {
		return true
	}
	return strings.Contains(a, q) && len(a) <= len(q)+10
}

func isNotInDocument(reference string) bool {
	r := strings.TrimSpace(strings.ToUpper(reference))
	return r == "NOT_IN_DOCUMENT" ||
		r == "NOT IN DOCUMENT" ||
		r == "NOT_IN_CONTEXT" ||
		r == "NOT IN CONTEXT"
}

func containsInsufficientAnswer(a string) bool {
	return strings.Contains(a, "insufficient") ||
		strings.Contains(a, "not in the document") ||
		strings.Contains(a, "not in the provided") ||
		strings.Contains(a, "does not contain") ||
		strings.Contains(a, "cannot answer") ||
		strings.Contains(a, "not enough information") ||
		strings.Contains(a, "nicht ausreichend")
}

func scoreUnanswerableBehavior(answer string) (float64, string) {
	a := strings.ToLower(strings.TrimSpace(answer))

	switch {
	case a == "":
		return 0.0, "empty answer for unanswerable question"

	case strings.Contains(a, "not_in_document"),
		strings.Contains(a, "not in document"),
		strings.Contains(a, "not in the document"),
		strings.Contains(a, "not in the documents"),
		strings.Contains(a, "not in the provided document"),
		strings.Contains(a, "not in the provided documents"),
		strings.Contains(a, "not in the provided context"),
		strings.Contains(a, "not provided in the context"),
		strings.Contains(a, "not available in the provided"),
		strings.Contains(a, "not mentioned in the provided"),
		strings.Contains(a, "does not contain information"),
		strings.Contains(a, "do not contain information"),
		strings.Contains(a, "does not provide information"),
		strings.Contains(a, "do not provide information"),
		strings.Contains(a, "cannot be determined from the provided"),
		strings.Contains(a, "cannot answer based on the provided"),
		strings.Contains(a, "i don't know based on the provided"),
		strings.Contains(a, "don't have enough information"),
		strings.Contains(a, "do not have enough information"),
		strings.Contains(a, "doesn't have enough information"),
		strings.Contains(a, "does not have enough information"),
		strings.Contains(a, "not enough information"),
		strings.Contains(a, "not specified"),
		strings.Contains(a, "does not specify"),
		strings.Contains(a, "doesn't specify"),
		strings.Contains(a, "does not explicitly state"),
		strings.Contains(a, "does not state"),
		strings.Contains(a, "is not explicitly stated"),
		strings.Contains(a, "is not stated"),
		strings.Contains(a, "insufficient"),
		strings.Contains(a, "nicht im dokument"),
		strings.Contains(a, "nicht in den dokumenten"),
		strings.Contains(a, "nicht im bereitgestellten kontext"),
		strings.Contains(a, "nicht aus dem bereitgestellten kontext"),
		strings.Contains(a, "nicht ausreichend"):
		return 1.0, "correctly abstained for an unanswerable question"

	case strings.Contains(a, "i don't know"),
		strings.Contains(a, "cannot answer"),
		strings.Contains(a, "kann ich nicht beantworten"),
		strings.Contains(a, "weiß ich nicht"):
		return 0.8, "abstained, but without explicitly tying the abstention to the provided documents"

	default:
		return 0.0, "question was marked unanswerable but the model still produced a substantive answer"
	}
}

func truncate(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
