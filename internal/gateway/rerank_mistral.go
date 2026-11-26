package gateway

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// DefaultCohereBaseURL is the Cohere Trial API root.
const DefaultCohereBaseURL = "https://api.cohere.com/v2"

// RerankRequest is the POST /v1/rerank body.
type RerankRequest struct {
	Query              string   `json:"query"`
	Documents          []string `json:"documents"`
	TopN               int      `json:"top_n,omitempty"`
	DataClassification string   `json:"data_classification,omitempty"`
}

// RerankResult is a scored document reference.
type RerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// RerankResponse is the POST /v1/rerank response.
type RerankResponse struct {
	Provider string         `json:"provider"`
	Results  []RerankResult `json:"results"`
}

func tokenizeTerms(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// RerankBM25 deterministically ranks documents against query with Okapi BM25.
// Ties break by ascending original index.
func RerankBM25(query string, documents []string, topN int) []RerankResult {
	const k1, b = 1.5, 0.75
	queryTerms := tokenizeTerms(query)
	docTerms := make([][]string, len(documents))
	docFreq := make(map[string]int)
	totalLen := 0
	for i, doc := range documents {
		docTerms[i] = tokenizeTerms(doc)
		totalLen += len(docTerms[i])
		seen := make(map[string]struct{})
		for _, term := range docTerms[i] {
			if _, ok := seen[term]; !ok {
				seen[term] = struct{}{}
				docFreq[term]++
			}
		}
	}
	avgLen := 1.0
	if len(documents) > 0 && totalLen > 0 {
		avgLen = float64(totalLen) / float64(len(documents))
	}
	results := make([]RerankResult, len(documents))
	for i, terms := range docTerms {
		freq := make(map[string]int)
		for _, term := range terms {
			freq[term]++
		}
		score := 0.0
		for _, term := range queryTerms {
			tf := float64(freq[term])
			if tf == 0 {
				continue
			}
			idf := math.Log(1 + (float64(len(documents))-float64(docFreq[term])+0.5)/(float64(docFreq[term])+0.5))
			score += idf * tf * (k1 + 1) / (tf + k1*(1-b+b*float64(len(terms))/avgLen))
		}
		results[i] = RerankResult{Index: i, RelevanceScore: score}
	}
	sort.SliceStable(results, func(a, c int) bool { return results[a].RelevanceScore > results[c].RelevanceScore })
	if topN > 0 && topN < len(results) {
		results = results[:topN]
	}
	return results
}

func (g *Gateway) handleRerank(w http.ResponseWriter, r *http.Request) {
	var req RerankRequest
	if err := decodeJSONBody(r, &req); err != nil || strings.TrimSpace(req.Query) == "" || len(req.Documents) == 0 {
		writeGatewayError(w, &upstreamCallError{StatusCode: http.StatusBadRequest, Code: "invalid_rerank_request", Body: "query and non-empty documents are required"})
		return
	}
	classification, err := g.resolveClassification(r.Header.Get(headerClassification), req.DataClassification)
	if err != nil {
		writeGatewayError(w, &upstreamCallError{StatusCode: http.StatusBadRequest, Code: "invalid_data_classification", Body: err.Error()})
		return
	}
	cohereKey := strings.TrimSpace(os.Getenv("COHERE_API_KEY"))
	started := time.Now()
	if classification != ClassificationLocalOnly && cohereKey != "" {
		if results, ok := cohereRerank(r.Context(), cohereKey, req); ok {
			EmitUsageTelemetry(r.Context(), "rerank", "cohere", len(req.Documents), time.Since(started))
			writeJSON(w, http.StatusOK, RerankResponse{Provider: "cohere", Results: results})
			return
		}
	}
	results := RerankBM25(req.Query, req.Documents, req.TopN)
	EmitUsageTelemetry(r.Context(), "rerank", "local-bm25", len(req.Documents), time.Since(started))
	writeJSON(w, http.StatusOK, RerankResponse{Provider: "local-bm25", Results: results})
}

func cohereRerank(ctx context.Context, apiKey string, req RerankRequest) ([]RerankResult, bool) {
	base := strings.TrimRight(firstNonEmpty(os.Getenv("COHERE_BASE_URL"), DefaultCohereBaseURL), "/")
	payload := map[string]any{"model": "rerank-v3.5", "query": req.Query, "documents": req.Documents}
	if req.TopN > 0 {
		payload["top_n"] = req.TopN
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, false
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/rerank", bytes.NewReader(encoded))
	if err != nil {
		return nil, false
	}
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(httpReq)
	if err != nil {
		return nil, false
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		return nil, false
	}
	var parsed struct {
		Results []RerankResult `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil || len(parsed.Results) == 0 {
		return nil, false
	}
	return parsed.Results, true
}

// EmitUsageTelemetry posts a best-effort OTLP/HTTP JSON span to
// OTEL_EXPORTER_OTLP_ENDPOINT. It is a no-op when the endpoint is unset.
func EmitUsageTelemetry(ctx context.Context, operation, provider string, itemCount int, elapsed time.Duration) {
	endpoint := strings.TrimRight(strings.TrimSpace(os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")), "/")
	if endpoint == "" {
		return
	}
	if !strings.HasSuffix(endpoint, "/v1/traces") {
		endpoint += "/v1/traces"
	}
	end := time.Now()
	attr := func(key, value string) map[string]any {
		return map[string]any{"key": key, "value": map[string]any{"stringValue": value}}
	}
	body := map[string]any{"resourceSpans": []any{map[string]any{
		"resource": map[string]any{"attributes": []any{attr("service.name", "mw-rocksoul-gateway")}},
		"scopeSpans": []any{map[string]any{"spans": []any{map[string]any{
			"traceId":           randomHex(16),
			"spanId":            randomHex(8),
			"name":              "gateway." + operation,
			"kind":              3,
			"startTimeUnixNano": jsonNano(end.Add(-elapsed)),
			"endTimeUnixNano":   jsonNano(end),
			"attributes": []any{
				attr("gateway.provider", provider),
				attr("gateway.item_count", itoa(itemCount)),
			},
		}}}},
	}}}
	encoded, err := json.Marshal(body)
	if err != nil {
		return
	}
	telemetryCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Second)
	go func() {
		defer cancel()
		req, reqErr := http.NewRequestWithContext(telemetryCtx, http.MethodPost, endpoint, bytes.NewReader(encoded))
		if reqErr != nil {
			return
		}
		req.Header.Set("Content-Type", "application/json")
		if resp, doErr := http.DefaultClient.Do(req); doErr == nil {
			_ = resp.Body.Close()
		}
	}()
}

func randomHex(byteCount int) string {
	buf := make([]byte, byteCount)
	_, _ = rand.Read(buf)
	return hex.EncodeToString(buf)
}

func jsonNano(moment time.Time) string {
	return strconv.FormatInt(moment.UnixNano(), 10)
}

func itoa(value int) string { return strconv.Itoa(value) }
