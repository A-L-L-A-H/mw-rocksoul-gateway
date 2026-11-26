package gateway_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/gateway"
)

func TestGeminiConfigFromEnvAlias(t *testing.T) {
	env := map[string]string{"GOOGLE_AI_API_KEY": " alias-key "}
	cfg := gateway.GeminiConfigFromEnv(func(k string) string { return env[k] })
	if cfg.APIKey != "alias-key" {
		t.Fatalf("alias key not resolved: %q", cfg.APIKey)
	}
	env["GEMINI_API_KEY"] = "primary"
	cfg = gateway.GeminiConfigFromEnv(func(k string) string { return env[k] })
	if cfg.APIKey != "primary" {
		t.Fatalf("primary key must win: %q", cfg.APIKey)
	}
}

func TestGeminiNoKeyIsNoop(t *testing.T) {
	gw, _ := setupTestGateway(t)
	added, err := gw.ConfigureGemini(context.Background(), gateway.GeminiConfig{})
	if err != nil || added {
		t.Fatalf("expected noop, got added=%v err=%v", added, err)
	}
}

func TestGeminiRoutingModelsAndEgressPolicy(t *testing.T) {
	gw, handler := setupTestGateway(t)

	var upstreamHits atomic.Int32
	var gotAuth string
	var gotUser string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits.Add(1)
		gotAuth = r.Header.Get("Authorization")
		var req gateway.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Messages) > 0 {
			gotUser, _ = req.Messages[0].Content.(string)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"g1","object":"chat.completion","model":"gemini-2.5-pro","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`))
	}))
	defer upstream.Close()

	added, err := gw.ConfigureGemini(context.Background(), gateway.GeminiConfig{APIKey: "AIza-test-key", BaseURL: upstream.URL + "/v1beta/openai"})
	if err != nil || !added {
		t.Fatalf("configure gemini: added=%v err=%v", added, err)
	}
	// Idempotent refresh must not duplicate the provider.
	if _, err := gw.ConfigureGemini(context.Background(), gateway.GeminiConfig{APIKey: "AIza-test-key", BaseURL: upstream.URL + "/v1beta/openai"}); err != nil {
		t.Fatalf("reconfigure gemini: %v", err)
	}
	providers, _ := gw.ListProviders(context.Background())
	count := 0
	for _, provider := range providers {
		if provider.Name == gateway.GeminiProviderName {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected exactly 1 gemini provider, got %d", count)
	}

	modelsRec := doJSONRequest(t, handler, http.MethodGet, "/v1/models", nil, nil)
	for _, model := range gateway.GeminiFreeTierModels {
		if !strings.Contains(modelsRec.Body.String(), `"`+model+`"`) {
			t.Fatalf("model %s missing from /v1/models: %s", model, modelsRec.Body.String())
		}
	}

	rec := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":               "gemini-2.5-pro",
		"data_classification": "cloud_eligible",
		"messages":            []map[string]any{{"role": "user", "content": "mail alice@example.com"}},
	}, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if gotAuth != "Bearer AIza-test-key" {
		t.Fatalf("unexpected upstream auth header %q", gotAuth)
	}
	if strings.Contains(gotUser, "alice@example.com") {
		t.Fatalf("PII leaked to gemini upstream: %q", gotUser)
	}

	// Register external cloud endpoint to verify non-loopback local_only air-gap block.
	if _, err := gw.ConfigureGemini(context.Background(), gateway.GeminiConfig{APIKey: "AIza-test-key"}); err != nil {
		t.Fatalf("reconfigure default external gemini: %v", err)
	}
	before := upstreamHits.Load()
	blocked := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":               "gemini-2.5-flash",
		"data_classification": "local_only",
		"messages":            []map[string]any{{"role": "user", "content": "secret"}},
	}, nil)
	if blocked.Code != http.StatusForbidden || !strings.Contains(blocked.Body.String(), "local_only_egress_blocked") {
		t.Fatalf("expected 403 local_only_egress_blocked, got %d: %s", blocked.Code, blocked.Body.String())
	}
	if upstreamHits.Load() != before {
		t.Fatal("local_only request reached gemini upstream")
	}
}
