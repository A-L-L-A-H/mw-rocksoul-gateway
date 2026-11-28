package gateway_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/gateway"
)

func startC24MockUpstream(t *testing.T, hits *atomic.Int32, status int, retryAfter, reply string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if retryAfter != "" {
			w.Header().Set("Retry-After", retryAfter)
		}
		if status != http.StatusOK {
			w.WriteHeader(status)
		}
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func registerC24Provider(t *testing.T, app http.Handler, payload map[string]any) gateway.Provider {
	t.Helper()
	rec := doJSONRequest(t, app, http.MethodPost, "/api/ai/providers", payload, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create provider failed: %d %s", rec.Code, rec.Body.String())
	}
	var prov gateway.Provider
	if err := json.Unmarshal(rec.Body.Bytes(), &prov); err != nil {
		t.Fatalf("decode provider: %v", err)
	}
	if prov.ID == "" {
		t.Fatalf("expected non-empty provider ID, got body=%s", rec.Body.String())
	}
	return prov
}

func sendC24Chat(t *testing.T, app http.Handler, prompt string) *httptest.ResponseRecorder {
	t.Helper()
	return doJSONRequest(t, app, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":       "qwen2.5-coder:7b",
		"temperature": 0.7,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
	}, nil)
}

func TestC24QuotaRPDTPMCooldownAndLocalOllamaFailover(t *testing.T) {
	gw, app := setupTestGateway(t)
	gw.SetClock(func() time.Time { return time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC) })

	var groqHits, geminiHits, ollamaHits atomic.Int32
	groqBody := `{"id":"g1","object":"chat.completion","created":1730000000,"model":"qwen2.5-coder:7b",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"served by groq"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}`
	ollamaBody := `{"id":"o1","object":"chat.completion","created":1730000000,"model":"qwen2.5-coder:7b",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"served by local ollama"},` +
		`"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":8,"total_tokens":20}}`

	groqSrv := startC24MockUpstream(t, &groqHits, http.StatusOK, "", groqBody)
	geminiSrv := startC24MockUpstream(t, &geminiHits, http.StatusTooManyRequests, "45", `{"error":{"message":"429"}}`)
	ollamaSrv := startC24MockUpstream(t, &ollamaHits, http.StatusOK, "", ollamaBody)

	groqProv := registerC24Provider(t, app, map[string]any{
		"name": "groq", "kind": "openai_compat", "tier": "free",
		"baseUrl": groqSrv.URL + "/v1", "apiKey": "groq-test-key",
		"defaultModel": "qwen2.5-coder:7b", "models": []string{"qwen2.5-coder:7b"}, "priority": 10,
	})
	geminiProv := registerC24Provider(t, app, map[string]any{
		"name": "gemini", "kind": "openai_compat", "tier": "free",
		"baseUrl": geminiSrv.URL + "/v1", "apiKey": "gemini-test-key",
		"defaultModel": "qwen2.5-coder:7b", "models": []string{"qwen2.5-coder:7b"}, "priority": 20,
	})
	registerC24Provider(t, app, map[string]any{
		"name": "local-ollama", "kind": "openai_compat", "tier": "free", "isLocal": true,
		"baseUrl": ollamaSrv.URL + "/v1", "apiKey": "ollama",
		"defaultModel": "qwen2.5-coder:7b", "models": []string{"qwen2.5-coder:7b"}, "priority": 30,
	})

	gw.SetProviderQuotaLimits(groqProv.ID, gateway.ProviderQuotaLimits{RPDLimit: 1, TPMLimit: 100})
	gw.SetProviderQuotaLimits(geminiProv.ID, gateway.ProviderQuotaLimits{RPDLimit: 10, TPMLimit: 100})

	r1 := sendC24Chat(t, app, "c24 request 1")
	if r1.Code != http.StatusOK || r1.Header().Get("X-Rocksoul-Provider") != "groq" {
		t.Fatalf("req1 expected groq 200, got %d %q", r1.Code, r1.Header().Get("X-Rocksoul-Provider"))
	}

	r2 := sendC24Chat(t, app, "c24 request 2")
	if r2.Code != http.StatusOK || r2.Header().Get("X-Rocksoul-Provider") != "local-ollama" {
		t.Fatalf("req2 expected local-ollama 200, got %d %q", r2.Code, r2.Header().Get("X-Rocksoul-Provider"))
	}
	if !strings.Contains(r2.Body.String(), "served by local ollama") {
		t.Fatalf("req2 unexpected body: %s", r2.Body.String())
	}

	assertC24QuotaStates(t, gw)

	r3 := sendC24Chat(t, app, "c24 request 3")
	if r3.Code != http.StatusOK || r3.Header().Get("X-Rocksoul-Provider") != "local-ollama" {
		t.Fatalf("req3 expected local-ollama 200, got %d %q", r3.Code, r3.Header().Get("X-Rocksoul-Provider"))
	}
	if groqHits.Load() != 1 || geminiHits.Load() != 1 || ollamaHits.Load() != 2 {
		t.Fatalf("unexpected hits: groq=%d gemini=%d ollama=%d", groqHits.Load(), geminiHits.Load(), ollamaHits.Load())
	}
}

func assertC24QuotaStates(t *testing.T, gw *gateway.Gateway) {
	t.Helper()
	quotas, err := gw.ListProviderQuotas(context.Background())
	if err != nil {
		t.Fatalf("ListProviderQuotas failed: %v", err)
	}
	byName := make(map[string]gateway.ProviderQuotaStatus, len(quotas))
	for _, q := range quotas {
		byName[q.ProviderName] = q
	}
	if !byName["groq"].RPDExhausted {
		t.Fatalf("expected groq RPDExhausted=true, got %+v", byName["groq"])
	}
	if !byName["gemini"].InCooldown || byName["gemini"].LastBackoffSeconds < 45 {
		t.Fatalf("expected gemini in cooldown >=45s, got %+v", byName["gemini"])
	}
	if byName["local-ollama"].RPDExhausted || byName["local-ollama"].TPMExhausted {
		t.Fatalf("expected local-ollama unexhausted, got %+v", byName["local-ollama"])
	}
}
