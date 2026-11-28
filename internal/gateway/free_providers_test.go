package gateway_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/gateway"
	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/localai"
)

func TestFreeProvidersConfigFromEnvAndNoop(t *testing.T) {
	cfg := gateway.FreeProvidersConfigFromEnv(func(k string) string {
		if k == "NVIDIA_NIM_API_KEY" {
			return " nvapi-test "
		}
		return ""
	})
	if cfg.NvidiaAPIKey != "nvapi-test" || cfg.GroqAPIKey != "" || cfg.OpenRouterAPIKey != "" {
		t.Fatalf("unexpected env config: %+v", cfg)
	}

	gw, _ := setupTestGateway(t)
	registered, err := gw.ConfigureFreeProviders(context.Background(), gateway.FreeProvidersConfig{})
	if err != nil || len(registered) != 0 {
		t.Fatalf("expected empty noop, got registered=%v err=%v", registered, err)
	}
}

func TestFreeProvidersFailoverChainAndAirGap(t *testing.T) {
	gw, handler := setupTestGateway(t)

	var groqHits atomic.Int32
	var openRouterHits atomic.Int32

	geminiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"gemini quota 429"}`))
	}))
	defer geminiServer.Close()

	groqServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		groqHits.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"groq quota 429"}`))
	}))
	defer groqServer.Close()

	openRouterServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		openRouterHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"or1","object":"chat.completion","model":"openrouter/free","choices":[{"index":0,"message":{"role":"assistant","content":"failover-ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}`))
	}))
	defer openRouterServer.Close()

	if _, err := gw.ConfigureGemini(context.Background(), gateway.GeminiConfig{
		APIKey:  "gemini-k",
		BaseURL: geminiServer.URL + "/v1beta/openai",
	}); err != nil {
		t.Fatalf("configure gemini: %v", err)
	}

	cfg := gateway.FreeProvidersConfig{
		GroqAPIKey:        "groq-k",
		GroqBaseURL:       groqServer.URL + "/v1",
		OpenRouterAPIKey:  "or-k",
		OpenRouterBaseURL: openRouterServer.URL + "/v1",
	}
	registered, err := gw.ConfigureFreeProviders(context.Background(), cfg)
	if err != nil || len(registered) != 2 {
		t.Fatalf("expected 2 registered providers, got %v err=%v", registered, err)
	}
	// Idempotent re-configure must not create duplicates.
	if _, err := gw.ConfigureFreeProviders(context.Background(), cfg); err != nil {
		t.Fatalf("reconfigure free providers: %v", err)
	}

	rec := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"data_classification": "cloud_eligible",
		"messages":            []map[string]any{{"role": "user", "content": "hello"}},
	}, nil)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "failover-ok") {
		t.Fatalf("expected 200 failover-ok, got %d: %s", rec.Code, rec.Body.String())
	}
	if groqHits.Load() == 0 || openRouterHits.Load() == 0 {
		t.Fatalf("expected failover through groq (%d) to openrouter (%d)", groqHits.Load(), openRouterHits.Load())
	}

	// Switch to default external endpoints so local_only rejects non-loopback egress fail-closed.
	if _, err := gw.ConfigureGemini(context.Background(), gateway.GeminiConfig{APIKey: "gemini-k"}); err != nil {
		t.Fatalf("reset gemini external: %v", err)
	}
	if _, err := gw.ConfigureFreeProviders(context.Background(), gateway.FreeProvidersConfig{
		GroqAPIKey:       "groq-k",
		OpenRouterAPIKey: "or-k",
	}); err != nil {
		t.Fatalf("reset free providers external: %v", err)
	}
	blocked := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"data_classification": "local_only",
		"messages":            []map[string]any{{"role": "user", "content": "private"}},
	}, nil)
	if blocked.Code != http.StatusForbidden || !strings.Contains(blocked.Body.String(), "local_only_egress_blocked") {
		t.Fatalf("expected 403 local_only_egress_blocked, got %d: %s", blocked.Code, blocked.Body.String())
	}
}

func TestFreeTierRPDAndTPMQuotaTrackerSmartCooldownAndOllamaFailover(t *testing.T) {
	now := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)

	var ollamaHits atomic.Int32
	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/api/tags" {
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5-coder:7b","size":4000000000}]}`))
			return
		}
		ollamaHits.Add(1)
		_, _ = w.Write([]byte(`{"id":"ol-1","object":"chat.completion","model":"qwen2.5-coder:7b","choices":[{"index":0,"message":{"role":"assistant","content":"served-by-local-ollama"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":4,"total_tokens":8}}`))
	}))
	defer ollamaSrv.Close()

	discoverer := localai.NewDiscoverer([]localai.Target{
		{ID: "ollama", Name: "Ollama", Kind: "ollama", ProbeURL: ollamaSrv.URL + "/api/tags", OpenAIBase: ollamaSrv.URL + "/v1"},
	}, 2*time.Second, nil)

	gw, handler := setupTestGateway(t, func(o *gateway.Options) {
		o.LocalDiscovery = discoverer
		o.Clock = func() time.Time { return now }
	})

	var groqHits atomic.Int32
	var orHits atomic.Int32
	groqMode429 := false
	retryAfterHdr := ""

	groqSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		groqHits.Add(1)
		if groqMode429 {
			if retryAfterHdr != "" {
				w.Header().Set("Retry-After", retryAfterHdr)
			}
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":"rate limit"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"g1","object":"chat.completion","model":"llama-3.3-70b-versatile","choices":[{"index":0,"message":{"role":"assistant","content":"served-by-groq"},"finish_reason":"stop"}],"usage":{"prompt_tokens":300,"completion_tokens":200,"total_tokens":500}}`))
	}))
	defer groqSrv.Close()

	orSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		orHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"or1","object":"chat.completion","model":"openrouter/free","choices":[{"index":0,"message":{"role":"assistant","content":"served-by-openrouter"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}}`))
	}))
	defer orSrv.Close()

	groqProv, err := gw.AddProvider(context.Background(), gateway.ProviderCreateInput{
		Name:         gateway.GroqProviderName,
		Kind:         gateway.KindOpenAICompat,
		Tier:         gateway.TierFree,
		BaseURL:      groqSrv.URL + "/v1",
		APIKey:       "groq-k",
		DefaultModel: gateway.DefaultGroqModel,
		Priority:     60,
	})
	if err != nil {
		t.Fatal(err)
	}
	orProv, err := gw.AddProvider(context.Background(), gateway.ProviderCreateInput{
		Name:         gateway.OpenRouterProviderName,
		Kind:         gateway.KindOpenAICompat,
		Tier:         gateway.TierFree,
		BaseURL:      orSrv.URL + "/v1",
		APIKey:       "or-k",
		DefaultModel: gateway.DefaultOpenRouterModel,
		Priority:     70,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Set tight TPM=500 on Groq and RPD=1 on OpenRouter.
	gw.SetProviderQuotaLimits(groqProv.ID, gateway.ProviderQuotaLimits{RPDLimit: 10, TPMLimit: 500})
	gw.SetProviderQuotaLimits(orProv.ID, gateway.ProviderQuotaLimits{RPDLimit: 1, TPMLimit: 10000})

	highTemp := 0.8

	// 1. First call uses Groq (500 tokens -> reaches Groq TPM limit of 500).
	rec1 := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"temperature": highTemp,
		"messages":    []map[string]any{{"role": "user", "content": "req 1"}},
	}, nil)
	if rec1.Code != http.StatusOK || rec1.Header().Get("X-Rocksoul-Provider") != gateway.GroqProviderName {
		t.Fatalf("expected groq on req 1, got code=%d provider=%q body=%s", rec1.Code, rec1.Header().Get("X-Rocksoul-Provider"), rec1.Body.String())
	}

	// 2. Second call within same minute: Groq is TPM-exhausted -> proactively routes to OpenRouter (reaches OpenRouter RPD=1).
	rec2 := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"temperature": highTemp,
		"messages":    []map[string]any{{"role": "user", "content": "req 2"}},
	}, nil)
	if rec2.Code != http.StatusOK || rec2.Header().Get("X-Rocksoul-Provider") != gateway.OpenRouterProviderName {
		t.Fatalf("expected openrouter on req 2 when groq TPM exhausted, got code=%d provider=%q", rec2.Code, rec2.Header().Get("X-Rocksoul-Provider"))
	}

	// 3. Third call within same minute: BOTH Groq (TPM) and OpenRouter (RPD) are quota-exhausted -> automatically fails over to local Ollama!
	rec3 := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"temperature": highTemp,
		"messages":    []map[string]any{{"role": "user", "content": "req 3"}},
	}, nil)
	if rec3.Code != http.StatusOK || rec3.Header().Get("X-Rocksoul-Provider") != "local-ollama" || !strings.Contains(rec3.Body.String(), "served-by-local-ollama") {
		t.Fatalf("expected automatic failover to local-ollama on req 3, got code=%d provider=%q body=%s", rec3.Code, rec3.Header().Get("X-Rocksoul-Provider"), rec3.Body.String())
	}
	if ollamaHits.Load() != 1 {
		t.Fatalf("expected 1 ollama hit, got %d", ollamaHits.Load())
	}

	// Verify GET /api/ai/quotas reports RPD/TPM exhaustion accurately.
	quotaRec := doJSONRequest(t, handler, http.MethodGet, "/api/ai/quotas", nil, nil)
	if quotaRec.Code != http.StatusOK || !strings.Contains(quotaRec.Body.String(), `"tpmExhausted":true`) || !strings.Contains(quotaRec.Body.String(), `"rpdExhausted":true`) {
		t.Fatalf("expected tpmExhausted and rpdExhausted in /api/ai/quotas, got %d: %s", quotaRec.Code, quotaRec.Body.String())
	}

	// 4. Advance clock by 61 seconds: Groq 60s TPM window resets! Now test smart exponential cooldown backoff (30s -> 60s -> Retry-After 180s) & Ollama failover on 429.
	now = now.Add(61 * time.Second)
	groqMode429 = true

	rec4 := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"temperature": highTemp,
		"messages":    []map[string]any{{"role": "user", "content": "req 4 triggers 429"}},
	}, nil)
	if rec4.Code != http.StatusOK || rec4.Header().Get("X-Rocksoul-Provider") != "local-ollama" {
		t.Fatalf("expected 429 on groq to fail over to local-ollama, got code=%d provider=%q", rec4.Code, rec4.Header().Get("X-Rocksoul-Provider"))
	}
	q1, err := gw.GetProviderQuotaStatus(context.Background(), groqProv.ID)
	if err != nil || q1.ConsecutiveFailures != 1 || q1.LastBackoffSeconds != 30 || !q1.InCooldown {
		t.Fatalf("expected 1st failure backoff=30s, got %+v err=%v", q1, err)
	}

	// Advance past 30s cooldown and trigger 2nd consecutive 429 -> backoff doubles to 60s.
	now = now.Add(31 * time.Second)
	_ = doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"temperature": highTemp,
		"messages":    []map[string]any{{"role": "user", "content": "req 5 triggers 2nd 429"}},
	}, nil)
	q2, _ := gw.GetProviderQuotaStatus(context.Background(), groqProv.ID)
	if q2.ConsecutiveFailures != 2 || q2.LastBackoffSeconds != 60 {
		t.Fatalf("expected 2nd failure exponential backoff=60s, got %+v", q2)
	}

	// Advance past 60s cooldown and return Retry-After: 180 -> respects upstream Retry-After header.
	now = now.Add(61 * time.Second)
	retryAfterHdr = "180"
	_ = doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"temperature": highTemp,
		"messages":    []map[string]any{{"role": "user", "content": "req 6 triggers Retry-After 180"}},
	}, nil)
	q3, _ := gw.GetProviderQuotaStatus(context.Background(), groqProv.ID)
	if q3.ConsecutiveFailures != 3 || q3.LastBackoffSeconds != 180 {
		t.Fatalf("expected Retry-After=180s backoff, got %+v", q3)
	}
}
