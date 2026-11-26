package gateway_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/gateway"
)

func TestWhisperProxyAndLocalOnlyBlock(t *testing.T) {
	_, handler := setupTestGateway(t)
	var gotAuth, gotBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		gotAuth, gotBody = r.Header.Get("Authorization"), string(raw)
		if r.URL.Path != "/audio/transcriptions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"text":"hello"}`))
	}))
	defer upstream.Close()
	t.Setenv("GROQ_API_KEY", "gsk-test")
	t.Setenv("GROQ_BASE_URL", upstream.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", strings.NewReader("AUDIO"))
	req.Header.Set("Content-Type", "audio/wav")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "hello") || gotAuth != "Bearer gsk-test" || gotBody != "AUDIO" {
		t.Fatalf("proxy failed: %d %s auth=%q body=%q", rec.Code, rec.Body.String(), gotAuth, gotBody)
	}

	blocked := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", strings.NewReader("AUDIO"))
	blocked.Header.Set("X-Rocksoul-Data-Classification", "local_only")
	blockedRec := httptest.NewRecorder()
	handler.ServeHTTP(blockedRec, blocked)
	if blockedRec.Code != http.StatusForbidden || !strings.Contains(blockedRec.Body.String(), "local_only_egress_blocked") {
		t.Fatalf("expected 403 local_only_egress_blocked, got %d %s", blockedRec.Code, blockedRec.Body.String())
	}
}

func TestWhisperDeepgramFallback(t *testing.T) {
	_, handler := setupTestGateway(t)
	groq := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(500) }))
	defer groq.Close()
	deepgram := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Token dg-test" {
			t.Errorf("bad deepgram auth")
		}
		_, _ = w.Write([]byte(`{"results":"dg"}`))
	}))
	defer deepgram.Close()
	t.Setenv("GROQ_API_KEY", "g")
	t.Setenv("GROQ_BASE_URL", groq.URL)
	t.Setenv("DEEPGRAM_API_KEY", "dg-test")
	t.Setenv("DEEPGRAM_BASE_URL", deepgram.URL)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", strings.NewReader("A")))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "dg") {
		t.Fatalf("fallback failed: %d %s", rec.Code, rec.Body.String())
	}
}

func TestMistralPresetRegistered(t *testing.T) {
	gw, _ := setupTestGateway(t)
	cfg := gateway.FreeProvidersConfigFromEnv(func(k string) string {
		if k == "MISTRAL_API_KEY" {
			return "mk"
		}
		return ""
	})
	registered, err := gw.ConfigureFreeProviders(context.Background(), cfg)
	if err != nil || len(registered) != 1 || registered[0] != gateway.MistralProviderName {
		t.Fatalf("mistral not registered: %v %v", registered, err)
	}
	providers, _ := gw.ListProviders(context.Background())
	for _, p := range providers {
		if p.Name == gateway.MistralProviderName && p.Priority != 85 {
			t.Fatalf("priority %d", p.Priority)
		}
	}
}

func TestRerankBM25FallbackAndCohere(t *testing.T) {
	_, handler := setupTestGateway(t)
	t.Setenv("COHERE_API_KEY", "")
	body := `{"query":"go gateway","documents":["python script","go gateway proxy","rust"],"top_n":2}`
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/rerank", strings.NewReader(body)))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"provider":"local-bm25"`) || !strings.Contains(rec.Body.String(), `"index":1`) {
		t.Fatalf("bm25 failed: %d %s", rec.Code, rec.Body.String())
	}
	results := gateway.RerankBM25("go gateway", []string{"a", "go gateway"}, 0)
	if results[0].Index != 1 {
		t.Fatalf("bm25 order wrong: %+v", results)
	}

	cohere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"index":2,"relevance_score":0.9}]}`))
	}))
	defer cohere.Close()
	t.Setenv("COHERE_API_KEY", "ck")
	t.Setenv("COHERE_BASE_URL", cohere.URL)
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/rerank", strings.NewReader(body)))
	if !strings.Contains(rec.Body.String(), `"provider":"cohere"`) {
		t.Fatalf("cohere failed: %s", rec.Body.String())
	}
	local := httptest.NewRequest(http.MethodPost, "/v1/rerank", strings.NewReader(body))
	local.Header.Set("X-Rocksoul-Data-Classification", "local_only")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, local)
	if !strings.Contains(rec.Body.String(), `"provider":"local-bm25"`) {
		t.Fatalf("local_only must not hit cohere: %s", rec.Body.String())
	}
}
