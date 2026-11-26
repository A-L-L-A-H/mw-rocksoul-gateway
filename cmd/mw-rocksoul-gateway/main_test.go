package main

import (
	"bytes"
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/auth"
	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/gateway"
	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/localai"
)

func TestHTTPHandlersWithFailClosedRIDAuth(t *testing.T) {
	privKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate rsa key: %v", err)
	}

	var ridSrv *httptest.Server
	ridSrv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/.well-known/openid-configuration":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"issuer":   ridSrv.URL,
				"jwks_uri": ridSrv.URL + "/jwks",
			})
		case "/jwks":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"keys": []map[string]any{
					{
						"kty": "RSA",
						"kid": "gw-key-1",
						"use": "sig",
						"alg": "RS256",
						"n":   base64.RawURLEncoding.EncodeToString(privKey.N.Bytes()),
						"e":   base64.RawURLEncoding.EncodeToString(big.NewInt(int64(privKey.E)).Bytes()),
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer ridSrv.Close()

	localLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5-coder:7b"}]}`))
		case "/v1/chat/completions":
			_, _ = w.Write([]byte(`{"id":"chatcmpl-e2e","object":"chat.completion","model":"qwen2.5-coder:7b","choices":[{"index":0,"message":{"role":"assistant","content":"hello from rocksoul gateway"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer localLLM.Close()

	discoverer := localai.NewDiscoverer([]localai.Target{
		{ID: "ollama", Name: "Ollama", Kind: "ollama", ProbeURL: localLLM.URL + "/api/tags", OpenAIBase: localLLM.URL + "/v1"},
	}, 2*time.Second, nil)

	gw, err := gateway.Open(gateway.Options{
		DataDir:        t.TempDir(),
		LocalDiscovery: discoverer,
	})
	if err != nil {
		t.Fatalf("open gateway: %v", err)
	}
	defer func() { _ = gw.Close() }()

	_, err = gw.AddProvider(context.Background(), gateway.ProviderCreateInput{
		Name:         "local-ollama",
		Kind:         gateway.KindOpenAICompat,
		Tier:         gateway.TierFree,
		BaseURL:      localLLM.URL + "/v1",
		DefaultModel: "qwen2.5-coder:7b",
	})
	if err != nil {
		t.Fatalf("add provider: %v", err)
	}

	verifier, err := auth.NewRIDVerifier(auth.RIDConfig{
		Issuer:   ridSrv.URL,
		Audience: serviceName,
	})
	if err != nil {
		t.Fatalf("new rid verifier: %v", err)
	}

	handler := newHTTPHandler(gw, verifier, serverConfig{
		Port:                  defaultPort,
		AuthMode:              "rid",
		DefaultClassification: gateway.ClassificationCloudEligible,
	})

	// 1. GET /health succeeds without authentication.
	healthRec := httptest.NewRecorder()
	handler.ServeHTTP(healthRec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if healthRec.Code != http.StatusOK || !strings.Contains(healthRec.Body.String(), `"mw-rocksoul-gateway"`) {
		t.Fatalf("GET /health failed: %d %s", healthRec.Code, healthRec.Body.String())
	}

	// 2. All protected endpoints fail closed with 401 without Bearer token.
	protectedChecks := [][2]string{
		{http.MethodGet, "/v1/models"},
		{http.MethodGet, "/v1/usage"},
		{http.MethodGet, "/v1/local-providers"},
		{http.MethodPost, "/v1/chat/completions"},
		{http.MethodPost, "/v1/messages"},
	}
	for _, chk := range protectedChecks {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(chk[0], chk[1], nil))
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s %s without token: expected 401, got %d", chk[0], chk[1], rec.Code)
		}
	}

	// 3. Issue valid RS256 RID token and exercise all 5 protected endpoints.
	token := signTestRS256(t, privKey, ridSrv.URL, serviceName)
	authReq := func(method, path string, body any) *httptest.ResponseRecorder {
		var reader *bytes.Reader
		if body != nil {
			b, _ := json.Marshal(body)
			reader = bytes.NewReader(b)
		} else {
			reader = bytes.NewReader(nil)
		}
		req := httptest.NewRequest(method, path, reader)
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		return rec
	}

	if rec := authReq(http.MethodGet, "/v1/models", nil); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "qwen2.5-coder:7b") {
		t.Fatalf("GET /v1/models failed: %d %s", rec.Code, rec.Body.String())
	}
	if rec := authReq(http.MethodGet, "/v1/local-providers", nil); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"onlineCount":1`) {
		t.Fatalf("GET /v1/local-providers failed: %d %s", rec.Code, rec.Body.String())
	}
	if rec := authReq(http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "qwen2.5-coder:7b",
		"messages": []map[string]string{{"role": "user", "content": "ping"}},
	}); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "hello from rocksoul gateway") {
		t.Fatalf("POST /v1/chat/completions failed: %d %s", rec.Code, rec.Body.String())
	}
	if rec := authReq(http.MethodPost, "/v1/messages", map[string]any{
		"model":      "qwen2.5-coder:7b",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": "ping anthropic"}},
	}); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "hello from rocksoul gateway") {
		t.Fatalf("POST /v1/messages failed: %d %s", rec.Code, rec.Body.String())
	}
	if rec := authReq(http.MethodGet, "/v1/usage", nil); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"totalRequests":2`) {
		t.Fatalf("GET /v1/usage failed: %d %s", rec.Code, rec.Body.String())
	}
}

func signTestRS256(t *testing.T, priv *rsa.PrivateKey, issuer, audience string) string {
	t.Helper()
	hBytes, _ := json.Marshal(map[string]any{"alg": "RS256", "typ": "at+jwt", "kid": "gw-key-1"})
	pBytes, _ := json.Marshal(map[string]any{
		"iss":       issuer,
		"sub":       "rid_tester_1",
		"aud":       audience,
		"exp":       time.Now().Add(time.Hour).Unix(),
		"token_use": "access",
	})
	input := base64.RawURLEncoding.EncodeToString(hBytes) + "." + base64.RawURLEncoding.EncodeToString(pBytes)
	digest := sha256.Sum256([]byte(input))
	sig, err := rsa.SignPKCS1v15(rand.Reader, priv, crypto.SHA256, digest[:])
	if err != nil {
		t.Fatalf("sign token: %v", err)
	}
	return input + "." + base64.RawURLEncoding.EncodeToString(sig)
}
