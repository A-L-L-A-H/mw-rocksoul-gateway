package localai_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/localai"
)

func TestDefaultTargetsCoverOllamaLMStudioLlamaCpp(t *testing.T) {
	targets := localai.DefaultTargets()
	if len(targets) != 3 {
		t.Fatalf("expected 3 default targets, got %d", len(targets))
	}

	expectedPorts := map[string]string{
		"ollama":   ":11434",
		"lmstudio": ":1234",
		"llamacpp": ":8080",
	}
	for _, target := range targets {
		wantPort, ok := expectedPorts[target.ID]
		if !ok {
			t.Fatalf("unexpected default target id %q", target.ID)
		}
		if !strings.Contains(target.ProbeURL, wantPort) {
			t.Fatalf("target %s probeUrl %q missing expected port %s", target.ID, target.ProbeURL, wantPort)
		}
		if !localai.IsLoopbackEndpointURL(target.ProbeURL) || !localai.IsLoopbackEndpointURL(target.OpenAIBase) {
			t.Fatalf("default target %s must pass loopback guard", target.ID)
		}
		delete(expectedPorts, target.ID)
	}
	if len(expectedPorts) != 0 {
		t.Fatalf("missing default targets: %v", expectedPorts)
	}
}

func TestLoopbackURLValidation(t *testing.T) {
	validURLs := []string{
		"http://127.0.0.1:11434/api/tags",
		"http://localhost:1234/v1/models",
		"http://[::1]:8080/v1/models",
		"https://127.0.0.1:8443/v1",
	}
	for _, raw := range validURLs {
		if err := func() error { _, e := localai.ValidateLoopbackEndpointURL(raw); return e }(); err != nil {
			t.Errorf("expected valid loopback URL %q, got err: %v", raw, err)
		}
	}

	rejectedURLs := []string{
		"",
		"https://api.openai.com/v1",
		"http://10.0.0.5:11434/api/tags",
		"http://192.168.1.10:1234/v1/models",
		"http://0.0.0.0:8080/v1/models",
		"http://169.254.169.254/latest/meta-data",
		"http://user:pass@127.0.0.1:11434/api/tags",
		"http://127.0.0.1:11434/api/tags#frag",
		"file:///etc/passwd",
		"ftp://127.0.0.1:21/models",
	}
	for _, raw := range rejectedURLs {
		if localai.IsLoopbackEndpointURL(raw) {
			t.Errorf("expected non-loopback/invalid URL %q to be rejected", raw)
		}
	}
}

func TestDiscoverLoopbackProvidersAndRejectExternalTargets(t *testing.T) {
	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5-coder:14b","size":9000000000,"details":{"family":"qwen2"}}]}`))
	}))
	defer ollamaSrv.Close()

	lmStudioSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"llama-3.2-3b-instruct"}]}`))
	}))
	defer lmStudioSrv.Close()

	llamaCppSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"gemma-3-12b-it.gguf"}]}`))
	}))
	defer llamaCppSrv.Close()

	redirectSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://example.com/v1/models", http.StatusFound)
	}))
	defer redirectSrv.Close()

	targets := []localai.Target{
		{ID: "ollama", Name: "Ollama", Kind: "ollama", ProbeURL: ollamaSrv.URL + "/api/tags", OpenAIBase: ollamaSrv.URL + "/v1"},
		{ID: "lmstudio", Name: "LM Studio", Kind: "lmstudio", ProbeURL: lmStudioSrv.URL + "/v1/models", OpenAIBase: lmStudioSrv.URL + "/v1"},
		{ID: "llamacpp", Name: "llama.cpp", Kind: "llamacpp", ProbeURL: llamaCppSrv.URL + "/v1/models", OpenAIBase: llamaCppSrv.URL + "/v1"},
		{ID: "redirect-escape", Name: "Redirect", Kind: "llamacpp", ProbeURL: redirectSrv.URL + "/v1/models"},
		{ID: "external-cloud", Name: "External", Kind: "lmstudio", ProbeURL: "https://api.openai.com/v1/models"},
	}

	discoverer := localai.NewDiscoverer(targets, 2*time.Second, nil)
	result := discoverer.Discover(context.Background())

	if result.OnlineCount != 3 {
		t.Fatalf("expected 3 online local providers, got %d (%+v)", result.OnlineCount, result.Providers)
	}
	if len(result.Models) != 3 {
		t.Fatalf("expected 3 discovered models, got %d", len(result.Models))
	}
	if result.RecommendedDevModel != "qwen2.5-coder:14b" {
		t.Fatalf("expected recommended dev model qwen2.5-coder:14b, got %q", result.RecommendedDevModel)
	}
	if result.RecommendedWorkspaceModel == "" {
		t.Fatalf("expected non-empty recommended workspace model")
	}

	if result.Providers[3].Online || !strings.Contains(result.Providers[3].Error, "HTTP 302") {
		t.Fatalf("expected redirect target to fail without following external redirect, got %+v", result.Providers[3])
	}
	if result.Providers[4].Online || !strings.Contains(result.Providers[4].Error, "non-loopback") {
		t.Fatalf("expected external target to be blocked by loopback guard, got %+v", result.Providers[4])
	}
}
