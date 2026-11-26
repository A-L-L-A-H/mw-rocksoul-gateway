package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/gateway"
	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/localai"
)

func setupTestGateway(t *testing.T, opts ...func(*gateway.Options)) (*gateway.Gateway, http.Handler) {
	t.Helper()
	cfg := gateway.Options{
		DataDir: t.TempDir(),
	}
	for _, fn := range opts {
		fn(&cfg)
	}
	gw, err := gateway.Open(cfg)
	if err != nil {
		t.Fatalf("open test gateway: %v", err)
	}
	t.Cleanup(func() { _ = gw.Close() })
	return gw, gw.Handler()
}

func doJSONRequest(t *testing.T, handler http.Handler, method, target string, payload any, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var bodyReader io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal payload: %v", err)
		}
		bodyReader = bytes.NewReader(encoded)
	}
	httpReq := httptest.NewRequest(method, target, bodyReader)
	httpReq.Header.Set("Content-Type", "application/json")
	for key, val := range headers {
		httpReq.Header.Set(key, val)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httpReq)
	return recorder
}

func TestRTKCompressionSavesOver25PercentTokens(t *testing.T) {
	var logBuilder strings.Builder
	logBuilder.WriteString("\x1b[32m==================================================\x1b[0m\n")
	logBuilder.WriteString("diff --git a/internal/app.go b/internal/app.go\n")
	logBuilder.WriteString("index 1a2b3c4d..9e8f7a6b 100644\n")
	logBuilder.WriteString("old mode 100644\nnew mode 100755\n")
	for idx := 0; idx < 45; idx++ {
		logBuilder.WriteString(fmt.Sprintf("\x1b[36m=== RUN   TestWorkerCase_%02d\x1b[0m\n", idx))
		logBuilder.WriteString(fmt.Sprintf("\x1b[32m--- PASS: TestWorkerCase_%02d (0.00s)\x1b[0m\n", idx))
	}
	for idx := 0; idx < 25; idx++ {
		logBuilder.WriteString("\x1b[33m2026-11-26T12:00:01Z INFO polling worker queue...\x1b[0m\n\n\n")
	}
	logBuilder.WriteString("\x1b[31m--------------------------------------------------\x1b[0m\n")
	logBuilder.WriteString("\x1b[31mERROR: failed to bind listener on port 8080: address already in use\x1b[0m\n")

	noisyLog := logBuilder.String()
	rawTokens := len(noisyLog) / 4

	toolMessages := []gateway.ChatMessage{
		{Role: "user", Content: "Inspect the test failure."},
		{Role: "tool", ToolCallID: "call_01", Content: noisyLog},
	}
	compressedMsgs, tokensSaved := gateway.CompressMessages(toolMessages)
	if len(compressedMsgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(compressedMsgs))
	}
	compressedTool, ok := compressedMsgs[1].Content.(string)
	if !ok {
		t.Fatalf("expected string tool content")
	}
	if strings.Contains(compressedTool, "\x1b[") {
		t.Fatalf("ANSI escape codes were not stripped: %q", compressedTool)
	}
	if !strings.Contains(compressedTool, "ERROR: failed to bind listener on port 8080") {
		t.Fatalf("critical tail error line was lost during compression: %q", compressedTool)
	}
	if !strings.Contains(compressedTool, "[x25 repeated]") {
		t.Fatalf("expected consecutive repeated line deduplication marker, got: %q", compressedTool)
	}
	savingsPct := float64(tokensSaved) / float64(rawTokens) * 100.0
	if savingsPct <= 25.0 {
		t.Fatalf("expected >25%% token savings, got %.1f%% (saved=%d raw=%d)", savingsPct, tokensSaved, rawTokens)
	}

	// Anthropic tool_result content block
	anthToolMessages := []gateway.ChatMessage{
		{
			Role: "user",
			Content: []any{
				map[string]any{
					"type":        "tool_result",
					"tool_use_id": "toolu_01",
					"content":     noisyLog,
				},
			},
		},
	}
	anthCompressed, anthSaved := gateway.CompressMessages(anthToolMessages)
	if anthSaved <= rawTokens/4 {
		t.Fatalf("expected >25%% savings on Anthropic tool_result block, got saved=%d raw=%d", anthSaved, rawTokens)
	}
	blocks, ok := anthCompressed[0].Content.([]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("expected compressed Anthropic content blocks")
	}
	blockMap := blocks[0].(map[string]any)
	blockContent, _ := blockMap["content"].(string)
	if strings.Contains(blockContent, "\x1b[") || !strings.Contains(blockContent, "ERROR: failed to bind listener") {
		t.Fatalf("Anthropic tool_result compression invalid: %q", blockContent)
	}
}

func TestPIISecretSnifferRedactsAllSensitivePatternsWithoutMutatingInput(t *testing.T) {
	sensitivePrompt := strings.Join([]string{
		"Authorization: Bearer superSecretBearerToken99999",
		"OpenAI key: sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456",
		"AWS key: AKIAIOSFODNN7EXAMPLE",
		"JWT: eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzNCJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
		"DB: postgres://admin:TopSecretPass99@db.internal:5432/prod",
		"password=myUltraSecretPassword42",
		"Contact alice.smith@corp.example.com or +62 812-3456-7890",
		"SSN 123-45-6789 and Card 4532-0151-1283-0366",
		"NIK: 3201012345678901",
		"-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAtestkeymaterial\n-----END RSA PRIVATE KEY-----",
	}, "\n")

	original := []gateway.ChatMessage{
		{Role: "user", Content: sensitivePrompt},
	}
	sanitized, report := gateway.SniffAndRedactMessages(original)
	if report.RedactedCount < 10 {
		t.Fatalf("expected >=10 redactions, got %d (%+v)", report.RedactedCount, report.Categories)
	}

	// Verify caller slice was not mutated.
	if original[0].Content.(string) != sensitivePrompt {
		t.Fatalf("SniffAndRedactMessages mutated caller slice")
	}

	out := sanitized[0].Content.(string)
	forbidden := []string{
		"superSecretBearerToken99999",
		"sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456",
		"AKIAIOSFODNN7EXAMPLE",
		"eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9",
		"TopSecretPass99",
		"myUltraSecretPassword42",
		"alice.smith@corp.example.com",
		"812-3456-7890",
		"123-45-6789",
		"4532-0151-1283-0366",
		"3201012345678901",
		"MIIEpAIBAAKCAQEAtestkeymaterial",
	}
	for _, secret := range forbidden {
		if strings.Contains(out, secret) {
			t.Fatalf("secret/PII %q leaked in sanitized output:\n%s", secret, out)
		}
	}

	expectedTokens := []string{
		"[REDACTED_PRIVATE_KEY]",
		"[REDACTED_JWT]",
		"[REDACTED_TOKEN]",
		"[REDACTED_API_KEY]",
		"[REDACTED_AWS_KEY]",
		"[REDACTED]:[REDACTED]@",
		"[REDACTED_SECRET]",
		"[REDACTED_EMAIL]",
		"[REDACTED_SSN]",
		"[REDACTED_CARD]",
		"[REDACTED_PHONE]",
		"[REDACTED_NIK]",
	}
	for _, marker := range expectedTokens {
		if !strings.Contains(out, marker) {
			t.Errorf("expected redaction marker %q in output:\n%s", marker, out)
		}
	}
}

func TestDataClassificationLocalOnlyBlocksExternalEgressAndAllowsLoopback(t *testing.T) {
	gw, handler := setupTestGateway(t)

	// Register ONLY a non-loopback cloud provider (RFC 5737 TEST-NET address; must never be dialed).
	_, err := gw.AddProvider(context.Background(), gateway.ProviderCreateInput{
		Name:         "external-cloud",
		Kind:         gateway.KindOpenAICompat,
		Tier:         gateway.TierFree,
		BaseURL:      "https://api.openai.com/v1",
		APIKey:       "sk-cloud-key-test",
		DefaultModel: "gpt-4o-mini",
		Models:       []string{"gpt-4o-mini"},
		Priority:     1,
	})
	if err != nil {
		t.Fatalf("add external provider: %v", err)
	}

	// 1. local_only via header MUST strictly block external egress with 403.
	blockedRec := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "gpt-4o-mini",
		"messages": []map[string]string{{"role": "user", "content": "private payroll secret"}},
	}, map[string]string{
		"X-Rocksoul-Data-Classification": "local_only",
	})
	if blockedRec.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for local_only without local provider, got %d: %s", blockedRec.Code, blockedRec.Body.String())
	}
	if !strings.Contains(blockedRec.Body.String(), "local_only_egress_blocked") {
		t.Fatalf("expected local_only_egress_blocked code, got: %s", blockedRec.Body.String())
	}

	// 2. local_only via request body on /v1/messages MUST also block external egress with 403.
	blockedAnth := doJSONRequest(t, handler, http.MethodPost, "/v1/messages", map[string]any{
		"model":               "gpt-4o-mini",
		"data_classification": "local_only",
		"messages":            []map[string]any{{"role": "user", "content": "top secret local code"}},
	}, nil)
	if blockedAnth.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for Anthropic local_only without local provider, got %d: %s", blockedAnth.Code, blockedAnth.Body.String())
	}

	// 3. Invalid classification fails closed with 400.
	invalidClassRec := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":               "gpt-4o-mini",
		"data_classification": "yolo_public",
		"messages":            []map[string]string{{"role": "user", "content": "hello"}},
	}, nil)
	if invalidClassRec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for invalid classification, got %d: %s", invalidClassRec.Code, invalidClassRec.Body.String())
	}

	// 4. Register a loopback local provider: local_only MUST route strictly to loopback and succeed.
	var localHits atomic.Int32
	var receivedPrompt string
	localSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		localHits.Add(1)
		var req gateway.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Messages) > 0 {
			receivedPrompt, _ = req.Messages[0].Content.(string)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-local-1","object":"chat.completion","model":"qwen2.5-coder:14b","choices":[{"index":0,"message":{"role":"assistant","content":"local air-gapped answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}`))
	}))
	defer localSrv.Close()

	_, err = gw.AddProvider(context.Background(), gateway.ProviderCreateInput{
		Name:         "local-ollama",
		Kind:         gateway.KindOpenAICompat,
		Tier:         gateway.TierPaid, // Lower priority/worse tier than external-cloud to prove air-gap skips external-cloud!
		BaseURL:      localSrv.URL + "/v1",
		DefaultModel: "qwen2.5-coder:14b",
		Models:       []string{"qwen2.5-coder:14b", "gpt-4o-mini"},
		Priority:     999,
	})
	if err != nil {
		t.Fatalf("add local provider: %v", err)
	}

	localRec := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":               "gpt-4o-mini",
		"data_classification": "local_only",
		"messages":            []map[string]string{{"role": "user", "content": "Local key sk-local-unredacted-99999999"}},
	}, nil)
	if localRec.Code != http.StatusOK {
		t.Fatalf("expected 200 routing to local loopback provider, got %d: %s", localRec.Code, localRec.Body.String())
	}
	if localRec.Header().Get("X-Rocksoul-Provider") != "local-ollama" {
		t.Fatalf("expected local-ollama provider, got %q", localRec.Header().Get("X-Rocksoul-Provider"))
	}
	if localRec.Header().Get("X-Rocksoul-Classification") != "local_only" {
		t.Fatalf("expected X-Rocksoul-Classification=local_only, got %q", localRec.Header().Get("X-Rocksoul-Classification"))
	}
	if localHits.Load() != 1 {
		t.Fatalf("expected 1 local hit, got %d", localHits.Load())
	}
	if !strings.Contains(receivedPrompt, "sk-local-unredacted-99999999") {
		t.Fatalf("expected local_only request to remain unredacted for local inference, got %q", receivedPrompt)
	}

	// Verify /v1/usage tracks both blocked egress attempts and successful local tokens.
	usageRec := doJSONRequest(t, handler, http.MethodGet, "/v1/usage", nil, nil)
	if usageRec.Code != http.StatusOK {
		t.Fatalf("GET /v1/usage failed: %d %s", usageRec.Code, usageRec.Body.String())
	}
	var usage gateway.UsageSummary
	if err := json.Unmarshal(usageRec.Body.Bytes(), &usage); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}
	if usage.LocalOnlyBlockedEgress != 2 {
		t.Fatalf("expected 2 blocked local_only egress events, got %d", usage.LocalOnlyBlockedEgress)
	}
	if usage.TotalTokens != 17 {
		t.Fatalf("expected 17 total tokens, got %d", usage.TotalTokens)
	}
}

func TestCloudEligibleAppliesPIISnifferAndRTKBeforeEgress(t *testing.T) {
	gw, handler := setupTestGateway(t)

	var receivedMessages []gateway.ChatMessage
	cloudSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req gateway.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		receivedMessages = req.Messages
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-cloud-1","object":"chat.completion","model":"llama-3.3-70b-versatile","choices":[{"index":0,"message":{"role":"assistant","content":"safe cloud reply"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":6,"total_tokens":26}}`))
	}))
	defer cloudSrv.Close()

	_, err := gw.AddProvider(context.Background(), gateway.ProviderCreateInput{
		Name:         "cloud-mock",
		Kind:         gateway.KindOpenAICompat,
		Tier:         gateway.TierFree,
		BaseURL:      cloudSrv.URL + "/v1",
		DefaultModel: "llama-3.3-70b-versatile",
	})
	if err != nil {
		t.Fatalf("add provider: %v", err)
	}

	var noisyTool strings.Builder
	for i := 0; i < 35; i++ {
		noisyTool.WriteString(fmt.Sprintf("\x1b[32m=== RUN   TestCase_%02d\x1b[0m\n\x1b[32m--- PASS: TestCase_%02d (0.00s)\x1b[0m\n", i, i))
	}
	noisyTool.WriteString("ERROR: failed auth with sk-live-secretCloudLeak998877 and email ops@company.org\n")

	rec := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":               "llama-3.3-70b-versatile",
		"data_classification": "cloud_eligible",
		"messages": []map[string]any{
			{"role": "user", "content": "Debug this failure for alice@example.com"},
			{"role": "tool", "tool_call_id": "call_9", "content": noisyTool.String()},
		},
	}, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("X-Rocksoul-Classification") != "cloud_eligible" {
		t.Fatalf("expected cloud_eligible header, got %q", rec.Header().Get("X-Rocksoul-Classification"))
	}
	if rec.Header().Get("X-Rocksoul-Redactions") == "" || rec.Header().Get("X-Rocksoul-Redactions") == "0" {
		t.Fatalf("expected non-zero X-Rocksoul-Redactions header, got %q", rec.Header().Get("X-Rocksoul-Redactions"))
	}
	if rec.Header().Get("X-Rocksoul-RTK-Saved") == "" || rec.Header().Get("X-Rocksoul-RTK-Saved") == "0" {
		t.Fatalf("expected non-zero X-Rocksoul-RTK-Saved header, got %q", rec.Header().Get("X-Rocksoul-RTK-Saved"))
	}

	if len(receivedMessages) != 2 {
		t.Fatalf("expected 2 upstream messages, got %d", len(receivedMessages))
	}
	userOut, _ := receivedMessages[0].Content.(string)
	toolOut, _ := receivedMessages[1].Content.(string)
	if strings.Contains(userOut, "alice@example.com") || !strings.Contains(userOut, "[REDACTED_EMAIL]") {
		t.Fatalf("user email was not redacted before egress: %q", userOut)
	}
	if strings.Contains(toolOut, "sk-live-secretCloudLeak998877") || strings.Contains(toolOut, "ops@company.org") {
		t.Fatalf("tool secret/PII leaked to cloud upstream: %q", toolOut)
	}
	if strings.Contains(toolOut, "\x1b[") || !strings.Contains(toolOut, "ERROR: failed auth") {
		t.Fatalf("RTK compression did not clean ANSI or dropped diagnostic line: %q", toolOut)
	}

	// Verify /v1/usage captured PII redactions & RTK savings.
	usageRec := doJSONRequest(t, handler, http.MethodGet, "/v1/usage", nil, nil)
	var usage gateway.UsageSummary
	_ = json.Unmarshal(usageRec.Body.Bytes(), &usage)
	if usage.TotalPIIRedactions < 3 || usage.TotalRTKTokensSaved <= 0 {
		t.Fatalf("expected redactions >=3 and rtk saved >0 in /v1/usage, got %+v", usage)
	}
}

func TestLocalOnlyHostDefaultCannotBeDowngradedByRequest(t *testing.T) {
	gw, handler := setupTestGateway(t, func(o *gateway.Options) {
		o.DefaultClassification = gateway.ClassificationLocalOnly
	})

	_, err := gw.AddProvider(context.Background(), gateway.ProviderCreateInput{
		Name:         "cloud-only",
		Kind:         gateway.KindOpenAICompat,
		Tier:         gateway.TierFree,
		BaseURL:      "https://api.groq.com/openai/v1",
		DefaultModel: "llama-3.3-70b-versatile",
	})
	if err != nil {
		t.Fatalf("add provider: %v", err)
	}

	// Request tries to override with cloud_eligible, but host-level air-gap must still block.
	rec := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":               "llama-3.3-70b-versatile",
		"data_classification": "cloud_eligible",
		"messages":            []map[string]string{{"role": "user", "content": "attempt bypass"}},
	}, map[string]string{
		"X-Rocksoul-Data-Classification": "cloud_eligible",
	})
	if rec.Code != http.StatusForbidden {
		t.Fatalf("expected host-level local_only to block cloud_eligible downgrade with 403, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestLocalProvidersEndpointAndAutoDiscovery(t *testing.T) {
	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[{"name":"deepseek-coder:6.7b","size":4000000000}]}`))
	}))
	defer ollamaSrv.Close()

	discoverer := localai.NewDiscoverer([]localai.Target{
		{ID: "ollama", Name: "Ollama", Kind: "ollama", ProbeURL: ollamaSrv.URL + "/api/tags", OpenAIBase: ollamaSrv.URL + "/v1"},
	}, 2*time.Second, nil)

	_, handler := setupTestGateway(t, func(o *gateway.Options) {
		o.LocalDiscovery = discoverer
	})

	rec := doJSONRequest(t, handler, http.MethodGet, "/v1/local-providers", nil, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/local-providers failed: %d %s", rec.Code, rec.Body.String())
	}
	var result localai.DiscoveryResult
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal discovery: %v", err)
	}
	if result.OnlineCount != 1 || len(result.Models) != 1 || result.Models[0].ID != "deepseek-coder:6.7b" {
		t.Fatalf("unexpected local providers result: %+v", result)
	}

	modelsRec := doJSONRequest(t, handler, http.MethodGet, "/v1/models", nil, nil)
	if modelsRec.Code != http.StatusOK || !strings.Contains(modelsRec.Body.String(), `"deepseek-coder:6.7b"`) {
		t.Fatalf("GET /v1/models missing discovered local model: %d %s", modelsRec.Code, modelsRec.Body.String())
	}
}

func TestFailoverOn429SetsCooldownAndRoutesToBackup(t *testing.T) {
	_, app := setupTestGateway(t)

	var primaryHits atomic.Int32
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded"}}`))
	}))
	defer primaryServer.Close()

	var backupHits atomic.Int32
	backupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		backupHits.Add(1)
		if r.Header.Get("Authorization") != "Bearer backup-secret-key" {
			http.Error(w, "missing auth", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-backup-1",
			"object":"chat.completion",
			"created":1730000000,
			"model":"llama-3.3-70b-versatile",
			"choices":[{"index":0,"message":{"role":"assistant","content":"served by backup"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":18,"completion_tokens":7,"total_tokens":25}
		}`))
	}))
	defer backupServer.Close()

	primaryRec := doJSONRequest(t, app, http.MethodPost, "/api/ai/providers", map[string]any{
		"name":         "primary-free",
		"kind":         "openai_compat",
		"tier":         "free",
		"baseUrl":      primaryServer.URL + "/v1",
		"apiKey":       "primary-secret-key",
		"defaultModel": "llama-3.3-70b-versatile",
		"models":       []string{"llama-3.3-70b-versatile"},
		"priority":     10,
	}, nil)
	if primaryRec.Code != http.StatusCreated {
		t.Fatalf("create primary provider failed: %d %s", primaryRec.Code, primaryRec.Body.String())
	}

	backupRec := doJSONRequest(t, app, http.MethodPost, "/api/ai/providers", map[string]any{
		"name":         "backup-cheap",
		"kind":         "openai_compat",
		"tier":         "cheap",
		"baseUrl":      backupServer.URL + "/v1",
		"apiKey":       "backup-secret-key",
		"defaultModel": "llama-3.3-70b-versatile",
		"models":       []string{"llama-3.3-70b-versatile"},
		"priority":     20,
	}, nil)
	if backupRec.Code != http.StatusCreated {
		t.Fatalf("create backup provider failed: %d %s", backupRec.Code, backupRec.Body.String())
	}

	highTemp := 0.7
	firstRec := doJSONRequest(t, app, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":       "llama-3.3-70b-versatile",
		"temperature": highTemp,
		"messages":    []map[string]string{{"role": "user", "content": "hello failover"}},
	}, nil)
	if firstRec.Code != http.StatusOK {
		t.Fatalf("expected 200 after failover, got %d: %s", firstRec.Code, firstRec.Body.String())
	}
	if firstRec.Header().Get("X-Rocksoul-Provider") != "backup-cheap" {
		t.Fatalf("expected X-Rocksoul-Provider=backup-cheap, got %q", firstRec.Header().Get("X-Rocksoul-Provider"))
	}
	if !strings.Contains(firstRec.Body.String(), "served by backup") {
		t.Fatalf("unexpected body: %s", firstRec.Body.String())
	}
	if primaryHits.Load() != 1 || backupHits.Load() != 1 {
		t.Fatalf("expected primary=1 backup=1 hits, got primary=%d backup=%d", primaryHits.Load(), backupHits.Load())
	}

	listRec := doJSONRequest(t, app, http.MethodGet, "/api/ai/providers", nil, nil)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list providers failed: %d", listRec.Code)
	}
	var listEnvelope struct {
		Providers []gateway.Provider `json:"providers"`
	}
	_ = json.Unmarshal(listRec.Body.Bytes(), &listEnvelope)
	foundCooldown := false
	for _, provider := range listEnvelope.Providers {
		if provider.Name == "primary-free" && provider.InCooldown && provider.CooldownUntil != "" {
			foundCooldown = true
		}
	}
	if !foundCooldown {
		t.Fatalf("expected primary-free provider to be marked inCooldown, got: %s", listRec.Body.String())
	}

	secondRec := doJSONRequest(t, app, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":       "llama-3.3-70b-versatile",
		"temperature": highTemp,
		"messages":    []map[string]string{{"role": "user", "content": "second request during cooldown"}},
	}, nil)
	if secondRec.Code != http.StatusOK {
		t.Fatalf("second request failed: %d %s", secondRec.Code, secondRec.Body.String())
	}
	if primaryHits.Load() != 1 || backupHits.Load() != 2 {
		t.Fatalf("primary should have been skipped during cooldown; got primary=%d backup=%d", primaryHits.Load(), backupHits.Load())
	}
}

func TestExactPromptCacheServesSecondRequestWithZeroUpstreamCalls(t *testing.T) {
	_, app := setupTestGateway(t)

	var upstreamHits atomic.Int32
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-cache-1",
			"object":"chat.completion",
			"created":1730000100,
			"model":"gpt-oss-120b",
			"choices":[{"index":0,"message":{"role":"assistant","content":"deterministic answer 42"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":30,"completion_tokens":12,"total_tokens":42}
		}`))
	}))
	defer upstreamServer.Close()

	createRec := doJSONRequest(t, app, http.MethodPost, "/api/ai/providers", map[string]any{
		"name":         "cache-test-provider",
		"kind":         "openai_compat",
		"tier":         "free",
		"baseUrl":      upstreamServer.URL + "/v1",
		"defaultModel": "gpt-oss-120b",
	}, nil)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create provider failed: %d", createRec.Code)
	}

	reqBody := map[string]any{
		"model":       "gpt-oss-120b",
		"temperature": 0.1,
		"messages":    []map[string]string{{"role": "user", "content": "What is 6 * 7?"}},
	}

	firstRec := doJSONRequest(t, app, http.MethodPost, "/v1/chat/completions", reqBody, nil)
	if firstRec.Code != http.StatusOK || firstRec.Header().Get("X-Rocksoul-Cache") != "MISS" || upstreamHits.Load() != 1 {
		t.Fatalf("first completion failed: code=%d cache=%q hits=%d", firstRec.Code, firstRec.Header().Get("X-Rocksoul-Cache"), upstreamHits.Load())
	}

	secondRec := doJSONRequest(t, app, http.MethodPost, "/v1/chat/completions", reqBody, nil)
	if secondRec.Code != http.StatusOK || secondRec.Header().Get("X-Rocksoul-Cache") != "HIT" || upstreamHits.Load() != 1 {
		t.Fatalf("expected cache HIT with 0 new upstream calls: code=%d cache=%q hits=%d", secondRec.Code, secondRec.Header().Get("X-Rocksoul-Cache"), upstreamHits.Load())
	}

	statsRec := doJSONRequest(t, app, http.MethodGet, "/api/ai/stats", nil, nil)
	var stats gateway.GatewayStats
	_ = json.Unmarshal(statsRec.Body.Bytes(), &stats)
	if stats.CacheHits != 1 || stats.CacheTokensSaved != 42 || stats.TotalRequests != 2 {
		t.Fatalf("unexpected stats: total=%d cacheHits=%d cacheTokensSaved=%d", stats.TotalRequests, stats.CacheHits, stats.CacheTokensSaved)
	}

	purgeRec := doJSONRequest(t, app, http.MethodDelete, "/api/ai/cache", nil, nil)
	if purgeRec.Code != http.StatusOK || !strings.Contains(purgeRec.Body.String(), `"purged":1`) {
		t.Fatalf("cache purge failed: %d %s", purgeRec.Code, purgeRec.Body.String())
	}

	thirdRec := doJSONRequest(t, app, http.MethodPost, "/v1/chat/completions", reqBody, nil)
	if thirdRec.Header().Get("X-Rocksoul-Cache") != "MISS" || upstreamHits.Load() != 2 {
		t.Fatalf("expected cache MISS and upstream hit after purge, header=%q hits=%d", thirdRec.Header().Get("X-Rocksoul-Cache"), upstreamHits.Load())
	}
}

func TestAnthropicMessagesTranslatesClaudeCodeViaMagpieRoute(t *testing.T) {
	_, app := setupTestGateway(t)

	var capturedOpenAIReq gateway.ChatCompletionRequest
	openAIBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		rawBody, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(rawBody, &capturedOpenAIReq)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-magpie-99",
			"object":"chat.completion",
			"created":1730000200,
			"model":"qwen/qwen3-coder:free",
			"choices":[{"index":0,"message":{"role":"assistant","content":"Refactored cleanly via Magpie!"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":44,"completion_tokens":11,"total_tokens":55}
		}`))
	}))
	defer openAIBackend.Close()

	createRec := doJSONRequest(t, app, http.MethodPost, "/api/ai/providers", map[string]any{
		"name":         "openrouter-free",
		"kind":         "openai_compat",
		"tier":         "free",
		"baseUrl":      openAIBackend.URL + "/v1",
		"apiKey":       "sk-or-test-key",
		"defaultModel": "qwen/qwen3-coder:free",
		"models":       []string{"qwen/qwen3-coder:free"},
	}, nil)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create provider failed: %d", createRec.Code)
	}

	routeRec := doJSONRequest(t, app, http.MethodPut, "/api/ai/agents/claude-code", map[string]any{
		"targetModel": "qwen/qwen3-coder:free",
		"tierOrder":   "free,cheap,subscription,paid",
		"rtkEnabled":  true,
	}, nil)
	if routeRec.Code != http.StatusOK {
		t.Fatalf("update claude-code route failed: %d %s", routeRec.Code, routeRec.Body.String())
	}

	anthRec := doJSONRequest(t, app, http.MethodPost, "/v1/messages", map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"system":     "You are Claude Code.",
		"max_tokens": 512,
		"messages": []map[string]any{
			{"role": "user", "content": "Explain Magpie routing in one line."},
		},
	}, nil)
	if anthRec.Code != http.StatusOK {
		t.Fatalf("POST /v1/messages failed: %d %s", anthRec.Code, anthRec.Body.String())
	}

	if capturedOpenAIReq.Model != "qwen/qwen3-coder:free" {
		t.Fatalf("expected Magpie claude-code route model qwen/qwen3-coder:free, got %q", capturedOpenAIReq.Model)
	}
	if len(capturedOpenAIReq.Messages) != 2 || capturedOpenAIReq.Messages[0].Role != "system" || capturedOpenAIReq.Messages[1].Role != "user" {
		t.Fatalf("unexpected translated OpenAI messages: %+v", capturedOpenAIReq.Messages)
	}

	var anthResp gateway.AnthropicMessageResponse
	if err := json.Unmarshal(anthRec.Body.Bytes(), &anthResp); err != nil {
		t.Fatalf("unmarshal anthropic response: %v", err)
	}
	if anthResp.Type != "message" || anthResp.Role != "assistant" || len(anthResp.Content) != 1 || anthResp.Content[0].Text != "Refactored cleanly via Magpie!" {
		t.Fatalf("unexpected anthropic response: %+v", anthResp)
	}
	if anthResp.Usage.InputTokens != 44 || anthResp.Usage.OutputTokens != 11 {
		t.Fatalf("unexpected anthropic usage: %+v", anthResp.Usage)
	}
	if anthRec.Header().Get("X-MW-Agent") != "claude-code" {
		t.Fatalf("expected X-MW-Agent=claude-code, got %q", anthRec.Header().Get("X-MW-Agent"))
	}
}

func TestFreeLLMAPISnifferDetectsNewSHAAndIndexesModels(t *testing.T) {
	gw, app := setupTestGateway(t)

	currentSHA := "abc111222333444555666777888999000aaabbb1"
	upstreamMock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/tashfeenahmed/freellmapi/commits/main":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"sha":%q}`, currentSHA)
		case "/tashfeenahmed/freellmapi/main/server/src/providers/index.ts":
			_, _ = w.Write([]byte(`
register(new OpenAICompatProvider({
  platform: 'groq',
  name: 'Groq',
  baseUrl: 'https://api.groq.com/openai/v1',
}));
register(new OpenAICompatProvider({
  platform: 'powertokens',
  name: 'PowerTokens',
  baseUrl: 'https://api.powertokens.ai/v1',
}));
`))
		case "/tashfeenahmed/freellmapi/main/server/src/db/model-pricing.ts":
			_, _ = w.Write([]byte(`
export const MODEL_PRICING = [
  ['groq', 'llama-4-maverick-99b-sniffed', 0.10, 0.30],
  ['powertokens', 'glm-4.7-flash-sniffed', 0.06, 0.40],
];
`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstreamMock.Close()

	gw.GitHubAPIBase = upstreamMock.URL
	gw.RawBase = upstreamMock.URL

	firstRec := doJSONRequest(t, app, http.MethodPost, "/api/ai/catalog/sniff", nil, nil)
	if firstRec.Code != http.StatusOK {
		t.Fatalf("POST /api/ai/catalog/sniff failed: %d %s", firstRec.Code, firstRec.Body.String())
	}
	var firstResult gateway.SniffResult
	_ = json.Unmarshal(firstRec.Body.Bytes(), &firstResult)
	if !firstResult.Updated || firstResult.UpstreamSHA != currentSHA || firstResult.PreviousSHA != "" {
		t.Fatalf("unexpected first sniff result: %+v", firstResult)
	}
	if firstResult.ProvidersFound < 10 || firstResult.ModelsIndexed < 35 {
		t.Fatalf("expected merged seed + upstream catalog counts, got providers=%d models=%d", firstResult.ProvidersFound, firstResult.ModelsIndexed)
	}

	searchRec := doJSONRequest(t, app, http.MethodGet, "/api/ai/catalog?q=glm-4.7-flash-sniffed&freeOnly=true", nil, nil)
	if searchRec.Code != http.StatusOK || !strings.Contains(searchRec.Body.String(), "https://api.powertokens.ai/v1") {
		t.Fatalf("sniffed model not indexed in catalog: %d %s", searchRec.Code, searchRec.Body.String())
	}

	modelsRec := doJSONRequest(t, app, http.MethodGet, "/v1/models", nil, nil)
	if modelsRec.Code != http.StatusOK || !strings.Contains(modelsRec.Body.String(), `"agent/claude-code"`) || !strings.Contains(modelsRec.Body.String(), `"glm-4.7-flash-sniffed"`) {
		t.Fatalf("GET /v1/models missing expected entries: %d %s", modelsRec.Code, modelsRec.Body.String())
	}

	secondRec := doJSONRequest(t, app, http.MethodPost, "/api/ai/catalog/sniff", nil, nil)
	var secondResult gateway.SniffResult
	_ = json.Unmarshal(secondRec.Body.Bytes(), &secondResult)
	if secondResult.Updated || secondResult.PreviousSHA != currentSHA || secondResult.UpstreamSHA != currentSHA {
		t.Fatalf("expected updated=false when SHA unchanged, got: %+v", secondResult)
	}

	nextSHA := "def999888777666555444333222111000cccdde2"
	currentSHA = nextSHA
	thirdResult, err := gw.SniffUpstream(context.Background())
	if err != nil {
		t.Fatalf("third sniff failed: %v", err)
	}
	if !thirdResult.Updated || thirdResult.PreviousSHA != "abc111222333444555666777888999000aaabbb1" || thirdResult.UpstreamSHA != nextSHA {
		t.Fatalf("expected updated=true on new SHA, got: %+v", thirdResult)
	}
}

func TestProviderAPIKeysEncryptedAtRestAndNeverLeaked(t *testing.T) {
	gw, app := setupTestGateway(t)

	secretAPIKey := "sk-live-super-secret-never-leak-987654321"
	createRec := doJSONRequest(t, app, http.MethodPost, "/api/ai/providers", map[string]any{
		"name":         "groq-secure",
		"kind":         "openai_compat",
		"tier":         "free",
		"baseUrl":      "https://api.groq.com/openai/v1",
		"apiKey":       secretAPIKey,
		"defaultModel": "llama-3.3-70b-versatile",
	}, nil)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create provider failed: %d %s", createRec.Code, createRec.Body.String())
	}
	if strings.Contains(createRec.Body.String(), secretAPIKey) {
		t.Fatalf("API key leaked in POST /api/ai/providers response")
	}

	var created gateway.Provider
	_ = json.Unmarshal(createRec.Body.Bytes(), &created)
	if !created.HasAPIKey {
		t.Fatalf("expected hasApiKey=true")
	}

	listRec := doJSONRequest(t, app, http.MethodGet, "/api/ai/providers", nil, nil)
	if listRec.Code != http.StatusOK {
		t.Fatalf("GET /api/ai/providers failed: %d", listRec.Code)
	}
	if strings.Contains(listRec.Body.String(), secretAPIKey) || strings.Contains(listRec.Body.String(), "api_key_enc") {
		t.Fatalf("API key or encrypted blob leaked in GET /api/ai/providers: %s", listRec.Body.String())
	}

	var storedCipher string
	if err := gw.DB().QueryRow(`SELECT api_key_enc FROM ai_providers WHERE id = ?`, created.ID).Scan(&storedCipher); err != nil {
		t.Fatalf("query stored api_key_enc: %v", err)
	}
	if storedCipher == "" || storedCipher == secretAPIKey || strings.Contains(storedCipher, secretAPIKey) {
		t.Fatalf("API key is not encrypted at rest in SQLite: %q", storedCipher)
	}

	settingsRec := doJSONRequest(t, app, http.MethodPut, "/api/ai/settings", map[string]any{
		"rtkEnabled":    false,
		"cacheEnabled":  true,
		"fallbackOrder": "cheap,free,paid",
	}, nil)
	if settingsRec.Code != http.StatusOK || !strings.Contains(settingsRec.Body.String(), `"fallbackOrder":"cheap,free,paid"`) {
		t.Fatalf("PUT /api/ai/settings failed: %d %s", settingsRec.Code, settingsRec.Body.String())
	}

	delRec := doJSONRequest(t, app, http.MethodDelete, "/api/ai/providers/"+created.ID, nil, nil)
	if delRec.Code != http.StatusOK || !strings.Contains(delRec.Body.String(), `"deleted":true`) {
		t.Fatalf("DELETE /api/ai/providers/:id failed: %d %s", delRec.Code, delRec.Body.String())
	}
}

func TestCavemanCompressionAndHeaderMode(t *testing.T) {
	verbosePrompt := strings.Join([]string{
		"Halo, selamat pagi! Tolong bantu saya sebenarnya pada dasarnya ini adalah sebuah permintaan yang sangat panjang sekali dong deh nih sih.",
		"Could you please kindly help me actually basically literally refactor the very verbose configuration object that is really quite simple?",
		"{",
		"    \"service\": \"mw-rocksoul-gateway\",",
		"    \"mode\": \"caveman\",",
		"    \"retries\": 3,",
		"    \"endpoints\": [",
		"        \"/v1/chat/completions\",",
		"        \"/v1/usage\"",
		"    ]",
		"}",
		"Thank you very much! Terima kasih banyak ya!",
	}, "\n")

	rawTokens := len(verbosePrompt) / 4
	compressedMsgs, savedTokens := gateway.CompressCaveman([]gateway.ChatMessage{
		{Role: "user", Content: verbosePrompt},
	})
	if len(compressedMsgs) != 1 {
		t.Fatalf("expected 1 compressed message, got %d", len(compressedMsgs))
	}
	compressedText, ok := compressedMsgs[0].Content.(string)
	if !ok {
		t.Fatalf("expected string content")
	}
	if savedTokens <= 0 {
		t.Fatalf("expected positive token savings from CompressCaveman, got %d", savedTokens)
	}
	savingsPct := float64(savedTokens) / float64(rawTokens) * 100.0
	if savingsPct < 30.0 {
		t.Fatalf("expected >=30%% token savings in caveman mode, got %.1f%% (saved=%d raw=%d, out=%q)", savingsPct, savedTokens, rawTokens, compressedText)
	}
	if !strings.Contains(compressedText, `{"service":"mw-rocksoul-gateway"`) {
		t.Fatalf("expected compacted JSON in caveman output, got: %q", compressedText)
	}

	gw, app := setupTestGateway(t)
	var receivedPrompt string
	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body gateway.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err == nil && len(body.Messages) > 0 {
			if text, isStr := body.Messages[0].Content.(string); isStr {
				receivedPrompt = text
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-caveman","object":"chat.completion","model":"mock-caveman","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}`))
	}))
	defer mockUpstream.Close()

	_, err := gw.AddProvider(context.Background(), gateway.ProviderCreateInput{
		Name:         "caveman-mock",
		Kind:         gateway.KindOpenAICompat,
		Tier:         gateway.TierFree,
		BaseURL:      mockUpstream.URL,
		DefaultModel: "mock-caveman",
		Models:       []string{"mock-caveman"},
	})
	if err != nil {
		t.Fatalf("add provider: %v", err)
	}

	reqBody := gateway.ChatCompletionRequest{
		Model:    "mock-caveman",
		Messages: []gateway.ChatMessage{{Role: "user", Content: verbosePrompt}},
	}
	recHeader := doJSONRequest(t, app, http.MethodPost, "/v1/chat/completions", reqBody, map[string]string{
		"X-MW-RTK-Mode": "caveman",
	})
	if recHeader.Code != http.StatusOK {
		t.Fatalf("expected 200 for caveman header request, got %d: %s", recHeader.Code, recHeader.Body.String())
	}
	savedHeader := recHeader.Header().Get("X-Rocksoul-RTK-Saved")
	if savedHeader == "" || savedHeader == "0" {
		t.Fatalf("expected non-zero X-Rocksoul-RTK-Saved header, got %q", savedHeader)
	}
	if len(receivedPrompt) >= len(verbosePrompt) {
		t.Fatalf("expected upstream prompt to be compressed, raw=%d sent=%d", len(verbosePrompt), len(receivedPrompt))
	}

	recQuery := doJSONRequest(t, app, http.MethodPost, "/v1/chat/completions?rtk=caveman", reqBody, nil)
	if recQuery.Code != http.StatusOK {
		t.Fatalf("expected 200 for ?rtk=caveman request, got %d: %s", recQuery.Code, recQuery.Body.String())
	}
	if savedQ := recQuery.Header().Get("X-Rocksoul-RTK-Saved"); savedQ == "" || savedQ == "0" {
		t.Fatalf("expected non-zero X-Rocksoul-RTK-Saved header for query mode, got %q", savedQ)
	}
}
