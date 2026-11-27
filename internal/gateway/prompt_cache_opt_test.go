package gateway

import (
	"strings"
	"testing"
)

func TestPromptCacheOptimizer(t *testing.T) {
	opt := NewPromptCacheOptimizer(1024)

	// 1. Stable prefix hash computation
	hash1 := opt.ComputeStablePrefixHash("system instruction", nil)
	hash2 := opt.ComputeStablePrefixHash("system instruction", nil)
	if hash1 != hash2 || len(hash1) != 64 {
		t.Fatalf("deterministic hash failed: %s vs %s", hash1, hash2)
	}

	// 2. Anthropic breakpoints
	systemPrompt := strings.Repeat("You are a deterministic coding agent. Follow all rules. ", 10)
	anthReq := &AnthropicMessageRequest{
		System: systemPrompt,
		Messages: []AnthropicMessage{
			{Role: "user", Content: "turn 1"},
			{Role: "assistant", Content: "reply 1"},
			{Role: "user", Content: "turn 2"},
		},
	}
	bps, ok := opt.OptimizeAnthropicRequest(anthReq)
	if !ok || len(bps) == 0 {
		t.Fatalf("expected anthropic breakpoints, got ok=%v bps=%+v", ok, bps)
	}
	if bps[0].Role != "system" || bps[0].Provider != KindAnthropic {
		t.Fatalf("unexpected system breakpoint: %+v", bps[0])
	}

	// 3. OpenAI prefix optimization
	openAIReq := &ChatCompletionRequest{
		Model: "gpt-4o",
		Messages: []ChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: "hello"},
		},
	}
	openAIHash, openAIOk := opt.OptimizeOpenAIRequest(openAIReq)
	if !openAIOk || len(openAIHash) != 64 {
		t.Fatalf("openai cache optimization failed: ok=%v hash=%s", openAIOk, openAIHash)
	}

	// 4. Gemini cached contents
	geminiResource, geminiOk := opt.OptimizeGeminiRequest(systemPrompt+strings.Repeat(" extra tokens", 200), "gemini-2.0-flash")
	if !geminiOk || !strings.HasPrefix(geminiResource, "cachedContents/gemini-2.0-flash-") {
		t.Fatalf("gemini cache optimization failed: ok=%v res=%s", geminiOk, geminiResource)
	}
}
