package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// PromptCacheBreakpoint represents an explicit prompt cache breakpoint marker.
type PromptCacheBreakpoint struct {
	Index    int    `json:"index"`
	Role     string `json:"role"`
	PrefixID string `json:"prefixId"`
	Provider string `json:"provider"`
}

// PromptCacheOptimizer analyzes messages and tool definitions to compute stable prefix
// cache breakpoints tailored for Anthropic (cache_control), OpenAI (prefix caching),
// and Gemini (cachedContents resource references).
type PromptCacheOptimizer struct {
	MinPrefixTokens int
}

// NewPromptCacheOptimizer constructs an optimizer with a token threshold floor.
func NewPromptCacheOptimizer(minPrefixTokens int) *PromptCacheOptimizer {
	if minPrefixTokens <= 0 {
		minPrefixTokens = 1024 // standard stable prefix floor
	}
	return &PromptCacheOptimizer{MinPrefixTokens: minPrefixTokens}
}

// ComputeStablePrefixHash computes a deterministic hex SHA256 of stable system/tool prefixes.
func (o *PromptCacheOptimizer) ComputeStablePrefixHash(systemPrompt string, tools json.RawMessage) string {
	h := sha256.New()
	h.Write([]byte(strings.TrimSpace(systemPrompt)))
	if len(tools) > 0 {
		h.Write(tools)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// extractStringContent extracts a string representation of any message content.
func extractStringContent(content any) string {
	if content == nil {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		return string(b)
	}
}

// OptimizeAnthropicRequest injects cache_control ephemerals at stable system and tool checkpoints.
func (o *PromptCacheOptimizer) OptimizeAnthropicRequest(req *AnthropicMessageRequest) ([]PromptCacheBreakpoint, bool) {
	if req == nil || len(req.Messages) == 0 {
		return nil, false
	}

	var breakpoints []PromptCacheBreakpoint
	sysText := extractStringContent(req.System)
	if len(sysText) >= 200 { // heuristic ~50+ tokens
		breakpoints = append(breakpoints, PromptCacheBreakpoint{
			Index:    -1,
			Role:     "system",
			PrefixID: o.ComputeStablePrefixHash(sysText, nil)[:16],
			Provider: KindAnthropic,
		})
	}

	// Mark the last user message before the current turn if conversation has multi-turn history.
	lastUserIdx := -1
	for i := len(req.Messages) - 2; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			lastUserIdx = i
			break
		}
	}

	if lastUserIdx >= 0 {
		breakpoints = append(breakpoints, PromptCacheBreakpoint{
			Index:    lastUserIdx,
			Role:     "user",
			PrefixID: fmt.Sprintf("anth-turn-%d", lastUserIdx),
			Provider: KindAnthropic,
		})
	}

	return breakpoints, len(breakpoints) > 0
}

// OptimizeOpenAIRequest identifies stable prefix blocks for OpenAI prompt caching.
func (o *PromptCacheOptimizer) OptimizeOpenAIRequest(req *ChatCompletionRequest) (string, bool) {
	if req == nil || len(req.Messages) == 0 {
		return "", false
	}

	var systemContent strings.Builder
	for _, m := range req.Messages {
		if m.Role == "system" {
			systemContent.WriteString(extractStringContent(m.Content))
			systemContent.WriteString("\n")
		}
	}

	prefixHash := o.ComputeStablePrefixHash(systemContent.String(), req.Tools)
	return prefixHash, true
}

// OptimizeGeminiRequest formats a Gemini cachedContent resource identifier if eligible.
func (o *PromptCacheOptimizer) OptimizeGeminiRequest(systemPrompt string, model string) (string, bool) {
	clean := strings.TrimSpace(systemPrompt)
	if len(clean) < 1000 {
		return "", false
	}
	hash := sha256.Sum256([]byte(clean))
	prefixID := hex.EncodeToString(hash[:8])
	return fmt.Sprintf("cachedContents/%s-%s", model, prefixID), true
}
