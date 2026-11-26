package gateway

import (
	"context"
	"errors"
	"strings"
)

const (
	// GeminiProviderName is the stable registry name of the Google AI Studio provider.
	GeminiProviderName = "google-ai-studio"
	// DefaultGeminiBaseURL is the Google AI Studio OpenAI-compatible endpoint root.
	DefaultGeminiBaseURL = "https://generativelanguage.googleapis.com/v1beta/openai"
	// DefaultGeminiModel is the default Free Tier model.
	DefaultGeminiModel = "gemini-2.5-flash"
)

// GeminiFreeTierModels lists Google AI Studio Free Tier models served by the gateway.
var GeminiFreeTierModels = []string{"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"}

// GeminiConfig configures the Google AI Studio (Gemini Free Tier) provider.
type GeminiConfig struct {
	APIKey  string
	BaseURL string
}

// GeminiConfigFromEnv reads GEMINI_API_KEY (alias GOOGLE_AI_API_KEY) and the
// optional GEMINI_BASE_URL override through the supplied lookup function.
func GeminiConfigFromEnv(lookup func(string) string) GeminiConfig {
	pick := func(keys ...string) string {
		for _, key := range keys {
			if value := strings.TrimSpace(lookup(key)); value != "" {
				return value
			}
		}
		return ""
	}
	return GeminiConfig{
		APIKey:  pick("GEMINI_API_KEY", "GOOGLE_AI_API_KEY"),
		BaseURL: pick("GEMINI_BASE_URL"),
	}
}

// ConfigureGemini registers (or refreshes) the Google AI Studio provider.
// An empty API key is a no-op and returns (false, nil). The provider is an
// external cloud egress target: local_only requests never select it, and
// cloud_eligible requests pass through the PII/secret sniffer and RTK
// compression before forwarding.
func (g *Gateway) ConfigureGemini(ctx context.Context, cfg GeminiConfig) (bool, error) {
	apiKey := strings.TrimSpace(cfg.APIKey)
	if apiKey == "" {
		return false, nil
	}
	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		baseURL = DefaultGeminiBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(baseURL, "/chat/completions") {
		baseURL = strings.TrimSuffix(baseURL, "/chat/completions")
	}

	existing, err := g.ListProviders(ctx)
	if err != nil {
		return false, err
	}
	for _, provider := range existing {
		if provider.Name == GeminiProviderName {
			if err := g.DeleteProvider(ctx, provider.ID); err != nil {
				return false, err
			}
		}
	}

	_, err = g.AddProvider(ctx, ProviderCreateInput{
		Name:         GeminiProviderName,
		Kind:         KindGemini,
		Tier:         TierFree,
		BaseURL:      baseURL,
		APIKey:       apiKey,
		DefaultModel: DefaultGeminiModel,
		Models:       GeminiFreeTierModels,
		Priority:     50,
	})
	if err != nil {
		return false, errors.Join(errors.New("register google ai studio provider"), err)
	}
	return true, nil
}
