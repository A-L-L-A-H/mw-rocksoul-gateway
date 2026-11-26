package gateway

import (
	"context"
	"errors"
	"strings"
)

const (
	GroqProviderName       = "groq-lpu-free"
	OpenRouterProviderName = "openrouter-free"
	NvidiaProviderName     = "nvidia-nim-free"

	DefaultGroqBaseURL       = "https://api.groq.com/openai/v1"
	DefaultOpenRouterBaseURL = "https://openrouter.ai/api/v1"
	DefaultNvidiaBaseURL     = "https://integrate.api.nvidia.com/v1"

	DefaultGroqModel       = "llama-3.3-70b-versatile"
	DefaultOpenRouterModel = "openrouter/free"
	DefaultNvidiaModel     = "meta/llama-3.3-70b-instruct"
)

// FreeProvidersConfig configures optional zero-credit-card fallback providers.
type FreeProvidersConfig struct {
	GroqAPIKey        string
	GroqBaseURL       string
	OpenRouterAPIKey  string
	OpenRouterBaseURL string
	NvidiaAPIKey      string
	NvidiaBaseURL     string
}

// FreeProvidersConfigFromEnv reads Groq, OpenRouter, and NVIDIA NIM keys/URLs.
func FreeProvidersConfigFromEnv(lookup func(string) string) FreeProvidersConfig {
	pick := func(keys ...string) string {
		for _, k := range keys {
			if val := strings.TrimSpace(lookup(k)); val != "" {
				return val
			}
		}
		return ""
	}
	return FreeProvidersConfig{
		GroqAPIKey:        pick("GROQ_API_KEY"),
		GroqBaseURL:       pick("GROQ_BASE_URL"),
		OpenRouterAPIKey:  pick("OPENROUTER_API_KEY"),
		OpenRouterBaseURL: pick("OPENROUTER_BASE_URL"),
		NvidiaAPIKey:      pick("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
		NvidiaBaseURL:     pick("NVIDIA_BASE_URL", "NVIDIA_NIM_BASE_URL"),
	}
}

func normalizeOpenAIBase(raw, fallback string) string {
	base := strings.TrimSpace(raw)
	if base == "" {
		base = fallback
	}
	base = strings.TrimRight(base, "/")
	if strings.HasSuffix(base, "/chat/completions") {
		base = strings.TrimSuffix(base, "/chat/completions")
	}
	return base
}

type freeProviderSpec struct {
	Name         string
	BaseURL      string
	APIKey       string
	DefaultModel string
	Models       []string
	Priority     int
}

// ConfigureFreeProviders idempotently registers configured zero-card cloud fallback providers
// ordered after Gemini (50): Groq (60) -> OpenRouter :free (70) -> NVIDIA NIM (80).
// Empty API keys are clean no-ops. local_only requests strictly never route to these providers.
func (g *Gateway) ConfigureFreeProviders(ctx context.Context, cfg FreeProvidersConfig) ([]string, error) {
	specs := []freeProviderSpec{
		{
			Name:         GroqProviderName,
			BaseURL:      normalizeOpenAIBase(cfg.GroqBaseURL, DefaultGroqBaseURL),
			APIKey:       strings.TrimSpace(cfg.GroqAPIKey),
			DefaultModel: DefaultGroqModel,
			Models:       []string{DefaultGroqModel, "qwen/qwen3-32b", "openai/gpt-oss-120b"},
			Priority:     60,
		},
		{
			Name:         OpenRouterProviderName,
			BaseURL:      normalizeOpenAIBase(cfg.OpenRouterBaseURL, DefaultOpenRouterBaseURL),
			APIKey:       strings.TrimSpace(cfg.OpenRouterAPIKey),
			DefaultModel: DefaultOpenRouterModel,
			Models:       []string{DefaultOpenRouterModel, "meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.0-flash-exp:free"},
			Priority:     70,
		},
		{
			Name:         NvidiaProviderName,
			BaseURL:      normalizeOpenAIBase(cfg.NvidiaBaseURL, DefaultNvidiaBaseURL),
			APIKey:       strings.TrimSpace(cfg.NvidiaAPIKey),
			DefaultModel: DefaultNvidiaModel,
			Models:       []string{DefaultNvidiaModel, "nvidia/llama-3.1-nemotron-70b-instruct"},
			Priority:     80,
		},
	}

	existing, err := g.ListProviders(ctx)
	if err != nil {
		return nil, err
	}
	existingByName := make(map[string]string, len(existing))
	for _, p := range existing {
		existingByName[p.Name] = p.ID
	}

	var registered []string
	for _, spec := range specs {
		if spec.APIKey == "" {
			continue
		}
		if id, ok := existingByName[spec.Name]; ok {
			if err := g.DeleteProvider(ctx, id); err != nil {
				return nil, err
			}
		}
		_, err := g.AddProvider(ctx, ProviderCreateInput{
			Name:         spec.Name,
			Kind:         KindOpenAICompat,
			Tier:         TierFree,
			BaseURL:      spec.BaseURL,
			APIKey:       spec.APIKey,
			DefaultModel: spec.DefaultModel,
			Models:       spec.Models,
			Priority:     spec.Priority,
		})
		if err != nil {
			return nil, errors.Join(errors.New("register free fallback provider "+spec.Name), err)
		}
		registered = append(registered, spec.Name)
	}
	return registered, nil
}
