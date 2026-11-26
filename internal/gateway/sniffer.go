package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// --- PII & Secret Egress Sniffer ---

// PIISniffReport summarizes all secrets and PII redacted prior to cloud egress.
// Raw secret material is never recorded in the report.
type PIISniffReport struct {
	RedactedCount int            `json:"redactedCount"`
	Categories    map[string]int `json:"categories,omitempty"`
}

var (
	pemPrivateKeyPattern = regexp.MustCompile(`(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----`)
	authHeaderPattern    = regexp.MustCompile(`(?i)\b(Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]{8,}`)
	apiKeyTokenPattern   = regexp.MustCompile(`\b(?:sk-proj-|sk-ant-|sk-live-|rk_live_|pk_live_|sk-|gsk_|xai-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|xoxb-|xoxp-|hf_)[A-Za-z0-9_-]{8,}\b`)
	awsAccessKeyPattern  = regexp.MustCompile(`\b(?:AKIA|ASIA)[0-9A-Z]{16}\b`)
	googleAPIKeyPattern  = regexp.MustCompile(`\bAIza[0-9A-Za-z_-]{35}\b`)
	jwtTokenPattern      = regexp.MustCompile(`\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b`)
	urlCredsPattern      = regexp.MustCompile(`(?i)((?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|rediss?|amqps?)://)([^:@/\s]+):([^@/\s]+)@`)
	kvSecretPattern      = regexp.MustCompile(`(?i)\b(password|passwd|secret|client_secret|private_key|access_token|refresh_token|token|credential|api_key|apikey|x-api-key)\s*([:=])\s*(?:"[^"]*"|'[^']*'|[^\s,;'"}\]]+)`)
	emailPIIPattern      = regexp.MustCompile(`\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`)
	ssnPIIPattern        = regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)
	cardCandidatePattern = regexp.MustCompile(`\b(?:\d[ -]?){13,19}\b`)
	phonePIIPattern      = regexp.MustCompile(`(?:\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,5}\b|\b08\d{2}[\s.-]?\d{3,4}[\s.-]?\d{3,5}\b)`)
	nikPIIPattern        = regexp.MustCompile(`(?i)\b(nik)\s*([:=])\s*\d{16}\b`)
)

func (r *PIISniffReport) add(category string, count int) {
	if count <= 0 {
		return
	}
	if r.Categories == nil {
		r.Categories = make(map[string]int)
	}
	r.Categories[category] += count
	r.RedactedCount += count
}

func (r *PIISniffReport) merge(other PIISniffReport) {
	for cat, count := range other.Categories {
		r.add(cat, count)
	}
}

// SniffAndRedactText detects and redacts credentials, API keys, private keys,
// JWTs, URL passwords, and PII (emails, SSNs, Luhn-valid cards, phones, NIK)
// before external cloud egress.
func SniffAndRedactText(input string, knownSecrets ...string) (string, PIISniffReport) {
	report := PIISniffReport{}
	if input == "" {
		return "", report
	}

	output := input

	for _, secret := range knownSecrets {
		trimmed := strings.TrimSpace(secret)
		if len(trimmed) < 6 {
			continue
		}
		if hits := strings.Count(output, trimmed); hits > 0 {
			output = strings.ReplaceAll(output, trimmed, "[REDACTED_SECRET]")
			report.add("known_secret", hits)
		}
	}

	replaceRule := func(re *regexp.Regexp, category, replacement string) {
		matches := re.FindAllStringIndex(output, -1)
		if len(matches) == 0 {
			return
		}
		output = re.ReplaceAllString(output, replacement)
		report.add(category, len(matches))
	}

	replaceRule(pemPrivateKeyPattern, "private_key", "[REDACTED_PRIVATE_KEY]")
	replaceRule(jwtTokenPattern, "jwt", "[REDACTED_JWT]")
	replaceRule(authHeaderPattern, "auth_header", "$1 [REDACTED_TOKEN]")
	replaceRule(apiKeyTokenPattern, "api_key", "[REDACTED_API_KEY]")
	replaceRule(awsAccessKeyPattern, "aws_access_key", "[REDACTED_AWS_KEY]")
	replaceRule(googleAPIKeyPattern, "google_api_key", "[REDACTED_API_KEY]")
	replaceRule(urlCredsPattern, "url_credentials", "${1}[REDACTED]:[REDACTED]@")

	output = kvSecretPattern.ReplaceAllStringFunc(output, func(matched string) string {
		sepIdx := strings.IndexAny(matched, ":=")
		if sepIdx < 0 {
			return matched
		}
		keyPart := strings.TrimSpace(matched[:sepIdx])
		sepChar := matched[sepIdx : sepIdx+1]
		valPart := strings.TrimSpace(matched[sepIdx+1:])
		if strings.HasPrefix(valPart, "[REDACTED") {
			return matched
		}
		report.add("kv_secret", 1)
		return keyPart + sepChar + "[REDACTED_SECRET]"
	})

	output = nikPIIPattern.ReplaceAllStringFunc(output, func(matched string) string {
		sepIdx := strings.IndexAny(matched, ":=")
		if sepIdx < 0 {
			return matched
		}
		keyPart := strings.TrimSpace(matched[:sepIdx])
		sepChar := matched[sepIdx : sepIdx+1]
		report.add("pii_nik", 1)
		return keyPart + sepChar + "[REDACTED_NIK]"
	})

	replaceRule(emailPIIPattern, "pii_email", "[REDACTED_EMAIL]")
	replaceRule(ssnPIIPattern, "pii_ssn", "[REDACTED_SSN]")

	output = cardCandidatePattern.ReplaceAllStringFunc(output, func(candidate string) string {
		digits := extractDigits(candidate)
		if len(digits) < 13 || len(digits) > 19 {
			return candidate
		}
		if !isValidLuhn(digits) {
			return candidate
		}
		report.add("pii_card", 1)
		return "[REDACTED_CARD]"
	})

	replaceRule(phonePIIPattern, "pii_phone", "[REDACTED_PHONE]")

	return output, report
}

// SniffAndRedactMessages scans and scrubs secrets and PII across all chat
// messages and structured content blocks without mutating the caller's slice.
func SniffAndRedactMessages(messages []ChatMessage, knownSecrets ...string) ([]ChatMessage, PIISniffReport) {
	totalReport := PIISniffReport{}
	if len(messages) == 0 {
		return nil, totalReport
	}

	sanitized := make([]ChatMessage, len(messages))
	for idx, msg := range messages {
		copied := msg
		switch contentVal := msg.Content.(type) {
		case string:
			cleaned, rep := SniffAndRedactText(contentVal, knownSecrets...)
			copied.Content = cleaned
			totalReport.merge(rep)

		case []any:
			nextBlocks, rep := sniffContentBlocks(contentVal, knownSecrets...)
			copied.Content = nextBlocks
			totalReport.merge(rep)

		case []map[string]any:
			genericSlice := make([]any, len(contentVal))
			for blockIdx, blockMap := range contentVal {
				genericSlice[blockIdx] = blockMap
			}
			nextBlocks, rep := sniffContentBlocks(genericSlice, knownSecrets...)
			copied.Content = nextBlocks
			totalReport.merge(rep)
		}
		sanitized[idx] = copied
	}
	return sanitized, totalReport
}

func sniffContentBlocks(blocks []any, knownSecrets ...string) ([]any, PIISniffReport) {
	outBlocks := make([]any, len(blocks))
	totalReport := PIISniffReport{}

	for idx, rawBlock := range blocks {
		blockMap, ok := rawBlock.(map[string]any)
		if !ok {
			outBlocks[idx] = rawBlock
			continue
		}
		cloned := make(map[string]any, len(blockMap))
		for k, v := range blockMap {
			cloned[k] = v
		}

		if textStr, isStr := cloned["text"].(string); isStr {
			cleaned, rep := SniffAndRedactText(textStr, knownSecrets...)
			cloned["text"] = cleaned
			totalReport.merge(rep)
		}
		switch inner := cloned["content"].(type) {
		case string:
			cleaned, rep := SniffAndRedactText(inner, knownSecrets...)
			cloned["content"] = cleaned
			totalReport.merge(rep)
		case []any:
			subBlocks, rep := sniffContentBlocks(inner, knownSecrets...)
			cloned["content"] = subBlocks
			totalReport.merge(rep)
		}

		outBlocks[idx] = cloned
	}
	return outBlocks, totalReport
}

func extractDigits(raw string) string {
	var b strings.Builder
	for i := 0; i < len(raw); i++ {
		if raw[i] >= '0' && raw[i] <= '9' {
			b.WriteByte(raw[i])
		}
	}
	return b.String()
}

func isValidLuhn(digits string) bool {
	sum := 0
	alt := false
	for i := len(digits) - 1; i >= 0; i-- {
		d := int(digits[i] - '0')
		if alt {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		sum += d
		alt = !alt
	}
	return sum%10 == 0
}

// --- Upstream FreeLLMAPI Catalog Sniffer ---

type upstreamProviderMeta struct {
	Slug    string
	Name    string
	BaseURL string
}

var (
	openAICompatBlockRegex = regexp.MustCompile(`(?s)new\s+OpenAICompatProvider\s*\(\s*\{(.*?)\}\s*\)`)
	platformFieldRegex     = regexp.MustCompile(`platform\s*:\s*['"]([a-zA-Z0-9_-]+)['"]`)
	nameFieldRegex         = regexp.MustCompile(`name\s*:\s*['"]([^'"]+)['"]`)
	baseURLFieldRegex      = regexp.MustCompile(`baseUrl\s*:\s*['"](https?://[^'"]+)['"]`)
	modelPricingRowRegex   = regexp.MustCompile(`\[\s*['"]([a-zA-Z0-9_-]+)['"]\s*,\s*['"]([^'"]+)['"]\s*,`)
)

func knownProviderDirectory() map[string]upstreamProviderMeta {
	return map[string]upstreamProviderMeta{
		"groq":          {Slug: "groq", Name: "Groq", BaseURL: "https://api.groq.com/openai/v1"},
		"cerebras":      {Slug: "cerebras", Name: "Cerebras", BaseURL: "https://api.cerebras.ai/v1"},
		"openrouter":    {Slug: "openrouter", Name: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1"},
		"google":        {Slug: "gemini", Name: "Google Gemini", BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai"},
		"gemini":        {Slug: "gemini", Name: "Google Gemini", BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai"},
		"mistral":       {Slug: "mistral", Name: "Mistral", BaseURL: "https://api.mistral.ai/v1"},
		"nvidia":        {Slug: "nvidia", Name: "NVIDIA NIM", BaseURL: "https://integrate.api.nvidia.com/v1"},
		"sambanova":     {Slug: "sambanova", Name: "SambaNova", BaseURL: "https://api.sambanova.ai/v1"},
		"github-models": {Slug: "github-models", Name: "GitHub Models", BaseURL: "https://models.inference.ai.azure.com"},
		"deepseek":      {Slug: "deepseek", Name: "DeepSeek", BaseURL: "https://api.deepseek.com/v1"},
		"huggingface":   {Slug: "huggingface", Name: "HuggingFace Router", BaseURL: "https://router.huggingface.co/v1"},
		"cloudflare":    {Slug: "cloudflare", Name: "Cloudflare Workers AI", BaseURL: "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"},
		"cohere":        {Slug: "cohere", Name: "Cohere", BaseURL: "https://api.cohere.ai/compatibility/v1"},
		"zhipu":         {Slug: "zhipu", Name: "Zhipu AI", BaseURL: "https://open.bigmodel.cn/api/paas/v4"},
		"ollama":        {Slug: "ollama", Name: "Ollama Cloud", BaseURL: "https://ollama.com/v1"},
		"kilo":          {Slug: "kilo", Name: "Kilo Gateway", BaseURL: "https://api.kilo.ai/api/gateway/v1"},
		"pollinations":  {Slug: "pollinations", Name: "Pollinations", BaseURL: "https://gen.pollinations.ai/v1"},
		"reka":          {Slug: "reka", Name: "Reka", BaseURL: "https://api.reka.ai/v1"},
		"anyapi":        {Slug: "anyapi", Name: "AnyAPI", BaseURL: "https://api.anyapi.ai/v1"},
		"powertokens":   {Slug: "powertokens", Name: "PowerTokens", BaseURL: "https://api.powertokens.ai/v1"},
		"cheapllm":      {Slug: "cheapllm", Name: "cheapllm.shop", BaseURL: "https://cheapllm.shop/v1"},
		"tokenreply":    {Slug: "tokenreply", Name: "TokenReply", BaseURL: "https://api.tokenreply.com/v1"},
		"radeon":        {Slug: "radeon", Name: "AMD Radeon Cloud", BaseURL: "https://developer.amd.com.cn/radeon/api/v1"},
		"longcat":       {Slug: "longcat", Name: "LongCat", BaseURL: "https://api.longcat.chat/openai/v1"},
	}
}

func curatedFreeCatalog() []CatalogEntry {
	return []CatalogEntry{
		// Groq
		{ProviderSlug: "groq", ProviderName: "Groq", BaseURL: "https://api.groq.com/openai/v1", ModelID: "llama-3.3-70b-versatile", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "groq", ProviderName: "Groq", BaseURL: "https://api.groq.com/openai/v1", ModelID: "llama-3.1-8b-instant", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "groq", ProviderName: "Groq", BaseURL: "https://api.groq.com/openai/v1", ModelID: "meta-llama/llama-4-scout-17b-16e-instruct", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "groq", ProviderName: "Groq", BaseURL: "https://api.groq.com/openai/v1", ModelID: "openai/gpt-oss-120b", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "groq", ProviderName: "Groq", BaseURL: "https://api.groq.com/openai/v1", ModelID: "qwen/qwen3-32b", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		// Cerebras
		{ProviderSlug: "cerebras", ProviderName: "Cerebras", BaseURL: "https://api.cerebras.ai/v1", ModelID: "llama3.1-8b", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "cerebras", ProviderName: "Cerebras", BaseURL: "https://api.cerebras.ai/v1", ModelID: "gpt-oss-120b", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "cerebras", ProviderName: "Cerebras", BaseURL: "https://api.cerebras.ai/v1", ModelID: "qwen-3-235b-a22b-instruct-2507", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "cerebras", ProviderName: "Cerebras", BaseURL: "https://api.cerebras.ai/v1", ModelID: "zai-glm-4.7", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		// OpenRouter :free
		{ProviderSlug: "openrouter", ProviderName: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1", ModelID: "meta-llama/llama-3.3-70b-instruct:free", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "openrouter", ProviderName: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1", ModelID: "qwen/qwen3-coder:free", ContextWindow: 262144, FreeTier: true, Source: "seed"},
		{ProviderSlug: "openrouter", ProviderName: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1", ModelID: "google/gemma-4-31b-it:free", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "openrouter", ProviderName: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1", ModelID: "openai/gpt-oss-120b:free", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "openrouter", ProviderName: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1", ModelID: "moonshotai/kimi-k2.6:free", ContextWindow: 262144, FreeTier: true, Source: "seed"},
		{ProviderSlug: "openrouter", ProviderName: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1", ModelID: "nvidia/nemotron-3-super-120b-a12b:free", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		// Google Gemini OpenAI-compat
		{ProviderSlug: "gemini", ProviderName: "Google Gemini", BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai", ModelID: "gemini-2.5-flash", ContextWindow: 1048576, FreeTier: true, Source: "seed"},
		{ProviderSlug: "gemini", ProviderName: "Google Gemini", BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai", ModelID: "gemini-2.5-flash-lite", ContextWindow: 1048576, FreeTier: true, Source: "seed"},
		{ProviderSlug: "gemini", ProviderName: "Google Gemini", BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai", ModelID: "gemini-2.5-pro", ContextWindow: 1048576, FreeTier: true, Source: "seed"},
		{ProviderSlug: "gemini", ProviderName: "Google Gemini", BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai", ModelID: "gemma-4-31b-it", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		// Mistral
		{ProviderSlug: "mistral", ProviderName: "Mistral", BaseURL: "https://api.mistral.ai/v1", ModelID: "mistral-small-latest", ContextWindow: 128000, FreeTier: true, Source: "seed"},
		{ProviderSlug: "mistral", ProviderName: "Mistral", BaseURL: "https://api.mistral.ai/v1", ModelID: "codestral-latest", ContextWindow: 256000, FreeTier: true, Source: "seed"},
		{ProviderSlug: "mistral", ProviderName: "Mistral", BaseURL: "https://api.mistral.ai/v1", ModelID: "open-mistral-nemo", ContextWindow: 128000, FreeTier: true, Source: "seed"},
		// NVIDIA NIM
		{ProviderSlug: "nvidia", ProviderName: "NVIDIA NIM", BaseURL: "https://integrate.api.nvidia.com/v1", ModelID: "meta/llama-3.3-70b-instruct", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "nvidia", ProviderName: "NVIDIA NIM", BaseURL: "https://integrate.api.nvidia.com/v1", ModelID: "deepseek-ai/deepseek-r1", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "nvidia", ProviderName: "NVIDIA NIM", BaseURL: "https://integrate.api.nvidia.com/v1", ModelID: "qwen/qwen2.5-coder-32b-instruct", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		// SambaNova
		{ProviderSlug: "sambanova", ProviderName: "SambaNova", BaseURL: "https://api.sambanova.ai/v1", ModelID: "Meta-Llama-3.3-70B-Instruct", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "sambanova", ProviderName: "SambaNova", BaseURL: "https://api.sambanova.ai/v1", ModelID: "DeepSeek-R1-Distill-Llama-70B", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "sambanova", ProviderName: "SambaNova", BaseURL: "https://api.sambanova.ai/v1", ModelID: "Qwen2.5-Coder-32B-Instruct", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		// GitHub Models
		{ProviderSlug: "github-models", ProviderName: "GitHub Models", BaseURL: "https://models.inference.ai.azure.com", ModelID: "gpt-4o-mini", ContextWindow: 128000, FreeTier: true, Source: "seed"},
		{ProviderSlug: "github-models", ProviderName: "GitHub Models", BaseURL: "https://models.inference.ai.azure.com", ModelID: "Llama-3.3-70B-Instruct", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "github-models", ProviderName: "GitHub Models", BaseURL: "https://models.inference.ai.azure.com", ModelID: "DeepSeek-R1", ContextWindow: 128000, FreeTier: true, Source: "seed"},
		// DeepSeek
		{ProviderSlug: "deepseek", ProviderName: "DeepSeek", BaseURL: "https://api.deepseek.com/v1", ModelID: "deepseek-chat", ContextWindow: 128000, FreeTier: false, Source: "seed"},
		{ProviderSlug: "deepseek", ProviderName: "DeepSeek", BaseURL: "https://api.deepseek.com/v1", ModelID: "deepseek-reasoner", ContextWindow: 128000, FreeTier: false, Source: "seed"},
		// HuggingFace Router
		{ProviderSlug: "huggingface", ProviderName: "HuggingFace Router", BaseURL: "https://router.huggingface.co/v1", ModelID: "Qwen/Qwen3-Coder-Next", ContextWindow: 262144, FreeTier: true, Source: "seed"},
		{ProviderSlug: "huggingface", ProviderName: "HuggingFace Router", BaseURL: "https://router.huggingface.co/v1", ModelID: "deepseek-ai/DeepSeek-V4-Flash", ContextWindow: 131072, FreeTier: true, Source: "seed"},
		{ProviderSlug: "huggingface", ProviderName: "HuggingFace Router", BaseURL: "https://router.huggingface.co/v1", ModelID: "moonshotai/Kimi-K2.6", ContextWindow: 262144, FreeTier: true, Source: "seed"},
	}
}

func (g *Gateway) seedInitialCatalog(ctx context.Context) error {
	now := time.Now().UTC().Format(time.RFC3339)
	tx, err := g.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	for _, entry := range curatedFreeCatalog() {
		entryID := makeGatewayID("cat", entry.ProviderSlug, entry.ModelID)
		_, execErr := tx.ExecContext(ctx, `
INSERT OR IGNORE INTO ai_catalog (id, provider_slug, provider_name, base_url, model_id, context_window, free_tier, source, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			entryID, entry.ProviderSlug, entry.ProviderName, entry.BaseURL, entry.ModelID,
			entry.ContextWindow, boolToInt(entry.FreeTier), entry.Source, now,
		)
		if execErr != nil {
			return execErr
		}
	}
	return tx.Commit()
}

// SniffUpstream checks the configured upstream FreeLLMAPI repository commit SHA,
// fetches live provider definitions and model catalog entries, merges with the
// curated seed catalog, and upserts all rows into ai_catalog.
// Blocked when the gateway operates in strict local_only default air-gap mode.
func (g *Gateway) SniffUpstream(ctx context.Context) (SniffResult, error) {
	if g.defaultClassification == ClassificationLocalOnly {
		return SniffResult{}, errors.New("upstream catalog sniffing is disabled in local_only air-gap mode")
	}

	settings := g.GetSettings(ctx)
	upstreamRepo := strings.TrimSpace(settings.UpstreamRepo)
	if upstreamRepo == "" {
		upstreamRepo = DefaultUpstreamRepo
	}
	previousSHA := settings.UpstreamSHA

	latestSHA, shaErr := g.fetchUpstreamCommitSHA(ctx, upstreamRepo)

	providerDir := knownProviderDirectory()
	discoveredProviders := make(map[string]upstreamProviderMeta)
	upstreamEntries := make([]CatalogEntry, 0)

	providersBody, providersErr := g.fetchUpstreamRaw(ctx, upstreamRepo, "server/src/providers/index.ts")
	if providersErr == nil && len(providersBody) > 0 {
		for _, parsed := range parseProvidersTS(string(providersBody)) {
			providerDir[parsed.Slug] = parsed
			discoveredProviders[parsed.Slug] = parsed
		}
	}

	pricingBody, pricingErr := g.fetchUpstreamRaw(ctx, upstreamRepo, "server/src/db/model-pricing.ts")
	if pricingErr == nil && len(pricingBody) > 0 {
		parsedEntries := parseModelPricingTS(string(pricingBody), providerDir)
		upstreamEntries = append(upstreamEntries, parsedEntries...)
	}

	if len(upstreamEntries) == 0 && len(discoveredProviders) == 0 {
		jsonBody, jsonErr := g.fetchUpstreamRaw(ctx, upstreamRepo, "catalog.json")
		if jsonErr == nil && len(jsonBody) > 0 {
			upstreamEntries = append(upstreamEntries, parseCatalogJSON(jsonBody, providerDir)...)
		}
	}

	if shaErr != nil && len(upstreamEntries) == 0 && len(discoveredProviders) == 0 {
		return SniffResult{
			Updated:     false,
			PreviousSHA: previousSHA,
		}, fmt.Errorf("sniff upstream %s failed: %w", upstreamRepo, shaErr)
	}

	mergedByKey := make(map[string]CatalogEntry)
	for _, seedEntry := range curatedFreeCatalog() {
		key := seedEntry.ProviderSlug + ":" + seedEntry.ModelID
		mergedByKey[key] = seedEntry
	}

	providersWithModels := make(map[string]bool)
	for _, upstreamEntry := range upstreamEntries {
		key := upstreamEntry.ProviderSlug + ":" + upstreamEntry.ModelID
		mergedByKey[key] = upstreamEntry
		providersWithModels[upstreamEntry.ProviderSlug] = true
	}

	for slug, meta := range discoveredProviders {
		if providersWithModels[slug] {
			continue
		}
		hasSeed := false
		for _, existing := range mergedByKey {
			if existing.ProviderSlug == slug {
				hasSeed = true
				break
			}
		}
		if !hasSeed && meta.BaseURL != "" {
			autoEntry := CatalogEntry{
				ProviderSlug:  meta.Slug,
				ProviderName:  meta.Name,
				BaseURL:       meta.BaseURL,
				ModelID:       "auto",
				ContextWindow: 128000,
				FreeTier:      true,
				Source:        "freellmapi",
			}
			mergedByKey[meta.Slug+":auto"] = autoEntry
		}
	}

	now := time.Now().UTC().Format(time.RFC3339)
	uniqueProviders := make(map[string]bool)

	tx, err := g.db.BeginTx(ctx, nil)
	if err != nil {
		return SniffResult{}, err
	}
	defer func() { _ = tx.Rollback() }()

	for _, entry := range mergedByKey {
		entryID := makeGatewayID("cat", entry.ProviderSlug, entry.ModelID)
		if entry.ContextWindow <= 0 {
			entry.ContextWindow = inferContextWindow(entry.ModelID)
		}
		uniqueProviders[entry.ProviderSlug] = true

		_, execErr := tx.ExecContext(ctx, `
INSERT INTO ai_catalog (id, provider_slug, provider_name, base_url, model_id, context_window, free_tier, source, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  provider_slug = excluded.provider_slug,
  provider_name = excluded.provider_name,
  base_url = excluded.base_url,
  model_id = excluded.model_id,
  context_window = excluded.context_window,
  free_tier = excluded.free_tier,
  source = excluded.source,
  updated_at = excluded.updated_at`,
			entryID, entry.ProviderSlug, entry.ProviderName, entry.BaseURL, entry.ModelID,
			entry.ContextWindow, boolToInt(entry.FreeTier), entry.Source, now,
		)
		if execErr != nil {
			return SniffResult{}, execErr
		}
	}

	if latestSHA != "" {
		if _, err := tx.ExecContext(ctx, `INSERT INTO ai_settings (key, value) VALUES ('upstream_sha', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value`, latestSHA); err != nil {
			return SniffResult{}, err
		}
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO ai_settings (key, value) VALUES ('last_sniff_at', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value`, now); err != nil {
		return SniffResult{}, err
	}

	if err := tx.Commit(); err != nil {
		return SniffResult{}, err
	}

	updated := (latestSHA != "" && latestSHA != previousSHA) || (previousSHA == "" && len(upstreamEntries) > 0)
	return SniffResult{
		Updated:        updated,
		UpstreamSHA:    latestSHA,
		PreviousSHA:    previousSHA,
		ProvidersFound: len(uniqueProviders),
		ModelsIndexed:  len(mergedByKey),
		LastSniffAt:    now,
	}, nil
}

func (g *Gateway) fetchUpstreamCommitSHA(ctx context.Context, repo string) (string, error) {
	apiBase := strings.TrimRight(g.GitHubAPIBase, "/")
	if apiBase == "" {
		apiBase = DefaultGitHubAPIBase
	}

	candidatePaths := []string{
		fmt.Sprintf("%s/repos/%s/commits/main", apiBase, repo),
		fmt.Sprintf("%s/repos/%s/commits/HEAD", apiBase, repo),
	}

	var lastErr error
	for _, endpoint := range candidatePaths {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			lastErr = err
			continue
		}
		httpReq.Header.Set("Accept", "application/vnd.github+json")
		httpReq.Header.Set("User-Agent", "mw-rocksoul-gateway")

		httpResp, err := g.httpClient.Do(httpReq)
		if err != nil {
			lastErr = err
			continue
		}
		bodyBytes, readErr := io.ReadAll(io.LimitReader(httpResp.Body, 512*1024))
		_ = httpResp.Body.Close()
		if readErr != nil {
			lastErr = readErr
			continue
		}
		if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
			lastErr = fmt.Errorf("github api %s returned %d", endpoint, httpResp.StatusCode)
			continue
		}

		var commitEnvelope struct {
			SHA string `json:"sha"`
		}
		if err := json.Unmarshal(bodyBytes, &commitEnvelope); err != nil {
			lastErr = err
			continue
		}
		if strings.TrimSpace(commitEnvelope.SHA) != "" {
			return strings.TrimSpace(commitEnvelope.SHA), nil
		}
		lastErr = errors.New("empty commit sha in github response")
	}
	return "", lastErr
}

func (g *Gateway) fetchUpstreamRaw(ctx context.Context, repo, relPath string) ([]byte, error) {
	rawBase := strings.TrimRight(g.RawBase, "/")
	if rawBase == "" {
		rawBase = DefaultRawBase
	}
	targetURL := fmt.Sprintf("%s/%s/main/%s", rawBase, repo, strings.TrimLeft(relPath, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("User-Agent", "mw-rocksoul-gateway")

	httpResp, err := g.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, fmt.Errorf("raw fetch %s returned %d", targetURL, httpResp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(httpResp.Body, 4*1024*1024))
}

func parseProvidersTS(sourceText string) []upstreamProviderMeta {
	matches := openAICompatBlockRegex.FindAllStringSubmatch(sourceText, -1)
	providers := make([]upstreamProviderMeta, 0, len(matches))
	seen := make(map[string]bool)

	for _, blockMatch := range matches {
		if len(blockMatch) < 2 {
			continue
		}
		body := blockMatch[1]
		platformMatch := platformFieldRegex.FindStringSubmatch(body)
		urlMatch := baseURLFieldRegex.FindStringSubmatch(body)
		if len(platformMatch) < 2 || len(urlMatch) < 2 {
			continue
		}
		slug := strings.ToLower(strings.TrimSpace(platformMatch[1]))
		if slug == "" || slug == "custom" || seen[slug] {
			continue
		}
		seen[slug] = true

		displayName := slug
		if nameMatch := nameFieldRegex.FindStringSubmatch(body); len(nameMatch) >= 2 {
			displayName = strings.TrimSpace(nameMatch[1])
		}
		providers = append(providers, upstreamProviderMeta{
			Slug:    slug,
			Name:    displayName,
			BaseURL: strings.TrimSpace(urlMatch[1]),
		})
	}
	return providers
}

func parseModelPricingTS(sourceText string, providerDir map[string]upstreamProviderMeta) []CatalogEntry {
	matches := modelPricingRowRegex.FindAllStringSubmatch(sourceText, -1)
	entries := make([]CatalogEntry, 0, len(matches))
	seen := make(map[string]bool)

	for _, rowMatch := range matches {
		if len(rowMatch) < 3 {
			continue
		}
		rawPlatform := strings.ToLower(strings.TrimSpace(rowMatch[1]))
		modelID := strings.TrimSpace(rowMatch[2])
		if rawPlatform == "" || modelID == "" {
			continue
		}

		meta, ok := providerDir[rawPlatform]
		if !ok {
			meta = upstreamProviderMeta{
				Slug:    rawPlatform,
				Name:    strings.ToUpper(rawPlatform[:1]) + rawPlatform[1:],
				BaseURL: "",
			}
		}

		dedupKey := meta.Slug + ":" + modelID
		if seen[dedupKey] {
			continue
		}
		seen[dedupKey] = true

		entries = append(entries, CatalogEntry{
			ProviderSlug:  meta.Slug,
			ProviderName:  meta.Name,
			BaseURL:       meta.BaseURL,
			ModelID:       modelID,
			ContextWindow: inferContextWindow(modelID),
			FreeTier:      true,
			Source:        "freellmapi",
		})
	}
	return entries
}

func parseCatalogJSON(rawJSON []byte, providerDir map[string]upstreamProviderMeta) []CatalogEntry {
	type jsonCatalogRow struct {
		ProviderSlug  string `json:"providerSlug"`
		Platform      string `json:"platform"`
		ProviderName  string `json:"providerName"`
		BaseURL       string `json:"baseUrl"`
		ModelID       string `json:"modelId"`
		ContextWindow int    `json:"contextWindow"`
		FreeTier      *bool  `json:"freeTier"`
	}

	var rows []jsonCatalogRow
	if err := json.Unmarshal(rawJSON, &rows); err != nil {
		var wrapped struct {
			Models  []jsonCatalogRow `json:"models"`
			Entries []jsonCatalogRow `json:"entries"`
		}
		if wrapErr := json.Unmarshal(rawJSON, &wrapped); wrapErr != nil {
			return nil
		}
		rows = append(rows, wrapped.Models...)
		rows = append(rows, wrapped.Entries...)
	}

	entries := make([]CatalogEntry, 0, len(rows))
	for _, row := range rows {
		slug := strings.ToLower(strings.TrimSpace(row.ProviderSlug))
		if slug == "" {
			slug = strings.ToLower(strings.TrimSpace(row.Platform))
		}
		modelID := strings.TrimSpace(row.ModelID)
		if slug == "" || modelID == "" {
			continue
		}
		meta := providerDir[slug]
		name := strings.TrimSpace(row.ProviderName)
		if name == "" {
			name = meta.Name
			if name == "" {
				name = slug
			}
		}
		baseURL := strings.TrimSpace(row.BaseURL)
		if baseURL == "" {
			baseURL = meta.BaseURL
		}
		ctxWindow := row.ContextWindow
		if ctxWindow <= 0 {
			ctxWindow = inferContextWindow(modelID)
		}
		freeTier := true
		if row.FreeTier != nil {
			freeTier = *row.FreeTier
		}
		entries = append(entries, CatalogEntry{
			ProviderSlug:  slug,
			ProviderName:  name,
			BaseURL:       baseURL,
			ModelID:       modelID,
			ContextWindow: ctxWindow,
			FreeTier:      freeTier,
			Source:        "freellmapi",
		})
	}
	return entries
}

func inferContextWindow(modelID string) int {
	lower := strings.ToLower(modelID)
	switch {
	case strings.Contains(lower, "gemini-2.5"), strings.Contains(lower, "gemini-3"):
		return 1048576
	case strings.Contains(lower, "qwen3-coder"), strings.Contains(lower, "kimi-k2"), strings.Contains(lower, "codestral"):
		return 262144
	case strings.Contains(lower, "llama-3.3"), strings.Contains(lower, "llama-4"), strings.Contains(lower, "gpt-oss"), strings.Contains(lower, "deepseek"), strings.Contains(lower, "gemma-4"):
		return 131072
	default:
		return 128000
	}
}
