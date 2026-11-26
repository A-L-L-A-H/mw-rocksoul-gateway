package gateway

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/localai"
	_ "modernc.org/sqlite"
)

// DataClassification controls egress privacy and sanitization policy per request.
type DataClassification string

const (
	ClassificationLocalOnly     DataClassification = "local_only"
	ClassificationCloudEligible DataClassification = "cloud_eligible"

	KindOpenAICompat = "openai_compat"
	KindAnthropic    = "anthropic"
	KindGemini       = "gemini"

	TierFree         = "free"
	TierCheap        = "cheap"
	TierSubscription = "subscription"
	TierPaid         = "paid"

	DefaultFallbackOrder = "free,cheap,subscription,paid"
	DefaultUpstreamRepo  = "tashfeenahmed/freellmapi"
	DefaultGitHubAPIBase = "https://api.github.com"
	DefaultRawBase       = "https://raw.githubusercontent.com"

	defaultCooldownDuration = 60 * time.Second
)

var seededAgentIDs = []string{
	"rocksoul",
	"claude-code",
	"codex",
	"gemini-cli",
	"opencode",
	"antigravity",
	"default",
}

// ParseDataClassification validates and normalizes a data classification value.
// An empty string returns ("", nil); unknown values fail closed with an error.
func ParseDataClassification(raw string) (DataClassification, error) {
	cleaned := strings.ToLower(strings.TrimSpace(raw))
	switch cleaned {
	case "":
		return "", nil
	case string(ClassificationLocalOnly):
		return ClassificationLocalOnly, nil
	case string(ClassificationCloudEligible):
		return ClassificationCloudEligible, nil
	default:
		return "", fmt.Errorf("invalid data classification %q: must be %q or %q", raw, ClassificationLocalOnly, ClassificationCloudEligible)
	}
}

// Options configures a standalone mw-rocksoul-gateway instance.
type Options struct {
	DataDir               string
	DefaultClassification DataClassification
	LocalDiscovery        *localai.Discoverer
	HTTPClient            *http.Client
	LoopbackHTTPClient    *http.Client
	GitHubAPIBase         string
	RawBase               string
}

// Gateway implements the sovereign AI routing gateway combining multi-provider
// failover, RTK & caveman token compression, PII/secret egress scrubbing,
// strict local_only air-gap enforcement, Magpie per-agent routing, exact SQLite
// response caching, and upstream catalog sniffing.
type Gateway struct {
	db                    *sql.DB
	dataDir               string
	box                   *SecretBox
	httpClient            *http.Client
	loopbackClient        *http.Client
	defaultClassification DataClassification
	localDiscovery        *localai.Discoverer

	GitHubAPIBase string
	RawBase       string

	mu sync.RWMutex
}

// Provider represents a sanitized AI provider configuration returned by the API.
// Encrypted credentials are never exposed.
type Provider struct {
	ID            string   `json:"id"`
	Name          string   `json:"name"`
	Kind          string   `json:"kind"`
	Tier          string   `json:"tier"`
	BaseURL       string   `json:"baseUrl"`
	IsLocal       bool     `json:"isLocal"`
	HasAPIKey     bool     `json:"hasApiKey"`
	DefaultModel  string   `json:"defaultModel"`
	Models        []string `json:"models"`
	Priority      int      `json:"priority"`
	Enabled       bool     `json:"enabled"`
	CooldownUntil string   `json:"cooldownUntil,omitempty"`
	InCooldown    bool     `json:"inCooldown"`
	ReqCount      int64    `json:"reqCount"`
	ErrCount      int64    `json:"errCount"`
	TokensIn      int64    `json:"tokensIn"`
	TokensOut     int64    `json:"tokensOut"`
	CreatedAt     string   `json:"createdAt"`
}

// ProviderCreateInput carries fields for registering a new AI provider.
type ProviderCreateInput struct {
	Name         string   `json:"name"`
	Kind         string   `json:"kind"`
	Tier         string   `json:"tier"`
	BaseURL      string   `json:"baseUrl"`
	APIKey       string   `json:"apiKey"`
	DefaultModel string   `json:"defaultModel"`
	Models       []string `json:"models"`
	Priority     int      `json:"priority"`
	Enabled      *bool    `json:"enabled,omitempty"`
}

type storedProvider struct {
	Provider
	APIKeyEnc string
}

// AgentRoute represents a per-agent model switchboard entry.
type AgentRoute struct {
	AgentID     string `json:"agentId"`
	TargetModel string `json:"targetModel"`
	TierOrder   string `json:"tierOrder"`
	RTKEnabled  bool   `json:"rtkEnabled"`
	UpdatedAt   string `json:"updatedAt"`
}

// AgentRouteUpdateInput carries partial updates for an agent route.
type AgentRouteUpdateInput struct {
	TargetModel *string `json:"targetModel,omitempty"`
	TierOrder   *string `json:"tierOrder,omitempty"`
	RTKEnabled  *bool   `json:"rtkEnabled,omitempty"`
}

// CatalogEntry represents an indexed upstream or curated AI model.
type CatalogEntry struct {
	ID            string `json:"id"`
	ProviderSlug  string `json:"providerSlug"`
	ProviderName  string `json:"providerName"`
	BaseURL       string `json:"baseUrl"`
	ModelID       string `json:"modelId"`
	ContextWindow int    `json:"contextWindow"`
	FreeTier      bool   `json:"freeTier"`
	Source        string `json:"source"`
	UpdatedAt     string `json:"updatedAt"`
}

// SniffResult summarizes an upstream catalog synchronization run.
type SniffResult struct {
	Updated        bool   `json:"updated"`
	UpstreamSHA    string `json:"upstreamSha"`
	PreviousSHA    string `json:"previousSha"`
	ProvidersFound int    `json:"providersFound"`
	ModelsIndexed  int    `json:"modelsIndexed"`
	LastSniffAt    string `json:"lastSniffAt,omitempty"`
}

// GatewaySettings captures persisted gateway-wide feature toggles and upstream config.
type GatewaySettings struct {
	RTKEnabled    bool   `json:"rtkEnabled"`
	CacheEnabled  bool   `json:"cacheEnabled"`
	FallbackOrder string `json:"fallbackOrder"`
	UpstreamRepo  string `json:"upstreamRepo"`
	UpstreamSHA   string `json:"upstreamSha"`
	LastSniffAt   string `json:"lastSniffAt"`
}

// SettingsUpdateInput carries updates to gateway settings.
type SettingsUpdateInput struct {
	RTKEnabled    *bool   `json:"rtkEnabled,omitempty"`
	CacheEnabled  *bool   `json:"cacheEnabled,omitempty"`
	FallbackOrder *string `json:"fallbackOrder,omitempty"`
	UpstreamRepo  *string `json:"upstreamRepo,omitempty"`
}

// RequestLog records telemetry for a single routed AI request.
type RequestLog struct {
	ID               string `json:"id"`
	AgentID          string `json:"agentId"`
	RequestedModel   string `json:"requestedModel"`
	ResolvedProvider string `json:"resolvedProvider"`
	ResolvedModel    string `json:"resolvedModel"`
	Classification   string `json:"classification"`
	StatusCode       int    `json:"statusCode"`
	LatencyMS        int64  `json:"latencyMs"`
	PromptTokens     int    `json:"promptTokens"`
	CompletionTokens int    `json:"completionTokens"`
	RTKTokensSaved   int    `json:"rtkTokensSaved"`
	RedactionsCount  int    `json:"redactionsCount"`
	EgressBlocked    bool   `json:"egressBlocked"`
	CacheHit         bool   `json:"cacheHit"`
	CreatedAt        string `json:"createdAt"`
}

// GatewayStats aggregates gateway usage, savings, inventory, and recent logs.
type GatewayStats struct {
	TotalRequests          int64           `json:"totalRequests"`
	CacheHits              int64           `json:"cacheHits"`
	TotalRTKTokensSaved    int64           `json:"totalRtkTokensSaved"`
	CacheTokensSaved       int64           `json:"cacheTokensSaved"`
	TotalPIIRedactions     int64           `json:"totalPiiRedactions"`
	LocalOnlyBlockedEgress int64           `json:"localOnlyBlockedEgress"`
	AvgLatencyMS           float64         `json:"avgLatencyMs"`
	ProvidersCount         int             `json:"providersCount"`
	CatalogCount           int             `json:"catalogCount"`
	Settings               GatewaySettings `json:"settings"`
	RecentLogs             []RequestLog    `json:"recentLogs"`
}

// ProviderUsage summarizes cumulative usage per provider.
type ProviderUsage struct {
	ProviderID string `json:"providerId"`
	Name       string `json:"name"`
	Tier       string `json:"tier"`
	IsLocal    bool   `json:"isLocal"`
	Requests   int64  `json:"requests"`
	Errors     int64  `json:"errors"`
	TokensIn   int64  `json:"tokensIn"`
	TokensOut  int64  `json:"tokensOut"`
}

// AgentUsage summarizes cumulative usage per agent profile.
type AgentUsage struct {
	AgentID          string `json:"agentId"`
	Requests         int64  `json:"requests"`
	PromptTokens     int64  `json:"promptTokens"`
	CompletionTokens int64  `json:"completionTokens"`
	RTKTokensSaved   int64  `json:"rtkTokensSaved"`
	RedactionsCount  int64  `json:"redactionsCount"`
}

// UsageSummary is the payload returned by GET /v1/usage.
type UsageSummary struct {
	TotalRequests          int64           `json:"totalRequests"`
	PromptTokens           int64           `json:"promptTokens"`
	CompletionTokens       int64           `json:"completionTokens"`
	TotalTokens            int64           `json:"totalTokens"`
	TotalRTKTokensSaved    int64           `json:"totalRtkTokensSaved"`
	CacheHits              int64           `json:"cacheHits"`
	CacheTokensSaved       int64           `json:"cacheTokensSaved"`
	TotalPIIRedactions     int64           `json:"totalPiiRedactions"`
	LocalOnlyBlockedEgress int64           `json:"localOnlyBlockedEgress"`
	AvgLatencyMS           float64         `json:"avgLatencyMs"`
	ByProvider             []ProviderUsage `json:"byProvider"`
	ByAgent                []AgentUsage    `json:"byAgent"`
	RecentLogs             []RequestLog    `json:"recentLogs"`
}

// ChatMessage matches OpenAI chat completion message wire structure and also
// carries structured content slices (including Anthropic tool_result blocks).
type ChatMessage struct {
	Role       string          `json:"role"`
	Content    any             `json:"content"`
	Name       string          `json:"name,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
}

// ChatCompletionRequest matches OpenAI POST /v1/chat/completions wire format
// with optional Rocksoul data_classification policy field.
type ChatCompletionRequest struct {
	Model              string          `json:"model"`
	Messages           []ChatMessage   `json:"messages"`
	Temperature        *float64        `json:"temperature,omitempty"`
	MaxTokens          *int            `json:"max_tokens,omitempty"`
	TopP               *float64        `json:"top_p,omitempty"`
	Stream             bool            `json:"stream,omitempty"`
	Tools              json.RawMessage `json:"tools,omitempty"`
	ToolChoice         json.RawMessage `json:"tool_choice,omitempty"`
	Stop               json.RawMessage `json:"stop,omitempty"`
	DataClassification string          `json:"data_classification,omitempty"`
}

// ChatCompletionChoice represents one completion candidate returned by an upstream provider.
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatCompletionUsage reports OpenAI token consumption.
type ChatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse matches OpenAI chat completion response JSON.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   ChatCompletionUsage    `json:"usage"`
}

// OpenAIModelObject represents a single model entry in GET /v1/models.
type OpenAIModelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// OpenAIModelList represents the GET /v1/models envelope.
type OpenAIModelList struct {
	Object string              `json:"object"`
	Models []OpenAIModelObject `json:"data"`
}

// AnthropicMessage represents a message in Anthropic POST /v1/messages.
type AnthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// AnthropicMessageRequest matches Anthropic Messages API input.
type AnthropicMessageRequest struct {
	Model              string             `json:"model"`
	System             any                `json:"system,omitempty"`
	Messages           []AnthropicMessage `json:"messages"`
	MaxTokens          int                `json:"max_tokens,omitempty"`
	Temperature        *float64           `json:"temperature,omitempty"`
	TopP               *float64           `json:"top_p,omitempty"`
	Stream             bool               `json:"stream,omitempty"`
	StopSequences      []string           `json:"stop_sequences,omitempty"`
	DataClassification string             `json:"data_classification,omitempty"`
}

// AnthropicContentBlock represents an output block in an Anthropic message response.
type AnthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// AnthropicUsage tracks Anthropic input/output token counts.
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// AnthropicMessageResponse matches Anthropic Messages API output format.
type AnthropicMessageResponse struct {
	ID         string                  `json:"id"`
	Type       string                  `json:"type"`
	Role       string                  `json:"role"`
	Model      string                  `json:"model"`
	Content    []AnthropicContentBlock `json:"content"`
	StopReason string                  `json:"stop_reason"`
	Usage      AnthropicUsage          `json:"usage"`
}

// NewGateway initializes a standalone Gateway backed by SQLite in dataDir.
func NewGateway(dataDir string) (*Gateway, error) {
	return Open(Options{DataDir: dataDir})
}

// Open initializes SQLite storage, AES-256-GCM SecretBox, schema migrations,
// default agent routes, and curated catalog entries.
func Open(opts Options) (*Gateway, error) {
	cleanDir := strings.TrimSpace(opts.DataDir)
	if cleanDir == "" {
		return nil, errors.New("gateway dataDir is required")
	}
	cleanDir = filepath.Clean(cleanDir)
	if err := os.MkdirAll(cleanDir, 0o755); err != nil {
		return nil, fmt.Errorf("create gateway data dir: %w", err)
	}

	defaultClass := opts.DefaultClassification
	if defaultClass == "" {
		defaultClass = ClassificationCloudEligible
	} else {
		parsed, err := ParseDataClassification(string(defaultClass))
		if err != nil {
			return nil, err
		}
		defaultClass = parsed
	}

	dbPath := filepath.Join(cleanDir, "mw-rocksoul-gateway.db")
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=foreign_keys(ON)", filepath.ToSlash(dbPath))
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open gateway sqlite: %w", err)
	}
	db.SetMaxOpenConns(1)

	box, err := OpenSecretBox(cleanDir)
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("open secret box: %w", err)
	}

	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 90 * time.Second}
	}
	loopbackClient := opts.LoopbackHTTPClient
	if loopbackClient == nil {
		loopbackClient = localai.NewLoopbackHTTPClient(90 * time.Second)
	}
	ghBase := strings.TrimSpace(opts.GitHubAPIBase)
	if ghBase == "" {
		ghBase = DefaultGitHubAPIBase
	}
	rawBase := strings.TrimSpace(opts.RawBase)
	if rawBase == "" {
		rawBase = DefaultRawBase
	}

	gw := &Gateway{
		db:                    db,
		dataDir:               cleanDir,
		box:                   box,
		httpClient:            httpClient,
		loopbackClient:        loopbackClient,
		defaultClassification: defaultClass,
		localDiscovery:        opts.LocalDiscovery,
		GitHubAPIBase:         ghBase,
		RawBase:               rawBase,
	}
	if err := gw.migrate(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return gw, nil
}

// Close closes the underlying SQLite database handle.
func (g *Gateway) Close() error {
	if g == nil || g.db == nil {
		return nil
	}
	return g.db.Close()
}

// DataDir returns the gateway's local state directory.
func (g *Gateway) DataDir() string {
	return g.dataDir
}

// DB returns the underlying SQLite handle.
func (g *Gateway) DB() *sql.DB {
	return g.db
}

// DefaultClassification returns the gateway-wide default data classification policy.
func (g *Gateway) DefaultClassification() DataClassification {
	return g.defaultClassification
}

func (g *Gateway) migrate() error {
	schema := `
CREATE TABLE IF NOT EXISTS ai_providers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'openai_compat',
  tier TEXT NOT NULL DEFAULT 'free',
  base_url TEXT NOT NULL,
  api_key_enc TEXT NOT NULL DEFAULT '',
  default_model TEXT NOT NULL DEFAULT '',
  models_json TEXT NOT NULL DEFAULT '[]',
  priority INTEGER NOT NULL DEFAULT 100,
  enabled INTEGER NOT NULL DEFAULT 1,
  cooldown_until TEXT NOT NULL DEFAULT '',
  req_count INTEGER NOT NULL DEFAULT 0,
  err_count INTEGER NOT NULL DEFAULT 0,
  tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_agent_routes (
  agent_id TEXT PRIMARY KEY,
  target_model TEXT NOT NULL DEFAULT '',
  tier_order TEXT NOT NULL DEFAULT 'free,cheap,subscription,paid',
  rtk_enabled INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_catalog (
  id TEXT PRIMARY KEY,
  provider_slug TEXT NOT NULL,
  provider_name TEXT NOT NULL,
  base_url TEXT NOT NULL,
  model_id TEXT NOT NULL,
  context_window INTEGER NOT NULL DEFAULT 128000,
  free_tier INTEGER NOT NULL DEFAULT 1,
  source TEXT NOT NULL DEFAULT 'seed',
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ai_catalog_provider ON ai_catalog(provider_slug);
CREATE INDEX IF NOT EXISTS idx_ai_catalog_model ON ai_catalog(model_id);

CREATE TABLE IF NOT EXISTS ai_response_cache (
  cache_key TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  response_json TEXT NOT NULL,
  tokens_saved INTEGER NOT NULL DEFAULT 0,
  hits INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_request_logs (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL DEFAULT 'default',
  requested_model TEXT NOT NULL,
  resolved_provider TEXT NOT NULL DEFAULT '',
  resolved_model TEXT NOT NULL DEFAULT '',
  classification TEXT NOT NULL DEFAULT 'cloud_eligible',
  status_code INTEGER NOT NULL DEFAULT 0,
  latency_ms INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  rtk_tokens_saved INTEGER NOT NULL DEFAULT 0,
  redactions_count INTEGER NOT NULL DEFAULT 0,
  egress_blocked INTEGER NOT NULL DEFAULT 0,
  cache_hit INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ai_request_logs_created ON ai_request_logs(created_at DESC);

CREATE TABLE IF NOT EXISTS ai_settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
`
	if _, err := g.db.Exec(schema); err != nil {
		return fmt.Errorf("migrate ai gateway tables: %w", err)
	}

	now := time.Now().UTC().Format(time.RFC3339)

	defaultSettings := [][2]string{
		{"rtk_enabled", "true"},
		{"cache_enabled", "true"},
		{"fallback_order", DefaultFallbackOrder},
		{"upstream_repo", DefaultUpstreamRepo},
		{"upstream_sha", ""},
		{"last_sniff_at", ""},
	}
	for _, settingPair := range defaultSettings {
		if _, err := g.db.Exec(`INSERT OR IGNORE INTO ai_settings (key, value) VALUES (?, ?)`, settingPair[0], settingPair[1]); err != nil {
			return fmt.Errorf("seed ai setting %s: %w", settingPair[0], err)
		}
	}

	for _, agentID := range seededAgentIDs {
		if _, err := g.db.Exec(`
INSERT OR IGNORE INTO ai_agent_routes (agent_id, target_model, tier_order, rtk_enabled, updated_at)
VALUES (?, '', ?, 1, ?)`, agentID, DefaultFallbackOrder, now); err != nil {
			return fmt.Errorf("seed agent route %s: %w", agentID, err)
		}
	}

	if err := g.seedInitialCatalog(context.Background()); err != nil {
		return fmt.Errorf("seed initial ai catalog: %w", err)
	}

	return nil
}

func makeGatewayID(prefix string, parts ...string) string {
	hasher := sha256.New()
	for _, part := range parts {
		hasher.Write([]byte(part))
		hasher.Write([]byte{0})
	}
	return prefix + "_" + hex.EncodeToString(hasher.Sum(nil))[:16]
}

func normalizeTierOrder(raw string) (string, error) {
	parts := strings.Split(raw, ",")
	seen := make(map[string]bool, len(parts))
	ordered := make([]string, 0, len(parts))
	for _, part := range parts {
		tier := strings.TrimSpace(strings.ToLower(part))
		if tier == "" {
			continue
		}
		if !isValidTier(tier) {
			return "", fmt.Errorf("invalid tier %q; must be one of free, cheap, subscription, paid", tier)
		}
		if !seen[tier] {
			seen[tier] = true
			ordered = append(ordered, tier)
		}
	}
	if len(ordered) == 0 {
		return DefaultFallbackOrder, nil
	}
	return strings.Join(ordered, ","), nil
}

func isValidTier(tier string) bool {
	switch tier {
	case TierFree, TierCheap, TierSubscription, TierPaid:
		return true
	default:
		return false
	}
}

func isValidProviderKind(kind string) bool {
	switch kind {
	case KindOpenAICompat, KindAnthropic, KindGemini:
		return true
	default:
		return false
	}
}

func boolToInt(flag bool) int {
	if flag {
		return 1
	}
	return 0
}
