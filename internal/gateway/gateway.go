package gateway

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/localai"
)

const (
	headerAgentID            = "X-MW-Agent"
	headerRocksoulAgentID    = "X-Rocksoul-Agent"
	headerRTKMode            = "X-MW-RTK-Mode"
	headerRocksoulRTKMode    = "X-Rocksoul-RTK-Mode"
	headerClassification     = "X-Rocksoul-Data-Classification"
	headerMWClassification   = "X-MW-Data-Classification"
	headerCacheState         = "X-Rocksoul-Cache"
	headerEchoCacheState     = "X-MW-Echo-Cache"
	headerProvider           = "X-Rocksoul-Provider"
	headerEchoProvider       = "X-MW-Echo-Provider"
	headerModel              = "X-Rocksoul-Model"
	headerEchoModel          = "X-MW-Echo-Model"
	headerRTKSaved           = "X-Rocksoul-RTK-Saved"
	headerEchoRTKSaved       = "X-MW-Echo-RTK-Saved"
	headerOutClassification  = "X-Rocksoul-Classification"
	headerOutRedactions      = "X-Rocksoul-Redactions"
	maxIncomingRequestBytes  = 8 << 20 // 8 MiB
	maxUpstreamResponseBytes = 16 << 20
)

type routeOutcome struct {
	Response         ChatCompletionResponse
	RawBody          []byte
	ResolvedProvider string
	ResolvedModel    string
	AgentID          string
	Classification   DataClassification
	CacheHit         bool
	RTKTokensSaved   int
	RedactionsCount  int
	LatencyMS        int64
}

type upstreamCallError struct {
	StatusCode int
	Code       string
	Body       string
	Retryable  bool
	Cooldown   bool
}

func (e *upstreamCallError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s (HTTP %d): %s", e.Code, e.StatusCode, e.Body)
	}
	return fmt.Sprintf("upstream HTTP %d: %s", e.StatusCode, e.Body)
}

// RegisterRoutes mounts all OpenAI/Anthropic/Rocksoul gateway routes and
// control-plane endpoints onto mux.
func (g *Gateway) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/models", g.handleListOpenAIModels)
	mux.HandleFunc("POST /v1/chat/completions", g.handleChatCompletions)
	mux.HandleFunc("POST /v1/audio/transcriptions", g.handleAudioTranscriptions)
	mux.HandleFunc("POST /v1/rerank", g.handleRerank)
	mux.HandleFunc("POST /v1/messages", g.handleAnthropicMessages)
	mux.HandleFunc("GET /v1/usage", g.handleGetUsage)
	mux.HandleFunc("GET /v1/local-providers", g.handleListLocalProviders)

	mux.HandleFunc("GET /api/ai/stats", g.handleGetStats)
	mux.HandleFunc("GET /api/ai/providers", g.handleListProviders)
	mux.HandleFunc("POST /api/ai/providers", g.handleCreateProvider)
	mux.HandleFunc("DELETE /api/ai/providers/{id}", g.handleDeleteProvider)
	mux.HandleFunc("GET /api/ai/agents", g.handleListAgents)
	mux.HandleFunc("PUT /api/ai/agents/{agentId}", g.handleUpdateAgent)
	mux.HandleFunc("GET /api/ai/catalog", g.handleListCatalog)
	mux.HandleFunc("POST /api/ai/catalog/sniff", g.handleSniffCatalog)
	mux.HandleFunc("PUT /api/ai/settings", g.handleUpdateSettings)
	mux.HandleFunc("DELETE /api/ai/cache", g.handlePurgeCache)
}

// Handler returns a ready-to-serve http.Handler with all gateway routes mounted.
func (g *Gateway) Handler() http.Handler {
	mux := http.NewServeMux()
	g.RegisterRoutes(mux)
	return mux
}

// --- Settings & Store Operations ---

// GetSettings loads gateway settings from ai_settings with safe defaults.
func (g *Gateway) GetSettings(ctx context.Context) GatewaySettings {
	settings := GatewaySettings{
		RTKEnabled:    true,
		CacheEnabled:  true,
		FallbackOrder: DefaultFallbackOrder,
		UpstreamRepo:  DefaultUpstreamRepo,
	}

	rows, err := g.db.QueryContext(ctx, `SELECT key, value FROM ai_settings`)
	if err != nil {
		return settings
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var settingKey, settingVal string
		if rows.Scan(&settingKey, &settingVal) != nil {
			continue
		}
		switch settingKey {
		case "rtk_enabled":
			settings.RTKEnabled = settingVal != "false" && settingVal != "0"
		case "cache_enabled":
			settings.CacheEnabled = settingVal != "false" && settingVal != "0"
		case "fallback_order":
			if strings.TrimSpace(settingVal) != "" {
				settings.FallbackOrder = settingVal
			}
		case "upstream_repo":
			if strings.TrimSpace(settingVal) != "" {
				settings.UpstreamRepo = settingVal
			}
		case "upstream_sha":
			settings.UpstreamSHA = settingVal
		case "last_sniff_at":
			settings.LastSniffAt = settingVal
		}
	}
	return settings
}

// UpdateSettings updates gateway feature flags and fallback ordering.
func (g *Gateway) UpdateSettings(ctx context.Context, input SettingsUpdateInput) (GatewaySettings, error) {
	upsert := func(key, value string) error {
		_, err := g.db.ExecContext(ctx, `
INSERT INTO ai_settings (key, value) VALUES (?, ?)
ON CONFLICT(key) DO UPDATE SET value = excluded.value`, key, value)
		return err
	}

	if input.RTKEnabled != nil {
		if err := upsert("rtk_enabled", strconv.FormatBool(*input.RTKEnabled)); err != nil {
			return GatewaySettings{}, err
		}
	}
	if input.CacheEnabled != nil {
		if err := upsert("cache_enabled", strconv.FormatBool(*input.CacheEnabled)); err != nil {
			return GatewaySettings{}, err
		}
	}
	if input.FallbackOrder != nil {
		normalized, err := normalizeTierOrder(*input.FallbackOrder)
		if err != nil {
			return GatewaySettings{}, err
		}
		if err := upsert("fallback_order", normalized); err != nil {
			return GatewaySettings{}, err
		}
	}
	if input.UpstreamRepo != nil {
		repo := strings.TrimSpace(*input.UpstreamRepo)
		if repo == "" || !strings.Contains(repo, "/") {
			return GatewaySettings{}, errors.New("upstreamRepo must be in owner/repo format")
		}
		if err := upsert("upstream_repo", repo); err != nil {
			return GatewaySettings{}, err
		}
	}
	return g.GetSettings(ctx), nil
}

// AddProvider validates, encrypts credentials with SecretBox, and stores a new AI provider.
func (g *Gateway) AddProvider(ctx context.Context, input ProviderCreateInput) (Provider, error) {
	name := strings.TrimSpace(input.Name)
	if name == "" {
		return Provider{}, errors.New("provider name is required")
	}

	kind := strings.TrimSpace(strings.ToLower(input.Kind))
	if kind == "" {
		kind = KindOpenAICompat
	}
	if !isValidProviderKind(kind) {
		return Provider{}, fmt.Errorf("kind must be one of %s, %s, %s", KindOpenAICompat, KindAnthropic, KindGemini)
	}

	tier := strings.TrimSpace(strings.ToLower(input.Tier))
	if tier == "" {
		tier = TierFree
	}
	if !isValidTier(tier) {
		return Provider{}, fmt.Errorf("tier must be one of %s, %s, %s, %s", TierFree, TierCheap, TierSubscription, TierPaid)
	}

	baseURL := strings.TrimRight(strings.TrimSpace(input.BaseURL), "/")
	if baseURL == "" || (!strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://")) {
		return Provider{}, errors.New("valid http(s) baseUrl is required")
	}

	apiKeyEnc := ""
	trimmedKey := strings.TrimSpace(input.APIKey)
	if trimmedKey != "" {
		encrypted, err := g.box.Encrypt([]byte(trimmedKey))
		if err != nil {
			return Provider{}, fmt.Errorf("encrypt api key: %w", err)
		}
		apiKeyEnc = encrypted
	}

	cleanModels := make([]string, 0, len(input.Models))
	seenModels := make(map[string]bool)
	for _, rawModel := range input.Models {
		trimmedModel := strings.TrimSpace(rawModel)
		if trimmedModel != "" && !seenModels[trimmedModel] {
			seenModels[trimmedModel] = true
			cleanModels = append(cleanModels, trimmedModel)
		}
	}
	defaultModel := strings.TrimSpace(input.DefaultModel)
	if defaultModel == "" && len(cleanModels) > 0 {
		defaultModel = cleanModels[0]
	}
	if defaultModel != "" && !seenModels[defaultModel] && len(cleanModels) > 0 {
		cleanModels = append([]string{defaultModel}, cleanModels...)
	}

	modelsBytes, _ := json.Marshal(cleanModels)

	priority := input.Priority
	if priority <= 0 {
		priority = 100
	}
	enabled := true
	if input.Enabled != nil {
		enabled = *input.Enabled
	}

	now := time.Now().UTC().Format(time.RFC3339)
	providerID := makeGatewayID("aip", kind, name, time.Now().UTC().Format(time.RFC3339Nano))

	_, err := g.db.ExecContext(ctx, `
INSERT INTO ai_providers (
  id, name, kind, tier, base_url, api_key_enc, default_model, models_json,
  priority, enabled, cooldown_until, req_count, err_count, tokens_in, tokens_out, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 0, 0, 0, 0, ?)`,
		providerID, name, kind, tier, baseURL, apiKeyEnc, defaultModel, string(modelsBytes),
		priority, boolToInt(enabled), now,
	)
	if err != nil {
		return Provider{}, err
	}

	return Provider{
		ID:           providerID,
		Name:         name,
		Kind:         kind,
		Tier:         tier,
		BaseURL:      baseURL,
		IsLocal:      localai.IsLoopbackEndpointURL(baseURL),
		HasAPIKey:    apiKeyEnc != "",
		DefaultModel: defaultModel,
		Models:       cleanModels,
		Priority:     priority,
		Enabled:      enabled,
		CreatedAt:    now,
	}, nil
}

// ListProviders returns all providers with secrets stripped.
func (g *Gateway) ListProviders(ctx context.Context) ([]Provider, error) {
	storedList, err := g.loadStoredProviders(ctx, false)
	if err != nil {
		return nil, err
	}
	sanitized := make([]Provider, 0, len(storedList))
	for _, entry := range storedList {
		sanitized = append(sanitized, entry.Provider)
	}
	return sanitized, nil
}

func (g *Gateway) loadStoredProviders(ctx context.Context, enabledOnly bool) ([]storedProvider, error) {
	querySQL := `
SELECT id, name, kind, tier, base_url, api_key_enc, default_model, models_json,
       priority, enabled, cooldown_until, req_count, err_count, tokens_in, tokens_out, created_at
FROM ai_providers`
	if enabledOnly {
		querySQL += ` WHERE enabled = 1`
	}
	querySQL += ` ORDER BY priority ASC, created_at ASC`

	rows, err := g.db.QueryContext(ctx, querySQL)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	now := time.Now().UTC()
	providers := make([]storedProvider, 0)
	for rows.Next() {
		var record storedProvider
		var modelsRaw string
		var enabledInt int
		if err := rows.Scan(
			&record.ID, &record.Name, &record.Kind, &record.Tier, &record.BaseURL,
			&record.APIKeyEnc, &record.DefaultModel, &modelsRaw, &record.Priority,
			&enabledInt, &record.CooldownUntil, &record.ReqCount, &record.ErrCount,
			&record.TokensIn, &record.TokensOut, &record.CreatedAt,
		); err != nil {
			return nil, err
		}
		record.Enabled = enabledInt == 1
		record.IsLocal = localai.IsLoopbackEndpointURL(record.BaseURL)
		record.HasAPIKey = record.APIKeyEnc != ""
		record.Models = make([]string, 0)
		_ = json.Unmarshal([]byte(modelsRaw), &record.Models)
		if record.CooldownUntil != "" {
			if untilTime, parseErr := time.Parse(time.RFC3339, record.CooldownUntil); parseErr == nil && untilTime.After(now) {
				record.InCooldown = true
			}
		}
		providers = append(providers, record)
	}
	return providers, rows.Err()
}

// DeleteProvider removes a configured provider by ID.
func (g *Gateway) DeleteProvider(ctx context.Context, providerID string) error {
	execOut, err := g.db.ExecContext(ctx, `DELETE FROM ai_providers WHERE id = ?`, providerID)
	if err != nil {
		return err
	}
	affected, _ := execOut.RowsAffected()
	if affected == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// ListAgentRoutes returns all per-agent routing entries.
func (g *Gateway) ListAgentRoutes(ctx context.Context) ([]AgentRoute, error) {
	rows, err := g.db.QueryContext(ctx, `
SELECT agent_id, target_model, tier_order, rtk_enabled, updated_at
FROM ai_agent_routes ORDER BY CASE agent_id WHEN 'default' THEN 1 ELSE 0 END, agent_id ASC`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	routes := make([]AgentRoute, 0)
	for rows.Next() {
		var route AgentRoute
		var rtkInt int
		if err := rows.Scan(&route.AgentID, &route.TargetModel, &route.TierOrder, &rtkInt, &route.UpdatedAt); err != nil {
			return nil, err
		}
		route.RTKEnabled = rtkInt == 1
		routes = append(routes, route)
	}
	return routes, rows.Err()
}

// GetAgentRoute fetches a single agent route by ID.
func (g *Gateway) GetAgentRoute(ctx context.Context, agentID string) (AgentRoute, error) {
	var route AgentRoute
	var rtkInt int
	err := g.db.QueryRowContext(ctx, `
SELECT agent_id, target_model, tier_order, rtk_enabled, updated_at
FROM ai_agent_routes WHERE agent_id = ?`, agentID).Scan(
		&route.AgentID, &route.TargetModel, &route.TierOrder, &rtkInt, &route.UpdatedAt,
	)
	if err != nil {
		return AgentRoute{}, err
	}
	route.RTKEnabled = rtkInt == 1
	return route, nil
}

// UpdateAgentRoute creates or updates a per-agent model switchboard route.
func (g *Gateway) UpdateAgentRoute(ctx context.Context, agentID string, input AgentRouteUpdateInput) (AgentRoute, error) {
	cleanID := strings.TrimSpace(strings.ToLower(agentID))
	if cleanID == "" {
		return AgentRoute{}, errors.New("agentId is required")
	}

	existing, err := g.GetAgentRoute(ctx, cleanID)
	if errors.Is(err, sql.ErrNoRows) {
		existing = AgentRoute{
			AgentID:    cleanID,
			TierOrder:  DefaultFallbackOrder,
			RTKEnabled: true,
		}
	} else if err != nil {
		return AgentRoute{}, err
	}

	if input.TargetModel != nil {
		existing.TargetModel = strings.TrimSpace(*input.TargetModel)
	}
	if input.TierOrder != nil {
		normalized, tierErr := normalizeTierOrder(*input.TierOrder)
		if tierErr != nil {
			return AgentRoute{}, tierErr
		}
		existing.TierOrder = normalized
	}
	if input.RTKEnabled != nil {
		existing.RTKEnabled = *input.RTKEnabled
	}
	existing.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	_, err = g.db.ExecContext(ctx, `
INSERT INTO ai_agent_routes (agent_id, target_model, tier_order, rtk_enabled, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(agent_id) DO UPDATE SET
  target_model = excluded.target_model,
  tier_order = excluded.tier_order,
  rtk_enabled = excluded.rtk_enabled,
  updated_at = excluded.updated_at`,
		existing.AgentID, existing.TargetModel, existing.TierOrder, boolToInt(existing.RTKEnabled), existing.UpdatedAt,
	)
	return existing, err
}

// ListCatalog queries ai_catalog with optional keyword search and free-tier filter.
func (g *Gateway) ListCatalog(ctx context.Context, searchQuery string, freeOnly bool) ([]CatalogEntry, error) {
	clauses := make([]string, 0, 2)
	args := make([]any, 0, 4)

	if trimmed := strings.TrimSpace(strings.ToLower(searchQuery)); trimmed != "" {
		pattern := "%" + trimmed + "%"
		clauses = append(clauses, `(LOWER(provider_slug) LIKE ? OR LOWER(provider_name) LIKE ? OR LOWER(model_id) LIKE ? OR LOWER(base_url) LIKE ?)`)
		args = append(args, pattern, pattern, pattern, pattern)
	}
	if freeOnly {
		clauses = append(clauses, `free_tier = 1`)
	}

	whereSQL := ""
	if len(clauses) > 0 {
		whereSQL = "WHERE " + strings.Join(clauses, " AND ")
	}

	rows, err := g.db.QueryContext(ctx, fmt.Sprintf(`
SELECT id, provider_slug, provider_name, base_url, model_id, context_window, free_tier, source, updated_at
FROM ai_catalog %s
ORDER BY free_tier DESC, provider_slug ASC, model_id ASC`, whereSQL), args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	entries := make([]CatalogEntry, 0)
	for rows.Next() {
		var entry CatalogEntry
		var freeInt int
		if err := rows.Scan(
			&entry.ID, &entry.ProviderSlug, &entry.ProviderName, &entry.BaseURL,
			&entry.ModelID, &entry.ContextWindow, &freeInt, &entry.Source, &entry.UpdatedAt,
		); err != nil {
			return nil, err
		}
		entry.FreeTier = freeInt == 1
		entries = append(entries, entry)
	}
	return entries, rows.Err()
}

// PurgeCache clears all cached responses from ai_response_cache.
func (g *Gateway) PurgeCache(ctx context.Context) (int64, error) {
	execOut, err := g.db.ExecContext(ctx, `DELETE FROM ai_response_cache`)
	if err != nil {
		return 0, err
	}
	affected, _ := execOut.RowsAffected()
	return affected, nil
}

// Stats aggregates gateway telemetry, token savings, redactions, and recent logs.
func (g *Gateway) Stats(ctx context.Context) (GatewayStats, error) {
	stats := GatewayStats{
		Settings:   g.GetSettings(ctx),
		RecentLogs: make([]RequestLog, 0),
	}

	_ = g.db.QueryRowContext(ctx, `
SELECT COUNT(*),
       COALESCE(SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END), 0),
       COALESCE(SUM(rtk_tokens_saved), 0),
       COALESCE(SUM(redactions_count), 0),
       COALESCE(SUM(CASE WHEN egress_blocked = 1 THEN 1 ELSE 0 END), 0),
       COALESCE(AVG(latency_ms), 0)
FROM ai_request_logs`).Scan(
		&stats.TotalRequests,
		&stats.CacheHits,
		&stats.TotalRTKTokensSaved,
		&stats.TotalPIIRedactions,
		&stats.LocalOnlyBlockedEgress,
		&stats.AvgLatencyMS,
	)

	_ = g.db.QueryRowContext(ctx, `SELECT COALESCE(SUM(tokens_saved), 0) FROM ai_response_cache`).Scan(&stats.CacheTokensSaved)
	_ = g.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM ai_providers`).Scan(&stats.ProvidersCount)
	_ = g.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM ai_catalog`).Scan(&stats.CatalogCount)

	recent, err := g.listRecentRequestLogs(ctx, 25)
	if err != nil {
		return stats, err
	}
	stats.RecentLogs = recent
	return stats, nil
}

// Usage aggregates comprehensive token, savings, PII redaction, and egress metrics for GET /v1/usage.
func (g *Gateway) Usage(ctx context.Context) (UsageSummary, error) {
	summary := UsageSummary{
		ByProvider: make([]ProviderUsage, 0),
		ByAgent:    make([]AgentUsage, 0),
		RecentLogs: make([]RequestLog, 0),
	}

	_ = g.db.QueryRowContext(ctx, `
SELECT COUNT(*),
       COALESCE(SUM(prompt_tokens), 0),
       COALESCE(SUM(completion_tokens), 0),
       COALESCE(SUM(rtk_tokens_saved), 0),
       COALESCE(SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END), 0),
       COALESCE(SUM(redactions_count), 0),
       COALESCE(SUM(CASE WHEN egress_blocked = 1 THEN 1 ELSE 0 END), 0),
       COALESCE(AVG(latency_ms), 0)
FROM ai_request_logs`).Scan(
		&summary.TotalRequests,
		&summary.PromptTokens,
		&summary.CompletionTokens,
		&summary.TotalRTKTokensSaved,
		&summary.CacheHits,
		&summary.TotalPIIRedactions,
		&summary.LocalOnlyBlockedEgress,
		&summary.AvgLatencyMS,
	)
	summary.TotalTokens = summary.PromptTokens + summary.CompletionTokens

	_ = g.db.QueryRowContext(ctx, `SELECT COALESCE(SUM(tokens_saved), 0) FROM ai_response_cache`).Scan(&summary.CacheTokensSaved)

	providers, err := g.ListProviders(ctx)
	if err == nil {
		for _, p := range providers {
			summary.ByProvider = append(summary.ByProvider, ProviderUsage{
				ProviderID: p.ID,
				Name:       p.Name,
				Tier:       p.Tier,
				IsLocal:    p.IsLocal,
				Requests:   p.ReqCount,
				Errors:     p.ErrCount,
				TokensIn:   p.TokensIn,
				TokensOut:  p.TokensOut,
			})
		}
	}

	agentRows, err := g.db.QueryContext(ctx, `
SELECT agent_id,
       COUNT(*),
       COALESCE(SUM(prompt_tokens), 0),
       COALESCE(SUM(completion_tokens), 0),
       COALESCE(SUM(rtk_tokens_saved), 0),
       COALESCE(SUM(redactions_count), 0)
FROM ai_request_logs
GROUP BY agent_id
ORDER BY COUNT(*) DESC, agent_id ASC`)
	if err == nil {
		defer func() { _ = agentRows.Close() }()
		for agentRows.Next() {
			var au AgentUsage
			if agentRows.Scan(&au.AgentID, &au.Requests, &au.PromptTokens, &au.CompletionTokens, &au.RTKTokensSaved, &au.RedactionsCount) == nil {
				summary.ByAgent = append(summary.ByAgent, au)
			}
		}
	}

	recent, err := g.listRecentRequestLogs(ctx, 25)
	if err != nil {
		return summary, err
	}
	summary.RecentLogs = recent
	return summary, nil
}

func (g *Gateway) listRecentRequestLogs(ctx context.Context, limit int) ([]RequestLog, error) {
	rows, err := g.db.QueryContext(ctx, `
SELECT id, agent_id, requested_model, resolved_provider, resolved_model, classification,
       status_code, latency_ms, prompt_tokens, completion_tokens, rtk_tokens_saved,
       redactions_count, egress_blocked, cache_hit, created_at
FROM ai_request_logs
ORDER BY created_at DESC, rowid DESC
LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	logs := make([]RequestLog, 0)
	for rows.Next() {
		var logRow RequestLog
		var egressBlockedInt, cacheInt int
		if err := rows.Scan(
			&logRow.ID, &logRow.AgentID, &logRow.RequestedModel, &logRow.ResolvedProvider,
			&logRow.ResolvedModel, &logRow.Classification, &logRow.StatusCode, &logRow.LatencyMS,
			&logRow.PromptTokens, &logRow.CompletionTokens, &logRow.RTKTokensSaved,
			&logRow.RedactionsCount, &egressBlockedInt, &cacheInt, &logRow.CreatedAt,
		); err != nil {
			return nil, err
		}
		logRow.EgressBlocked = egressBlockedInt == 1
		logRow.CacheHit = cacheInt == 1
		logs = append(logs, logRow)
	}
	return logs, rows.Err()
}

// DiscoverLocalProviders runs loopback-only local AI runtime discovery.
func (g *Gateway) DiscoverLocalProviders(ctx context.Context) localai.DiscoveryResult {
	if g.localDiscovery != nil {
		return g.localDiscovery.Discover(ctx)
	}
	return localai.Discover(ctx, localai.DefaultTargets())
}

// --- Core Routing Engine (DataClassification + PII Sniffer + RTK + Cache + Failover) ---

func (g *Gateway) resolveClassification(headerVal, bodyVal string) (DataClassification, error) {
	parsedHeader, err := ParseDataClassification(headerVal)
	if err != nil {
		return "", err
	}
	parsedBody, err := ParseDataClassification(bodyVal)
	if err != nil {
		return "", err
	}

	// Host-level local_only air-gap cannot be downgraded by per-request input.
	if g.defaultClassification == ClassificationLocalOnly {
		return ClassificationLocalOnly, nil
	}
	// Strictest per-request classification wins.
	if parsedHeader == ClassificationLocalOnly || parsedBody == ClassificationLocalOnly {
		return ClassificationLocalOnly, nil
	}
	if parsedHeader != "" {
		return parsedHeader, nil
	}
	if parsedBody != "" {
		return parsedBody, nil
	}
	return g.defaultClassification, nil
}

func (g *Gateway) resolveRoutePlan(ctx context.Context, explicitAgentHeader, defaultAgentID, requestedModel string) (string, string, string, bool) {
	settings := g.GetSettings(ctx)

	agentID := strings.TrimSpace(strings.ToLower(explicitAgentHeader))
	viaAgentPrefix := false
	if strings.HasPrefix(requestedModel, "agent/") {
		agentID = strings.TrimSpace(strings.ToLower(strings.TrimPrefix(requestedModel, "agent/")))
		viaAgentPrefix = true
	}
	if agentID == "" {
		agentID = defaultAgentID
	}
	if agentID == "" {
		agentID = "default"
	}

	route, err := g.GetAgentRoute(ctx, agentID)
	if err != nil {
		route, err = g.GetAgentRoute(ctx, "default")
		if err != nil {
			route = AgentRoute{
				AgentID:    agentID,
				TierOrder:  settings.FallbackOrder,
				RTKEnabled: true,
			}
		}
	}

	effectiveModel := strings.TrimSpace(requestedModel)
	if viaAgentPrefix {
		effectiveModel = strings.TrimSpace(route.TargetModel)
	} else if route.TargetModel != "" {
		shouldOverride := agentID != "default" || effectiveModel == "" || strings.EqualFold(effectiveModel, "auto")
		if shouldOverride {
			effectiveModel = strings.TrimSpace(route.TargetModel)
		}
	}

	tierOrder := strings.TrimSpace(route.TierOrder)
	if tierOrder == "" {
		tierOrder = settings.FallbackOrder
	}
	rtkActive := settings.RTKEnabled && route.RTKEnabled
	return agentID, effectiveModel, tierOrder, rtkActive
}

func (g *Gateway) prepareOutboundMessages(
	classification DataClassification,
	rtkActive bool,
	rtkMode string,
	messages []ChatMessage,
) ([]ChatMessage, int, int) {
	outbound := messages
	redactions := 0
	rtkSaved := 0

	// cloud_eligible ALWAYS applies PII/secret sniffer + RTK compression before egress.
	if classification == ClassificationCloudEligible {
		var report PIISniffReport
		outbound, report = SniffAndRedactMessages(outbound)
		redactions = report.RedactedCount
	}

	if strings.EqualFold(strings.TrimSpace(rtkMode), "caveman") {
		outbound, rtkSaved = CompressCaveman(outbound)
	} else if rtkActive {
		outbound, rtkSaved = CompressMessages(outbound)
	}

	return outbound, rtkSaved, redactions
}

func (g *Gateway) routeCompletion(
	ctx context.Context,
	explicitAgentHeader, defaultAgentID, rtkMode string,
	classification DataClassification,
	incomingReq ChatCompletionRequest,
) (routeOutcome, error) {
	startedAt := time.Now()
	requestedModel := strings.TrimSpace(incomingReq.Model)
	settings := g.GetSettings(ctx)

	agentID, effectiveModel, tierOrder, rtkActive := g.resolveRoutePlan(ctx, explicitAgentHeader, defaultAgentID, requestedModel)

	outboundReq := incomingReq
	outboundReq.Model = effectiveModel
	outboundReq.DataClassification = ""

	preparedMsgs, rtkTokensSaved, redactionsCount := g.prepareOutboundMessages(classification, rtkActive, rtkMode, outboundReq.Messages)
	outboundReq.Messages = preparedMsgs

	// Exact SQLite response cache check for deterministic non-streaming requests.
	cacheEligible := settings.CacheEnabled && !outboundReq.Stream && (outboundReq.Temperature == nil || *outboundReq.Temperature <= 0.2)
	cacheKey := ""
	if cacheEligible {
		cacheKey = computePromptCacheKey(agentID, classification, outboundReq)
		if cachedBody, cachedResp, hit := g.lookupResponseCache(ctx, cacheKey); hit {
			latencyMS := time.Since(startedAt).Milliseconds()
			g.recordRequestLog(ctx, RequestLog{
				AgentID:          agentID,
				RequestedModel:   requestedModel,
				ResolvedProvider: "cache",
				ResolvedModel:    cachedResp.Model,
				Classification:   string(classification),
				StatusCode:       http.StatusOK,
				LatencyMS:        latencyMS,
				PromptTokens:     cachedResp.Usage.PromptTokens,
				CompletionTokens: cachedResp.Usage.CompletionTokens,
				RTKTokensSaved:   rtkTokensSaved,
				RedactionsCount:  redactionsCount,
				CacheHit:         true,
			})
			return routeOutcome{
				Response:         cachedResp,
				RawBody:          cachedBody,
				ResolvedProvider: "cache",
				ResolvedModel:    cachedResp.Model,
				AgentID:          agentID,
				Classification:   classification,
				CacheHit:         true,
				RTKTokensSaved:   rtkTokensSaved,
				RedactionsCount:  redactionsCount,
				LatencyMS:        latencyMS,
			}, nil
		}
	}

	candidates, err := g.selectCandidateProviders(ctx, effectiveModel, tierOrder, classification)
	if err != nil {
		return routeOutcome{}, err
	}
	if len(candidates) == 0 {
		latencyMS := time.Since(startedAt).Milliseconds()
		if classification == ClassificationLocalOnly {
			g.recordRequestLog(ctx, RequestLog{
				AgentID:         agentID,
				RequestedModel:  requestedModel,
				Classification:  string(classification),
				StatusCode:      http.StatusForbidden,
				LatencyMS:       latencyMS,
				RTKTokensSaved:  rtkTokensSaved,
				RedactionsCount: redactionsCount,
				EgressBlocked:   true,
			})
			return routeOutcome{}, &upstreamCallError{
				StatusCode: http.StatusForbidden,
				Code:       "local_only_egress_blocked",
				Body:       "data classification local_only strictly blocks external egress; no healthy loopback local provider available",
			}
		}

		g.recordRequestLog(ctx, RequestLog{
			AgentID:         agentID,
			RequestedModel:  requestedModel,
			Classification:  string(classification),
			StatusCode:      http.StatusServiceUnavailable,
			LatencyMS:       latencyMS,
			RTKTokensSaved:  rtkTokensSaved,
			RedactionsCount: redactionsCount,
		})
		return routeOutcome{}, &upstreamCallError{
			StatusCode: http.StatusServiceUnavailable,
			Code:       "no_healthy_providers",
			Body:       "no healthy AI providers available (configure a provider or wait for cooldown expiry)",
		}
	}

	var lastCallErr error
	for _, candidate := range candidates {
		// Defense-in-depth air-gap assertion immediately prior to every upstream attempt.
		if classification == ClassificationLocalOnly {
			if _, loopErr := localai.ValidateLoopbackEndpointURL(candidate.BaseURL); loopErr != nil {
				g.recordRequestLog(ctx, RequestLog{
					AgentID:         agentID,
					RequestedModel:  requestedModel,
					Classification:  string(classification),
					StatusCode:      http.StatusForbidden,
					LatencyMS:       time.Since(startedAt).Milliseconds(),
					RTKTokensSaved:  rtkTokensSaved,
					RedactionsCount: redactionsCount,
					EgressBlocked:   true,
				})
				return routeOutcome{}, &upstreamCallError{
					StatusCode: http.StatusForbidden,
					Code:       "local_only_egress_blocked",
					Body:       fmt.Sprintf("blocked non-loopback egress to %s: %v", candidate.Name, loopErr),
				}
			}
		}

		resolvedModel := pickProviderModel(candidate, effectiveModel)
		attemptReq := outboundReq
		attemptReq.Model = resolvedModel

		apiKeyPlain := ""
		if candidate.APIKeyEnc != "" {
			plainBytes, decErr := g.box.Decrypt(candidate.APIKeyEnc)
			if decErr != nil {
				lastCallErr = fmt.Errorf("decrypt key for provider %s: %w", candidate.Name, decErr)
				g.recordProviderError(ctx, candidate.ID, false)
				continue
			}
			apiKeyPlain = string(plainBytes)
		}

		client := g.httpClientForClassification(classification)
		rawRespBytes, parsedResp, callErr := g.dispatchProvider(ctx, client, candidate, apiKeyPlain, attemptReq)
		if callErr != nil {
			lastCallErr = callErr
			var upErr *upstreamCallError
			if errors.As(callErr, &upErr) {
				g.recordProviderError(ctx, candidate.ID, upErr.Cooldown)
				if upErr.Retryable {
					continue
				}
				latencyMS := time.Since(startedAt).Milliseconds()
				g.recordRequestLog(ctx, RequestLog{
					AgentID:          agentID,
					RequestedModel:   requestedModel,
					ResolvedProvider: candidate.Name,
					ResolvedModel:    resolvedModel,
					Classification:   string(classification),
					StatusCode:       upErr.StatusCode,
					LatencyMS:        latencyMS,
					RTKTokensSaved:   rtkTokensSaved,
					RedactionsCount:  redactionsCount,
				})
				return routeOutcome{}, upErr
			}
			g.recordProviderError(ctx, candidate.ID, false)
			continue
		}

		if parsedResp.Model == "" {
			parsedResp.Model = resolvedModel
		}
		if len(rawRespBytes) == 0 {
			rawRespBytes, _ = json.Marshal(parsedResp)
		}

		latencyMS := time.Since(startedAt).Milliseconds()
		g.recordProviderSuccess(ctx, candidate.ID, parsedResp.Usage.PromptTokens, parsedResp.Usage.CompletionTokens)
		g.recordRequestLog(ctx, RequestLog{
			AgentID:          agentID,
			RequestedModel:   requestedModel,
			ResolvedProvider: candidate.Name,
			ResolvedModel:    parsedResp.Model,
			Classification:   string(classification),
			StatusCode:       http.StatusOK,
			LatencyMS:        latencyMS,
			PromptTokens:     parsedResp.Usage.PromptTokens,
			CompletionTokens: parsedResp.Usage.CompletionTokens,
			RTKTokensSaved:   rtkTokensSaved,
			RedactionsCount:  redactionsCount,
			CacheHit:         false,
		})

		if cacheEligible && cacheKey != "" {
			g.storeResponseCache(ctx, cacheKey, parsedResp.Model, rawRespBytes)
		}

		return routeOutcome{
			Response:         parsedResp,
			RawBody:          rawRespBytes,
			ResolvedProvider: candidate.Name,
			ResolvedModel:    parsedResp.Model,
			AgentID:          agentID,
			Classification:   classification,
			CacheHit:         false,
			RTKTokensSaved:   rtkTokensSaved,
			RedactionsCount:  redactionsCount,
			LatencyMS:        latencyMS,
		}, nil
	}

	latencyMS := time.Since(startedAt).Milliseconds()
	statusCode := http.StatusBadGateway
	var upErr *upstreamCallError
	if errors.As(lastCallErr, &upErr) && upErr.StatusCode > 0 {
		statusCode = upErr.StatusCode
	}
	g.recordRequestLog(ctx, RequestLog{
		AgentID:         agentID,
		RequestedModel:  requestedModel,
		Classification:  string(classification),
		StatusCode:      statusCode,
		LatencyMS:       latencyMS,
		RTKTokensSaved:  rtkTokensSaved,
		RedactionsCount: redactionsCount,
	})
	if lastCallErr == nil {
		lastCallErr = &upstreamCallError{StatusCode: http.StatusBadGateway, Body: "all upstream providers failed"}
	}
	return routeOutcome{}, lastCallErr
}

func (g *Gateway) httpClientForClassification(classification DataClassification) *http.Client {
	if classification == ClassificationLocalOnly {
		return g.loopbackClient
	}
	return g.httpClient
}

func (g *Gateway) selectCandidateProviders(
	ctx context.Context,
	effectiveModel, tierOrder string,
	classification DataClassification,
) ([]storedProvider, error) {
	enabledProviders, err := g.loadStoredProviders(ctx, true)
	if err != nil {
		return nil, err
	}

	tierRank := make(map[string]int)
	for idx, tierName := range strings.Split(tierOrder, ",") {
		cleanTier := strings.TrimSpace(strings.ToLower(tierName))
		if cleanTier != "" {
			tierRank[cleanTier] = idx
		}
	}
	getTierRank := func(tier string) int {
		if rank, ok := tierRank[strings.ToLower(tier)]; ok {
			return rank
		}
		return len(tierRank) + 10
	}

	healthy := make([]storedProvider, 0, len(enabledProviders))
	for _, provider := range enabledProviders {
		if provider.InCooldown {
			continue
		}
		if classification == ClassificationLocalOnly && !localai.IsLoopbackEndpointURL(provider.BaseURL) {
			continue
		}
		healthy = append(healthy, provider)
	}

	// If local_only has no explicitly registered loopback DB providers and a
	// local discoverer is configured, synthesize candidates from live loopback runtimes.
	if classification == ClassificationLocalOnly && len(healthy) == 0 && g.localDiscovery != nil {
		discovered := g.localDiscovery.Discover(ctx)
		for _, status := range discovered.Providers {
			if !status.Online || !localai.IsLoopbackEndpointURL(status.OpenAIBaseURL) {
				continue
			}
			modelIDs := make([]string, 0, len(status.Models))
			for _, m := range status.Models {
				modelIDs = append(modelIDs, m.ID)
			}
			defaultModel := ""
			if len(modelIDs) > 0 {
				defaultModel = modelIDs[0]
			}
			healthy = append(healthy, storedProvider{
				Provider: Provider{
					ID:           "local_" + status.ID,
					Name:         "local-" + status.ID,
					Kind:         KindOpenAICompat,
					Tier:         TierFree,
					BaseURL:      status.OpenAIBaseURL,
					IsLocal:      true,
					DefaultModel: defaultModel,
					Models:       modelIDs,
					Priority:     1000,
					Enabled:      true,
				},
			})
		}
	}

	modelSpecific := effectiveModel != "" && !strings.EqualFold(effectiveModel, "auto")
	sort.SliceStable(healthy, func(i, j int) bool {
		leftMatch := !modelSpecific || providerMatchesModel(healthy[i], effectiveModel)
		rightMatch := !modelSpecific || providerMatchesModel(healthy[j], effectiveModel)
		if leftMatch != rightMatch {
			return leftMatch
		}
		leftTier := getTierRank(healthy[i].Tier)
		rightTier := getTierRank(healthy[j].Tier)
		if leftTier != rightTier {
			return leftTier < rightTier
		}
		if healthy[i].Priority != healthy[j].Priority {
			return healthy[i].Priority < healthy[j].Priority
		}
		return healthy[i].CreatedAt < healthy[j].CreatedAt
	})

	return healthy, nil
}

func providerMatchesModel(provider storedProvider, targetModel string) bool {
	if targetModel == "" || strings.EqualFold(targetModel, "auto") {
		return true
	}
	if strings.EqualFold(provider.DefaultModel, targetModel) {
		return true
	}
	if len(provider.Models) == 0 {
		return true
	}
	for _, declaredModel := range provider.Models {
		if strings.EqualFold(declaredModel, targetModel) {
			return true
		}
	}
	return false
}

func pickProviderModel(provider storedProvider, requestedModel string) string {
	trimmed := strings.TrimSpace(requestedModel)
	if trimmed == "" || strings.EqualFold(trimmed, "auto") {
		if provider.DefaultModel != "" {
			return provider.DefaultModel
		}
		if len(provider.Models) > 0 {
			return provider.Models[0]
		}
		return "gpt-4o-mini"
	}
	if len(provider.Models) == 0 || providerMatchesModel(provider, trimmed) {
		return trimmed
	}
	if provider.DefaultModel != "" {
		return provider.DefaultModel
	}
	return trimmed
}

func (g *Gateway) dispatchProvider(
	ctx context.Context,
	client *http.Client,
	provider storedProvider,
	apiKey string,
	req ChatCompletionRequest,
) ([]byte, ChatCompletionResponse, error) {
	switch provider.Kind {
	case KindAnthropic:
		return g.dispatchAnthropicUpstream(ctx, client, provider, apiKey, req)
	case KindGemini:
		if strings.Contains(strings.ToLower(provider.BaseURL), "/openai") {
			return g.dispatchOpenAICompatUpstream(ctx, client, provider, apiKey, req)
		}
		return g.dispatchGeminiNativeUpstream(ctx, client, provider, apiKey, req)
	default:
		return g.dispatchOpenAICompatUpstream(ctx, client, provider, apiKey, req)
	}
}

func (g *Gateway) dispatchOpenAICompatUpstream(
	ctx context.Context,
	client *http.Client,
	provider storedProvider,
	apiKey string,
	req ChatCompletionRequest,
) ([]byte, ChatCompletionResponse, error) {
	endpoint := strings.TrimRight(provider.BaseURL, "/")
	if !strings.HasSuffix(endpoint, "/chat/completions") {
		endpoint += "/chat/completions"
	}

	reqBodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, ChatCompletionResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBodyBytes))
	if err != nil {
		return nil, ChatCompletionResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}

	httpResp, err := client.Do(httpReq)
	if err != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{
			StatusCode: http.StatusBadGateway,
			Body:       err.Error(),
			Retryable:  true,
		}
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBytes, readErr := io.ReadAll(io.LimitReader(httpResp.Body, maxUpstreamResponseBytes))
	if readErr != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{
			StatusCode: http.StatusBadGateway,
			Body:       readErr.Error(),
			Retryable:  true,
		}
	}

	if httpResp.StatusCode == http.StatusTooManyRequests {
		return nil, ChatCompletionResponse{}, &upstreamCallError{
			StatusCode: httpResp.StatusCode,
			Body:       string(respBytes),
			Retryable:  true,
			Cooldown:   true,
		}
	}
	if httpResp.StatusCode >= 500 || httpResp.StatusCode == http.StatusUnauthorized || httpResp.StatusCode == http.StatusForbidden {
		return nil, ChatCompletionResponse{}, &upstreamCallError{
			StatusCode: httpResp.StatusCode,
			Body:       string(respBytes),
			Retryable:  true,
		}
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, ChatCompletionResponse{}, &upstreamCallError{
			StatusCode: httpResp.StatusCode,
			Body:       string(respBytes),
			Retryable:  false,
		}
	}

	var parsed ChatCompletionResponse
	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{
			StatusCode: http.StatusBadGateway,
			Body:       "invalid JSON from upstream provider: " + err.Error(),
			Retryable:  true,
		}
	}
	return respBytes, parsed, nil
}

func (g *Gateway) dispatchAnthropicUpstream(
	ctx context.Context,
	client *http.Client,
	provider storedProvider,
	apiKey string,
	req ChatCompletionRequest,
) ([]byte, ChatCompletionResponse, error) {
	endpoint := strings.TrimRight(provider.BaseURL, "/")
	if !strings.HasSuffix(endpoint, "/messages") {
		if strings.HasSuffix(endpoint, "/v1") {
			endpoint += "/messages"
		} else {
			endpoint += "/v1/messages"
		}
	}

	anthReq := openAIToAnthropicRequest(req)
	reqBodyBytes, err := json.Marshal(anthReq)
	if err != nil {
		return nil, ChatCompletionResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBodyBytes))
	if err != nil {
		return nil, ChatCompletionResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	if apiKey != "" {
		httpReq.Header.Set("x-api-key", apiKey)
	}

	httpResp, err := client.Do(httpReq)
	if err != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: http.StatusBadGateway, Body: err.Error(), Retryable: true}
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBytes, readErr := io.ReadAll(io.LimitReader(httpResp.Body, maxUpstreamResponseBytes))
	if readErr != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: http.StatusBadGateway, Body: readErr.Error(), Retryable: true}
	}
	if httpResp.StatusCode == http.StatusTooManyRequests {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(respBytes), Retryable: true, Cooldown: true}
	}
	if httpResp.StatusCode >= 500 || httpResp.StatusCode == http.StatusUnauthorized || httpResp.StatusCode == http.StatusForbidden {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(respBytes), Retryable: true}
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(respBytes), Retryable: false}
	}

	var anthResp AnthropicMessageResponse
	if err := json.Unmarshal(respBytes, &anthResp); err != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: http.StatusBadGateway, Body: err.Error(), Retryable: true}
	}
	converted := anthropicToOpenAIResponse(anthResp, req.Model)
	encoded, _ := json.Marshal(converted)
	return encoded, converted, nil
}

func (g *Gateway) dispatchGeminiNativeUpstream(
	ctx context.Context,
	client *http.Client,
	provider storedProvider,
	apiKey string,
	req ChatCompletionRequest,
) ([]byte, ChatCompletionResponse, error) {
	endpoint := fmt.Sprintf("%s/models/%s:generateContent", strings.TrimRight(provider.BaseURL, "/"), req.Model)
	if apiKey != "" {
		endpoint += "?key=" + apiKey
	}

	type geminiPart struct {
		Text string `json:"text"`
	}
	type geminiContent struct {
		Role  string       `json:"role"`
		Parts []geminiPart `json:"parts"`
	}
	contents := make([]geminiContent, 0, len(req.Messages))
	for _, msg := range req.Messages {
		role := "user"
		if strings.EqualFold(msg.Role, "assistant") {
			role = "model"
		}
		textVal := flattenMessageContentToText(msg.Content)
		if textVal != "" {
			contents = append(contents, geminiContent{
				Role:  role,
				Parts: []geminiPart{{Text: textVal}},
			})
		}
	}

	payloadMap := map[string]any{"contents": contents}
	reqBodyBytes, _ := json.Marshal(payloadMap)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBodyBytes))
	if err != nil {
		return nil, ChatCompletionResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := client.Do(httpReq)
	if err != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: http.StatusBadGateway, Body: err.Error(), Retryable: true}
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBytes, readErr := io.ReadAll(io.LimitReader(httpResp.Body, maxUpstreamResponseBytes))
	if readErr != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: http.StatusBadGateway, Body: readErr.Error(), Retryable: true}
	}
	if httpResp.StatusCode == http.StatusTooManyRequests {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(respBytes), Retryable: true, Cooldown: true}
	}
	if httpResp.StatusCode >= 500 || httpResp.StatusCode == http.StatusUnauthorized || httpResp.StatusCode == http.StatusForbidden {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(respBytes), Retryable: true}
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(respBytes), Retryable: false}
	}

	var geminiEnvelope struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.Unmarshal(respBytes, &geminiEnvelope); err != nil {
		return nil, ChatCompletionResponse{}, &upstreamCallError{StatusCode: http.StatusBadGateway, Body: err.Error(), Retryable: true}
	}

	var replyBuilder strings.Builder
	if len(geminiEnvelope.Candidates) > 0 {
		for _, part := range geminiEnvelope.Candidates[0].Content.Parts {
			replyBuilder.WriteString(part.Text)
		}
	}

	converted := ChatCompletionResponse{
		ID:      makeGatewayID("chatcmpl", req.Model, time.Now().UTC().Format(time.RFC3339Nano)),
		Object:  "chat.completion",
		Created: time.Now().UTC().Unix(),
		Model:   req.Model,
		Choices: []ChatCompletionChoice{
			{
				Index:        0,
				Message:      ChatMessage{Role: "assistant", Content: replyBuilder.String()},
				FinishReason: "stop",
			},
		},
		Usage: ChatCompletionUsage{
			PromptTokens:     geminiEnvelope.UsageMetadata.PromptTokenCount,
			CompletionTokens: geminiEnvelope.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      geminiEnvelope.UsageMetadata.TotalTokenCount,
		},
	}
	encoded, _ := json.Marshal(converted)
	return encoded, converted, nil
}

// --- Exact Prompt Cache & Telemetry Helpers ---

func computePromptCacheKey(agentID string, classification DataClassification, req ChatCompletionRequest) string {
	canonical, _ := json.Marshal(struct {
		AgentID        string             `json:"agentId"`
		Classification DataClassification `json:"classification"`
		Model          string             `json:"model"`
		Messages       []ChatMessage      `json:"messages"`
		Temperature    *float64           `json:"temperature,omitempty"`
		MaxTokens      *int               `json:"max_tokens,omitempty"`
		Tools          json.RawMessage    `json:"tools,omitempty"`
		ToolChoice     json.RawMessage    `json:"tool_choice,omitempty"`
		Stop           json.RawMessage    `json:"stop,omitempty"`
	}{
		AgentID:        agentID,
		Classification: classification,
		Model:          req.Model,
		Messages:       req.Messages,
		Temperature:    req.Temperature,
		MaxTokens:      req.MaxTokens,
		Tools:          req.Tools,
		ToolChoice:     req.ToolChoice,
		Stop:           req.Stop,
	})
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])
}

func (g *Gateway) lookupResponseCache(ctx context.Context, cacheKey string) ([]byte, ChatCompletionResponse, bool) {
	var rawJSON string
	err := g.db.QueryRowContext(ctx, `SELECT response_json FROM ai_response_cache WHERE cache_key = ?`, cacheKey).Scan(&rawJSON)
	if err != nil || rawJSON == "" {
		return nil, ChatCompletionResponse{}, false
	}
	var parsed ChatCompletionResponse
	if unmarshalErr := json.Unmarshal([]byte(rawJSON), &parsed); unmarshalErr != nil {
		return nil, ChatCompletionResponse{}, false
	}

	savedTokens := parsed.Usage.TotalTokens
	if savedTokens <= 0 {
		savedTokens = parsed.Usage.PromptTokens + parsed.Usage.CompletionTokens
	}
	if savedTokens <= 0 {
		savedTokens = len(rawJSON) / 4
	}
	_, _ = g.db.ExecContext(ctx, `
UPDATE ai_response_cache SET hits = hits + 1, tokens_saved = tokens_saved + ? WHERE cache_key = ?`,
		savedTokens, cacheKey,
	)
	return []byte(rawJSON), parsed, true
}

func (g *Gateway) storeResponseCache(ctx context.Context, cacheKey, model string, rawResp []byte) {
	now := time.Now().UTC().Format(time.RFC3339)
	_, _ = g.db.ExecContext(ctx, `
INSERT INTO ai_response_cache (cache_key, model, response_json, tokens_saved, hits, created_at)
VALUES (?, ?, ?, 0, 0, ?)
ON CONFLICT(cache_key) DO UPDATE SET
  model = excluded.model,
  response_json = excluded.response_json,
  created_at = excluded.created_at`,
		cacheKey, model, string(rawResp), now,
	)
}

func (g *Gateway) recordProviderError(ctx context.Context, providerID string, setCooldown bool) {
	if strings.HasPrefix(providerID, "local_") {
		return
	}
	if setCooldown {
		until := time.Now().UTC().Add(defaultCooldownDuration).Format(time.RFC3339)
		_, _ = g.db.ExecContext(ctx, `
UPDATE ai_providers SET err_count = err_count + 1, cooldown_until = ? WHERE id = ?`, until, providerID)
		return
	}
	_, _ = g.db.ExecContext(ctx, `UPDATE ai_providers SET err_count = err_count + 1 WHERE id = ?`, providerID)
}

func (g *Gateway) recordProviderSuccess(ctx context.Context, providerID string, promptTokens, completionTokens int) {
	if strings.HasPrefix(providerID, "local_") {
		return
	}
	_, _ = g.db.ExecContext(ctx, `
UPDATE ai_providers
SET req_count = req_count + 1,
    tokens_in = tokens_in + ?,
    tokens_out = tokens_out + ?,
    cooldown_until = ''
WHERE id = ?`, promptTokens, completionTokens, providerID)
}

func (g *Gateway) recordRequestLog(ctx context.Context, entry RequestLog) {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	logID := makeGatewayID("log", entry.AgentID, entry.RequestedModel, now)
	classLabel := entry.Classification
	if classLabel == "" {
		classLabel = string(ClassificationCloudEligible)
	}
	_, _ = g.db.ExecContext(ctx, `
INSERT INTO ai_request_logs (
  id, agent_id, requested_model, resolved_provider, resolved_model, classification,
  status_code, latency_ms, prompt_tokens, completion_tokens, rtk_tokens_saved,
  redactions_count, egress_blocked, cache_hit, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		logID, entry.AgentID, entry.RequestedModel, entry.ResolvedProvider, entry.ResolvedModel, classLabel,
		entry.StatusCode, entry.LatencyMS, entry.PromptTokens, entry.CompletionTokens,
		entry.RTKTokensSaved, entry.RedactionsCount, boolToInt(entry.EgressBlocked),
		boolToInt(entry.CacheHit), time.Now().UTC().Format(time.RFC3339),
	)
}

// --- Anthropic <-> OpenAI Protocol Translation ---

func anthropicToOpenAIRequest(anthReq AnthropicMessageRequest) ChatCompletionRequest {
	messages := make([]ChatMessage, 0, len(anthReq.Messages)+1)

	if systemText := extractAnthropicSystemText(anthReq.System); systemText != "" {
		messages = append(messages, ChatMessage{Role: "system", Content: systemText})
	}

	for _, msg := range anthReq.Messages {
		switch contentVal := msg.Content.(type) {
		case string:
			messages = append(messages, ChatMessage{Role: msg.Role, Content: contentVal})
		case []any:
			var textParts []string
			for _, rawBlock := range contentVal {
				blockMap, ok := rawBlock.(map[string]any)
				if !ok {
					continue
				}
				blockType, _ := blockMap["type"].(string)
				switch blockType {
				case "tool_result":
					toolUseID, _ := blockMap["tool_use_id"].(string)
					toolText := flattenMessageContentToText(blockMap["content"])
					messages = append(messages, ChatMessage{
						Role:       "tool",
						ToolCallID: toolUseID,
						Content:    toolText,
					})
				case "text":
					if textStr, isStr := blockMap["text"].(string); isStr && textStr != "" {
						textParts = append(textParts, textStr)
					}
				case "tool_use":
					toolName, _ := blockMap["name"].(string)
					inputBytes, _ := json.Marshal(blockMap["input"])
					textParts = append(textParts, fmt.Sprintf("[tool_use %s: %s]", toolName, string(inputBytes)))
				}
			}
			if len(textParts) > 0 {
				messages = append(messages, ChatMessage{
					Role:    msg.Role,
					Content: strings.Join(textParts, "\n"),
				})
			}
		default:
			if flattened := flattenMessageContentToText(contentVal); flattened != "" {
				messages = append(messages, ChatMessage{Role: msg.Role, Content: flattened})
			}
		}
	}

	openAIReq := ChatCompletionRequest{
		Model:              anthReq.Model,
		Messages:           messages,
		Temperature:        anthReq.Temperature,
		TopP:               anthReq.TopP,
		Stream:             anthReq.Stream,
		DataClassification: anthReq.DataClassification,
	}
	if anthReq.MaxTokens > 0 {
		maxTok := anthReq.MaxTokens
		openAIReq.MaxTokens = &maxTok
	}
	if len(anthReq.StopSequences) > 0 {
		stopRaw, _ := json.Marshal(anthReq.StopSequences)
		openAIReq.Stop = stopRaw
	}
	return openAIReq
}

func openAIToAnthropicRequest(openAIReq ChatCompletionRequest) AnthropicMessageRequest {
	var systemParts []string
	anthMsgs := make([]AnthropicMessage, 0, len(openAIReq.Messages))

	for _, msg := range openAIReq.Messages {
		textVal := flattenMessageContentToText(msg.Content)
		switch strings.ToLower(msg.Role) {
		case "system":
			if textVal != "" {
				systemParts = append(systemParts, textVal)
			}
		case "tool":
			anthMsgs = append(anthMsgs, AnthropicMessage{
				Role: "user",
				Content: []map[string]any{
					{
						"type":        "tool_result",
						"tool_use_id": msg.ToolCallID,
						"content":     textVal,
					},
				},
			})
		case "assistant":
			anthMsgs = append(anthMsgs, AnthropicMessage{Role: "assistant", Content: textVal})
		default:
			anthMsgs = append(anthMsgs, AnthropicMessage{Role: "user", Content: textVal})
		}
	}

	maxTok := 4096
	if openAIReq.MaxTokens != nil && *openAIReq.MaxTokens > 0 {
		maxTok = *openAIReq.MaxTokens
	}
	anthReq := AnthropicMessageRequest{
		Model:       openAIReq.Model,
		Messages:    anthMsgs,
		MaxTokens:   maxTok,
		Temperature: openAIReq.Temperature,
		TopP:        openAIReq.TopP,
	}
	if len(systemParts) > 0 {
		anthReq.System = strings.Join(systemParts, "\n")
	}
	return anthReq
}

func openAIToAnthropicResponse(openAIResp ChatCompletionResponse) AnthropicMessageResponse {
	assistantText := ""
	stopReason := "end_turn"
	if len(openAIResp.Choices) > 0 {
		choice := openAIResp.Choices[0]
		assistantText = flattenMessageContentToText(choice.Message.Content)
		switch choice.FinishReason {
		case "", "stop":
			stopReason = "end_turn"
		case "length":
			stopReason = "max_tokens"
		case "tool_calls":
			stopReason = "tool_use"
		default:
			stopReason = choice.FinishReason
		}
	}

	msgID := openAIResp.ID
	if msgID == "" {
		msgID = makeGatewayID("msg", openAIResp.Model, time.Now().UTC().Format(time.RFC3339Nano))
	} else if !strings.HasPrefix(msgID, "msg_") {
		msgID = "msg_" + strings.TrimPrefix(msgID, "chatcmpl-")
	}

	return AnthropicMessageResponse{
		ID:         msgID,
		Type:       "message",
		Role:       "assistant",
		Model:      openAIResp.Model,
		Content:    []AnthropicContentBlock{{Type: "text", Text: assistantText}},
		StopReason: stopReason,
		Usage: AnthropicUsage{
			InputTokens:  openAIResp.Usage.PromptTokens,
			OutputTokens: openAIResp.Usage.CompletionTokens,
		},
	}
}

func anthropicToOpenAIResponse(anthResp AnthropicMessageResponse, fallbackModel string) ChatCompletionResponse {
	var textParts []string
	for _, block := range anthResp.Content {
		if block.Text != "" {
			textParts = append(textParts, block.Text)
		}
	}
	modelName := anthResp.Model
	if modelName == "" {
		modelName = fallbackModel
	}
	finishReason := "stop"
	switch anthResp.StopReason {
	case "max_tokens":
		finishReason = "length"
	case "tool_use":
		finishReason = "tool_calls"
	}

	return ChatCompletionResponse{
		ID:      "chatcmpl-" + strings.TrimPrefix(anthResp.ID, "msg_"),
		Object:  "chat.completion",
		Created: time.Now().UTC().Unix(),
		Model:   modelName,
		Choices: []ChatCompletionChoice{
			{
				Index:        0,
				Message:      ChatMessage{Role: "assistant", Content: strings.Join(textParts, "\n")},
				FinishReason: finishReason,
			},
		},
		Usage: ChatCompletionUsage{
			PromptTokens:     anthResp.Usage.InputTokens,
			CompletionTokens: anthResp.Usage.OutputTokens,
			TotalTokens:      anthResp.Usage.InputTokens + anthResp.Usage.OutputTokens,
		},
	}
}

func extractAnthropicSystemText(systemField any) string {
	switch typed := systemField.(type) {
	case string:
		return strings.TrimSpace(typed)
	case []any:
		parts := make([]string, 0, len(typed))
		for _, rawBlock := range typed {
			if blockMap, ok := rawBlock.(map[string]any); ok {
				if textStr, isStr := blockMap["text"].(string); isStr && strings.TrimSpace(textStr) != "" {
					parts = append(parts, textStr)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func flattenMessageContentToText(content any) string {
	switch typed := content.(type) {
	case string:
		return typed
	case []any:
		parts := make([]string, 0, len(typed))
		for _, rawBlock := range typed {
			switch blockVal := rawBlock.(type) {
			case string:
				parts = append(parts, blockVal)
			case map[string]any:
				if textStr, ok := blockVal["text"].(string); ok && textStr != "" {
					parts = append(parts, textStr)
				} else if nestedContent, exists := blockVal["content"]; exists {
					if nestedText := flattenMessageContentToText(nestedContent); nestedText != "" {
						parts = append(parts, nestedText)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		if typed == nil {
			return ""
		}
		encoded, _ := json.Marshal(typed)
		return string(encoded)
	}
}

// --- Standard Library HTTP Handlers ---

func writeJSON(w http.ResponseWriter, statusCode int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(payload)
}

func decodeJSONBody(r *http.Request, target any) error {
	if r.Body == nil {
		return errors.New("empty request body")
	}
	defer func() { _ = r.Body.Close() }()
	dec := json.NewDecoder(io.LimitReader(r.Body, maxIncomingRequestBytes))
	return dec.Decode(target)
}

func extractAgentHeader(r *http.Request) string {
	if val := strings.TrimSpace(r.Header.Get(headerRocksoulAgentID)); val != "" {
		return val
	}
	return strings.TrimSpace(r.Header.Get(headerAgentID))
}

func resolveRTKMode(r *http.Request) string {
	if mode := strings.TrimSpace(r.Header.Get(headerRocksoulRTKMode)); mode != "" {
		return strings.ToLower(mode)
	}
	if mode := strings.TrimSpace(r.Header.Get(headerRTKMode)); mode != "" {
		return strings.ToLower(mode)
	}
	return strings.ToLower(strings.TrimSpace(r.URL.Query().Get("rtk")))
}

func extractClassificationHeader(r *http.Request) string {
	if val := strings.TrimSpace(r.Header.Get(headerClassification)); val != "" {
		return val
	}
	return strings.TrimSpace(r.Header.Get(headerMWClassification))
}

func (g *Gateway) handleListOpenAIModels(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	nowUnix := time.Now().UTC().Unix()
	seen := make(map[string]bool)
	modelList := make([]OpenAIModelObject, 0)

	appendUnique := func(modelID, owner string) {
		cleanID := strings.TrimSpace(modelID)
		if cleanID == "" || seen[cleanID] {
			return
		}
		seen[cleanID] = true
		modelList = append(modelList, OpenAIModelObject{
			ID:      cleanID,
			Object:  "model",
			Created: nowUnix,
			OwnedBy: owner,
		})
	}

	if routes, err := g.ListAgentRoutes(ctx); err == nil {
		for _, route := range routes {
			appendUnique("agent/"+route.AgentID, "rocksoul-agent")
		}
	}

	if providers, err := g.ListProviders(ctx); err == nil {
		for _, provider := range providers {
			if !provider.Enabled {
				continue
			}
			owner := "provider:" + provider.Name
			if provider.DefaultModel != "" {
				appendUnique(provider.DefaultModel, owner)
			}
			for _, declared := range provider.Models {
				appendUnique(declared, owner)
			}
		}
	}

	if g.localDiscovery != nil {
		discovered := g.localDiscovery.Discover(ctx)
		for _, localModel := range discovered.Models {
			appendUnique(localModel.ID, "local:"+localModel.ProviderID)
		}
	}

	if catalogEntries, err := g.ListCatalog(ctx, "", true); err == nil {
		for _, entry := range catalogEntries {
			appendUnique(entry.ModelID, "catalog:"+entry.ProviderSlug)
		}
	}

	writeJSON(w, http.StatusOK, OpenAIModelList{
		Object: "list",
		Models: modelList,
	})
}

func (g *Gateway) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var completionReq ChatCompletionRequest
	if err := decodeJSONBody(r, &completionReq); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]string{"message": "invalid chat completion JSON", "type": "invalid_request_error"},
		})
		return
	}
	if len(completionReq.Messages) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]string{"message": "messages array is required", "type": "invalid_request_error"},
		})
		return
	}

	classification, classErr := g.resolveClassification(extractClassificationHeader(r), completionReq.DataClassification)
	if classErr != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]string{"message": classErr.Error(), "type": "invalid_request_error", "code": "invalid_classification"},
		})
		return
	}

	agentHeader := extractAgentHeader(r)
	rtkMode := resolveRTKMode(r)
	if completionReq.Stream {
		g.handleStreamingChatCompletion(w, r, agentHeader, rtkMode, classification, completionReq)
		return
	}

	outcome, err := g.routeCompletion(r.Context(), agentHeader, "default", rtkMode, classification, completionReq)
	if err != nil {
		writeGatewayError(w, err)
		return
	}

	setGatewayResponseHeaders(w, outcome)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(outcome.RawBody)
}

func (g *Gateway) handleStreamingChatCompletion(
	w http.ResponseWriter,
	r *http.Request,
	agentHeader, rtkMode string,
	classification DataClassification,
	incomingReq ChatCompletionRequest,
) {
	ctx := r.Context()
	startedAt := time.Now()
	requestedModel := strings.TrimSpace(incomingReq.Model)
	agentID, effectiveModel, tierOrder, rtkActive := g.resolveRoutePlan(ctx, agentHeader, "default", requestedModel)

	outboundReq := incomingReq
	outboundReq.Model = effectiveModel
	outboundReq.DataClassification = ""

	preparedMsgs, rtkTokensSaved, redactionsCount := g.prepareOutboundMessages(classification, rtkActive, rtkMode, outboundReq.Messages)
	outboundReq.Messages = preparedMsgs

	candidates, err := g.selectCandidateProviders(ctx, effectiveModel, tierOrder, classification)
	if err != nil {
		writeGatewayError(w, err)
		return
	}
	if len(candidates) == 0 {
		if classification == ClassificationLocalOnly {
			g.recordRequestLog(ctx, RequestLog{
				AgentID:         agentID,
				RequestedModel:  requestedModel,
				Classification:  string(classification),
				StatusCode:      http.StatusForbidden,
				LatencyMS:       time.Since(startedAt).Milliseconds(),
				RTKTokensSaved:  rtkTokensSaved,
				RedactionsCount: redactionsCount,
				EgressBlocked:   true,
			})
			writeGatewayError(w, &upstreamCallError{
				StatusCode: http.StatusForbidden,
				Code:       "local_only_egress_blocked",
				Body:       "data classification local_only strictly blocks external egress; no healthy loopback local provider available",
			})
			return
		}
		writeGatewayError(w, &upstreamCallError{
			StatusCode: http.StatusServiceUnavailable,
			Code:       "no_healthy_providers",
			Body:       "no healthy AI providers available",
		})
		return
	}

	client := g.httpClientForClassification(classification)
	var lastErr error
	for _, candidate := range candidates {
		if classification == ClassificationLocalOnly {
			if _, loopErr := localai.ValidateLoopbackEndpointURL(candidate.BaseURL); loopErr != nil {
				writeGatewayError(w, &upstreamCallError{
					StatusCode: http.StatusForbidden,
					Code:       "local_only_egress_blocked",
					Body:       fmt.Sprintf("blocked non-loopback egress to %s: %v", candidate.Name, loopErr),
				})
				return
			}
		}
		if candidate.Kind != KindOpenAICompat && !(candidate.Kind == KindGemini && strings.Contains(strings.ToLower(candidate.BaseURL), "/openai")) {
			continue
		}
		resolvedModel := pickProviderModel(candidate, effectiveModel)
		attemptReq := outboundReq
		attemptReq.Model = resolvedModel

		apiKeyPlain := ""
		if candidate.APIKeyEnc != "" {
			plainBytes, decErr := g.box.Decrypt(candidate.APIKeyEnc)
			if decErr != nil {
				lastErr = decErr
				g.recordProviderError(ctx, candidate.ID, false)
				continue
			}
			apiKeyPlain = string(plainBytes)
		}

		endpoint := strings.TrimRight(candidate.BaseURL, "/")
		if !strings.HasSuffix(endpoint, "/chat/completions") {
			endpoint += "/chat/completions"
		}
		reqBodyBytes, _ := json.Marshal(attemptReq)
		httpReq, reqErr := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBodyBytes))
		if reqErr != nil {
			lastErr = reqErr
			continue
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if apiKeyPlain != "" {
			httpReq.Header.Set("Authorization", "Bearer "+apiKeyPlain)
		}

		httpResp, doErr := client.Do(httpReq)
		if doErr != nil {
			lastErr = &upstreamCallError{StatusCode: http.StatusBadGateway, Body: doErr.Error(), Retryable: true}
			g.recordProviderError(ctx, candidate.ID, false)
			continue
		}

		if httpResp.StatusCode == http.StatusTooManyRequests {
			bodyBytes, _ := io.ReadAll(io.LimitReader(httpResp.Body, 64*1024))
			_ = httpResp.Body.Close()
			lastErr = &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(bodyBytes), Retryable: true, Cooldown: true}
			g.recordProviderError(ctx, candidate.ID, true)
			continue
		}
		if httpResp.StatusCode >= 500 || httpResp.StatusCode == http.StatusUnauthorized || httpResp.StatusCode == http.StatusForbidden {
			bodyBytes, _ := io.ReadAll(io.LimitReader(httpResp.Body, 64*1024))
			_ = httpResp.Body.Close()
			lastErr = &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(bodyBytes), Retryable: true}
			g.recordProviderError(ctx, candidate.ID, false)
			continue
		}
		if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
			bodyBytes, _ := io.ReadAll(io.LimitReader(httpResp.Body, 64*1024))
			_ = httpResp.Body.Close()
			writeGatewayError(w, &upstreamCallError{StatusCode: httpResp.StatusCode, Body: string(bodyBytes)})
			return
		}

		defer func() { _ = httpResp.Body.Close() }()
		contentType := httpResp.Header.Get("Content-Type")
		if contentType == "" {
			contentType = "text/event-stream"
		}
		w.Header().Set("Content-Type", contentType)
		w.Header().Set(headerCacheState, "BYPASS")
		w.Header().Set(headerEchoCacheState, "BYPASS")
		w.Header().Set(headerProvider, candidate.Name)
		w.Header().Set(headerEchoProvider, candidate.Name)
		w.Header().Set(headerModel, resolvedModel)
		w.Header().Set(headerEchoModel, resolvedModel)
		w.Header().Set(headerAgentID, agentID)
		w.Header().Set(headerRTKSaved, strconv.Itoa(rtkTokensSaved))
		w.Header().Set(headerEchoRTKSaved, strconv.Itoa(rtkTokensSaved))
		w.Header().Set(headerOutClassification, string(classification))
		w.Header().Set(headerOutRedactions, strconv.Itoa(redactionsCount))
		w.WriteHeader(http.StatusOK)

		_, _ = io.Copy(w, httpResp.Body)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}

		g.recordProviderSuccess(ctx, candidate.ID, 0, 0)
		g.recordRequestLog(ctx, RequestLog{
			AgentID:          agentID,
			RequestedModel:   requestedModel,
			ResolvedProvider: candidate.Name,
			ResolvedModel:    resolvedModel,
			Classification:   string(classification),
			StatusCode:       http.StatusOK,
			LatencyMS:        time.Since(startedAt).Milliseconds(),
			RTKTokensSaved:   rtkTokensSaved,
			RedactionsCount:  redactionsCount,
			CacheHit:         false,
		})
		return
	}

	if lastErr == nil {
		lastErr = &upstreamCallError{StatusCode: http.StatusBadGateway, Body: "no streaming-capable provider succeeded"}
	}
	writeGatewayError(w, lastErr)
}

func (g *Gateway) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	var anthReq AnthropicMessageRequest
	if err := decodeJSONBody(r, &anthReq); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"type":  "error",
			"error": map[string]string{"type": "invalid_request_error", "message": "invalid anthropic messages JSON"},
		})
		return
	}
	if len(anthReq.Messages) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"type":  "error",
			"error": map[string]string{"type": "invalid_request_error", "message": "messages array is required"},
		})
		return
	}

	classification, classErr := g.resolveClassification(extractClassificationHeader(r), anthReq.DataClassification)
	if classErr != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"type":  "error",
			"error": map[string]string{"type": "invalid_request_error", "code": "invalid_classification", "message": classErr.Error()},
		})
		return
	}

	openAIReq := anthropicToOpenAIRequest(anthReq)
	agentHeader := extractAgentHeader(r)
	rtkMode := resolveRTKMode(r)
	outcome, err := g.routeCompletion(r.Context(), agentHeader, "claude-code", rtkMode, classification, openAIReq)
	if err != nil {
		statusCode := http.StatusBadGateway
		errCode := "api_error"
		var upErr *upstreamCallError
		if errors.As(err, &upErr) && upErr.StatusCode > 0 {
			statusCode = upErr.StatusCode
			if upErr.Code != "" {
				errCode = upErr.Code
			}
		}
		writeJSON(w, statusCode, map[string]any{
			"type":  "error",
			"error": map[string]string{"type": errCode, "message": err.Error()},
		})
		return
	}

	setGatewayResponseHeaders(w, outcome)
	writeJSON(w, http.StatusOK, openAIToAnthropicResponse(outcome.Response))
}

func setGatewayResponseHeaders(w http.ResponseWriter, outcome routeOutcome) {
	cacheState := "MISS"
	if outcome.CacheHit {
		cacheState = "HIT"
	}
	w.Header().Set(headerCacheState, cacheState)
	w.Header().Set(headerEchoCacheState, cacheState)
	w.Header().Set(headerProvider, outcome.ResolvedProvider)
	w.Header().Set(headerEchoProvider, outcome.ResolvedProvider)
	w.Header().Set(headerModel, outcome.ResolvedModel)
	w.Header().Set(headerEchoModel, outcome.ResolvedModel)
	w.Header().Set(headerAgentID, outcome.AgentID)
	w.Header().Set(headerRTKSaved, strconv.Itoa(outcome.RTKTokensSaved))
	w.Header().Set(headerEchoRTKSaved, strconv.Itoa(outcome.RTKTokensSaved))
	w.Header().Set(headerOutClassification, string(outcome.Classification))
	w.Header().Set(headerOutRedactions, strconv.Itoa(outcome.RedactionsCount))
}

func writeGatewayError(w http.ResponseWriter, err error) {
	statusCode := http.StatusBadGateway
	errCode := "gateway_error"
	var upErr *upstreamCallError
	if errors.As(err, &upErr) {
		if upErr.StatusCode > 0 {
			statusCode = upErr.StatusCode
		}
		if upErr.Code != "" {
			errCode = upErr.Code
		}
	}
	writeJSON(w, statusCode, map[string]any{
		"error": map[string]string{
			"message": err.Error(),
			"type":    "gateway_error",
			"code":    errCode,
		},
	})
}

func (g *Gateway) handleGetUsage(w http.ResponseWriter, r *http.Request) {
	summary, err := g.Usage(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, summary)
}

func (g *Gateway) handleListLocalProviders(w http.ResponseWriter, r *http.Request) {
	discovery := g.DiscoverLocalProviders(r.Context())
	writeJSON(w, http.StatusOK, discovery)
}

func (g *Gateway) handleGetStats(w http.ResponseWriter, r *http.Request) {
	stats, err := g.Stats(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (g *Gateway) handleListProviders(w http.ResponseWriter, r *http.Request) {
	providers, err := g.ListProviders(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"providers": providers, "count": len(providers)})
}

func (g *Gateway) handleCreateProvider(w http.ResponseWriter, r *http.Request) {
	var input ProviderCreateInput
	if err := decodeJSONBody(r, &input); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid provider JSON"})
		return
	}
	created, err := g.AddProvider(r.Context(), input)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusCreated, created)
}

func (g *Gateway) handleDeleteProvider(w http.ResponseWriter, r *http.Request) {
	err := g.DeleteProvider(r.Context(), r.PathValue("id"))
	if errors.Is(err, sql.ErrNoRows) {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "provider not found"})
		return
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"deleted": true})
}

func (g *Gateway) handleListAgents(w http.ResponseWriter, r *http.Request) {
	routes, err := g.ListAgentRoutes(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"agents": routes, "count": len(routes)})
}

func (g *Gateway) handleUpdateAgent(w http.ResponseWriter, r *http.Request) {
	var input AgentRouteUpdateInput
	if err := decodeJSONBody(r, &input); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid agent route JSON"})
		return
	}
	updated, err := g.UpdateAgentRoute(r.Context(), r.PathValue("agentId"), input)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, updated)
}

func (g *Gateway) handleListCatalog(w http.ResponseWriter, r *http.Request) {
	freeParam := r.URL.Query().Get("freeOnly")
	freeOnly := strings.EqualFold(freeParam, "true") || freeParam == "1"
	entries, err := g.ListCatalog(r.Context(), r.URL.Query().Get("q"), freeOnly)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"entries": entries, "count": len(entries)})
}

func (g *Gateway) handleSniffCatalog(w http.ResponseWriter, r *http.Request) {
	sniffOut, err := g.SniffUpstream(r.Context())
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, sniffOut)
}

func (g *Gateway) handleUpdateSettings(w http.ResponseWriter, r *http.Request) {
	var input SettingsUpdateInput
	if err := decodeJSONBody(r, &input); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid settings JSON"})
		return
	}
	updated, err := g.UpdateSettings(r.Context(), input)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, updated)
}

func (g *Gateway) handlePurgeCache(w http.ResponseWriter, r *http.Request) {
	purged, err := g.PurgeCache(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]int64{"purged": purged})
}
