package gateway

import (
	"context"
	"database/sql"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	baseSmartCooldownDuration = 30 * time.Second
	maxSmartCooldownDuration  = 15 * time.Minute
	maxRetryAfterCooldown     = 30 * time.Minute
)

// ProviderQuotaLimits defines daily request (RPD) and per-minute token (TPM) limits for a provider.
// Zero values indicate unlimited capacity (e.g. local loopback runtimes).
type ProviderQuotaLimits struct {
	RPDLimit int64 `json:"rpdLimit"`
	TPMLimit int64 `json:"tpmLimit"`
}

// ProviderQuotaStatus reports live RPD/TPM usage, remaining quota, and smart cooldown backoff state.
type ProviderQuotaStatus struct {
	ProviderID          string `json:"providerId"`
	ProviderName        string `json:"providerName"`
	Tier                string `json:"tier"`
	IsLocal             bool   `json:"isLocal"`
	DayBucket           string `json:"dayBucket"`
	RPDLimit            int64  `json:"rpdLimit"`
	RPDUsed             int64  `json:"rpdUsed"`
	RPDRemaining        int64  `json:"rpdRemaining"`
	RPDExhausted        bool   `json:"rpdExhausted"`
	TPMLimit            int64  `json:"tpmLimit"`
	TPMUsed             int64  `json:"tpmUsed"`
	TPMRemaining        int64  `json:"tpmRemaining"`
	TPMExhausted        bool   `json:"tpmExhausted"`
	ConsecutiveFailures int    `json:"consecutiveFailures"`
	LastBackoffSeconds  int64  `json:"lastBackoffSeconds"`
	InCooldown          bool   `json:"inCooldown"`
	CooldownUntil       string `json:"cooldownUntil,omitempty"`
}

type tokenWindowEntry struct {
	at     time.Time
	tokens int64
}

type providerQuotaBucket struct {
	customLimits        *ProviderQuotaLimits
	dayBucket           string
	rpdUsed             int64
	tokenWindow         []tokenWindowEntry
	consecutiveFailures int
	lastBackoff         time.Duration
}

// QuotaTracker tracks per-provider RPD/TPM sliding quotas and exponential cooldown backoff.
type QuotaTracker struct {
	mu      sync.Mutex
	clock   func() time.Time
	buckets map[string]*providerQuotaBucket
}

// NewQuotaTracker creates a QuotaTracker with the given clock function.
func NewQuotaTracker(clock func() time.Time) *QuotaTracker {
	if clock == nil {
		clock = func() time.Time { return time.Now().UTC() }
	}
	return &QuotaTracker{
		clock:   clock,
		buckets: make(map[string]*providerQuotaBucket),
	}
}

// DefaultFreeTierQuotaLimits returns canonical free-tier RPD and TPM limits for known cloud providers.
func DefaultFreeTierQuotaLimits(providerName, tier string, isLocal bool) ProviderQuotaLimits {
	if !strings.EqualFold(strings.TrimSpace(tier), TierFree) {
		return ProviderQuotaLimits{}
	}
	name := strings.ToLower(strings.TrimSpace(providerName))
	switch name {
	case GeminiProviderName:
		return ProviderQuotaLimits{RPDLimit: 1500, TPMLimit: 1000000}
	case GroqProviderName:
		return ProviderQuotaLimits{RPDLimit: 14400, TPMLimit: 12000}
	case OpenRouterProviderName:
		return ProviderQuotaLimits{RPDLimit: 200, TPMLimit: 60000}
	case NvidiaProviderName:
		return ProviderQuotaLimits{RPDLimit: 1000, TPMLimit: 40000}
	case MistralProviderName:
		return ProviderQuotaLimits{RPDLimit: 500, TPMLimit: 500000}
	default:
		if isLocal || strings.HasPrefix(name, "local-") {
			return ProviderQuotaLimits{}
		}
		return ProviderQuotaLimits{RPDLimit: 1000, TPMLimit: 60000}
	}
}

func (q *QuotaTracker) setClock(clock func() time.Time) {
	if clock == nil {
		return
	}
	q.mu.Lock()
	q.clock = clock
	q.mu.Unlock()
}

func (q *QuotaTracker) getOrCreateLocked(providerID string) *providerQuotaBucket {
	b, ok := q.buckets[providerID]
	if !ok {
		b = &providerQuotaBucket{}
		q.buckets[providerID] = b
	}
	return b
}

func (q *QuotaTracker) resolveLimitsLocked(b *providerQuotaBucket, p Provider) ProviderQuotaLimits {
	if b.customLimits != nil {
		return *b.customLimits
	}
	return DefaultFreeTierQuotaLimits(p.Name, p.Tier, p.IsLocal)
}

func (q *QuotaTracker) refreshWindowsLocked(b *providerQuotaBucket, now time.Time) int64 {
	day := now.UTC().Format("2006-01-02")
	if b.dayBucket != day {
		b.dayBucket = day
		b.rpdUsed = 0
	}
	cutoff := now.Add(-60 * time.Second)
	kept := b.tokenWindow[:0]
	var tpmUsed int64
	for _, entry := range b.tokenWindow {
		if !entry.at.Before(cutoff) {
			kept = append(kept, entry)
			tpmUsed += entry.tokens
		}
	}
	b.tokenWindow = kept
	return tpmUsed
}

// SetLimits overrides the RPD and TPM quota limits for a specific provider ID.
func (q *QuotaTracker) SetLimits(providerID string, limits ProviderQuotaLimits) {
	q.mu.Lock()
	defer q.mu.Unlock()
	b := q.getOrCreateLocked(providerID)
	cp := limits
	b.customLimits = &cp
}

// Allow reports whether provider has remaining RPD and TPM quota for a new request.
func (q *QuotaTracker) Allow(provider Provider, estimatedTokens int) (bool, string) {
	if strings.HasPrefix(provider.ID, "local_") {
		return true, ""
	}
	q.mu.Lock()
	defer q.mu.Unlock()

	now := q.clock()
	b := q.getOrCreateLocked(provider.ID)
	limits := q.resolveLimitsLocked(b, provider)
	if limits.RPDLimit <= 0 && limits.TPMLimit <= 0 {
		return true, ""
	}
	tpmUsed := q.refreshWindowsLocked(b, now)

	if limits.RPDLimit > 0 && b.rpdUsed >= limits.RPDLimit {
		return false, "rpd_quota_exhausted"
	}
	if limits.TPMLimit > 0 {
		if tpmUsed >= limits.TPMLimit {
			return false, "tpm_quota_exhausted"
		}
		if estimatedTokens > 0 && tpmUsed > 0 && tpmUsed+int64(estimatedTokens) > limits.TPMLimit {
			return false, "tpm_quota_exhausted"
		}
	}
	return true, ""
}

// RecordSuccess records one completed request and its token consumption, resetting backoff state.
func (q *QuotaTracker) RecordSuccess(providerID string, promptTokens, completionTokens int) {
	q.mu.Lock()
	defer q.mu.Unlock()

	now := q.clock()
	b := q.getOrCreateLocked(providerID)
	_ = q.refreshWindowsLocked(b, now)

	b.rpdUsed++
	total := int64(promptTokens + completionTokens)
	if total < 0 {
		total = 0
	}
	if total > 0 {
		b.tokenWindow = append(b.tokenWindow, tokenWindowEntry{at: now, tokens: total})
	}
	b.consecutiveFailures = 0
	b.lastBackoff = 0
}

// RecordFailureAndComputeCooldown increments consecutive failure count and computes smart exponential backoff.
func (q *QuotaTracker) RecordFailureAndComputeCooldown(providerID string, retryAfter time.Duration) time.Duration {
	q.mu.Lock()
	defer q.mu.Unlock()

	b := q.getOrCreateLocked(providerID)
	b.consecutiveFailures++

	shift := b.consecutiveFailures - 1
	if shift > 6 {
		shift = 6
	}
	backoff := baseSmartCooldownDuration * time.Duration(1<<shift)
	if backoff > maxSmartCooldownDuration {
		backoff = maxSmartCooldownDuration
	}
	if retryAfter > backoff {
		backoff = retryAfter
		if backoff > maxRetryAfterCooldown {
			backoff = maxRetryAfterCooldown
		}
	}
	b.lastBackoff = backoff
	return backoff
}

// Status returns a snapshot of the provider's RPD/TPM quota and cooldown backoff status.
func (q *QuotaTracker) Status(provider Provider) ProviderQuotaStatus {
	q.mu.Lock()
	defer q.mu.Unlock()

	now := q.clock()
	b := q.getOrCreateLocked(provider.ID)
	limits := q.resolveLimitsLocked(b, provider)
	tpmUsed := q.refreshWindowsLocked(b, now)

	rpdRem := int64(0)
	rpdEx := false
	if limits.RPDLimit > 0 {
		rpdRem = limits.RPDLimit - b.rpdUsed
		if rpdRem <= 0 {
			rpdRem = 0
			rpdEx = true
		}
	}
	tpmRem := int64(0)
	tpmEx := false
	if limits.TPMLimit > 0 {
		tpmRem = limits.TPMLimit - tpmUsed
		if tpmRem <= 0 {
			tpmRem = 0
			tpmEx = true
		}
	}

	return ProviderQuotaStatus{
		ProviderID:          provider.ID,
		ProviderName:        provider.Name,
		Tier:                provider.Tier,
		IsLocal:             provider.IsLocal,
		DayBucket:           b.dayBucket,
		RPDLimit:            limits.RPDLimit,
		RPDUsed:             b.rpdUsed,
		RPDRemaining:        rpdRem,
		RPDExhausted:        rpdEx,
		TPMLimit:            limits.TPMLimit,
		TPMUsed:             tpmUsed,
		TPMRemaining:        tpmRem,
		TPMExhausted:        tpmEx,
		ConsecutiveFailures: b.consecutiveFailures,
		LastBackoffSeconds:  int64(b.lastBackoff / time.Second),
		InCooldown:          provider.InCooldown,
		CooldownUntil:       provider.CooldownUntil,
	}
}

// ParseRetryAfterHeader parses an HTTP Retry-After header (seconds or HTTP-date).
func ParseRetryAfterHeader(raw string, now time.Time) time.Duration {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return 0
	}
	if secs, err := strconv.Atoi(trimmed); err == nil && secs > 0 {
		return time.Duration(secs) * time.Second
	}
	if parsedTime, err := http.ParseTime(trimmed); err == nil {
		if delta := parsedTime.Sub(now); delta > 0 {
			return delta
		}
	}
	return 0
}

func (g *Gateway) now() time.Time {
	g.mu.RLock()
	clockFn := g.clock
	g.mu.RUnlock()
	if clockFn != nil {
		return clockFn().UTC()
	}
	return time.Now().UTC()
}

// SetClock overrides the gateway's time source (for deterministic quota and cooldown testing).
func (g *Gateway) SetClock(clock func() time.Time) {
	if clock == nil {
		return
	}
	g.mu.Lock()
	g.clock = clock
	g.mu.Unlock()
	if g.quotas != nil {
		g.quotas.setClock(clock)
	}
}

// SetProviderQuotaLimits configures custom RPD/TPM quota limits for a provider.
func (g *Gateway) SetProviderQuotaLimits(providerID string, limits ProviderQuotaLimits) {
	if g.quotas != nil {
		g.quotas.SetLimits(providerID, limits)
	}
}

// GetProviderQuotaStatus returns the current RPD/TPM quota and smart cooldown status for providerID.
func (g *Gateway) GetProviderQuotaStatus(ctx context.Context, providerID string) (ProviderQuotaStatus, error) {
	providers, err := g.ListProviders(ctx)
	if err != nil {
		return ProviderQuotaStatus{}, err
	}
	for _, p := range providers {
		if p.ID == providerID || strings.EqualFold(p.Name, providerID) {
			return g.quotas.Status(p), nil
		}
	}
	return ProviderQuotaStatus{}, sql.ErrNoRows
}

// ListProviderQuotas returns RPD/TPM quota and smart cooldown statuses for all configured providers.
func (g *Gateway) ListProviderQuotas(ctx context.Context) ([]ProviderQuotaStatus, error) {
	providers, err := g.ListProviders(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]ProviderQuotaStatus, 0, len(providers))
	for _, p := range providers {
		out = append(out, g.quotas.Status(p))
	}
	return out, nil
}

func (g *Gateway) handleListQuotas(w http.ResponseWriter, r *http.Request) {
	quotas, err := g.ListProviderQuotas(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"quotas": quotas, "count": len(quotas)})
}
