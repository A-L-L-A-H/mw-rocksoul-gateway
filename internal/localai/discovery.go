package localai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
)

const (
	DefaultProbeTimeout     = 800 * time.Millisecond
	maxLocalAIResponseBytes = 2 << 20 // 2 MiB
)

var devModelPattern = regexp.MustCompile(`(?i)coder|code|deepseek|starcoder|codellama|devstral|qwen.*coder`)

// Target defines a loopback local inference server probe configuration.
type Target struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Kind       string `json:"kind"` // "ollama", "lmstudio", "llamacpp"
	ProbeURL   string `json:"probeUrl"`
	OpenAIBase string `json:"openaiBaseUrl"`
}

// Model represents a model discovered on a loopback local inference runtime.
type Model struct {
	ID            string `json:"id"`
	ProviderID    string `json:"providerId"`
	ProviderName  string `json:"providerName"`
	ProviderKind  string `json:"providerKind"`
	Endpoint      string `json:"endpoint"`
	OpenAIBaseURL string `json:"openaiBaseUrl"`
	SizeBytes     int64  `json:"sizeBytes,omitempty"`
	Family        string `json:"family,omitempty"`
}

// ProviderStatus captures the probe result for a single local inference endpoint.
type ProviderStatus struct {
	ID            string  `json:"id"`
	Name          string  `json:"name"`
	Kind          string  `json:"kind"`
	Endpoint      string  `json:"endpoint"`
	OpenAIBaseURL string  `json:"openaiBaseUrl"`
	Online        bool    `json:"online"`
	LatencyMs     int64   `json:"latencyMs"`
	Models        []Model `json:"models"`
	Error         string  `json:"error,omitempty"`
}

// DiscoveryResult summarizes all probed loopback providers and discovered models.
type DiscoveryResult struct {
	OnlineCount               int              `json:"onlineCount"`
	Providers                 []ProviderStatus `json:"providers"`
	Models                    []Model          `json:"models"`
	RecommendedWorkspaceModel string           `json:"recommendedWorkspaceModel"`
	RecommendedDevModel       string           `json:"recommendedDevModel"`
}

// Discoverer probes loopback-only local AI inference runtimes.
type Discoverer struct {
	targets    []Target
	timeout    time.Duration
	httpClient *http.Client
}

// DefaultTargets returns the canonical loopback probe targets for
// Ollama (:11434), LM Studio (:1234), and llama.cpp (:8080).
func DefaultTargets() []Target {
	return []Target{
		{
			ID:         "ollama",
			Name:       "Ollama",
			Kind:       "ollama",
			ProbeURL:   "http://127.0.0.1:11434/api/tags",
			OpenAIBase: "http://127.0.0.1:11434/v1",
		},
		{
			ID:         "lmstudio",
			Name:       "LM Studio",
			Kind:       "lmstudio",
			ProbeURL:   "http://127.0.0.1:1234/v1/models",
			OpenAIBase: "http://127.0.0.1:1234/v1",
		},
		{
			ID:         "llamacpp",
			Name:       "llama.cpp",
			Kind:       "llamacpp",
			ProbeURL:   "http://127.0.0.1:8080/v1/models",
			OpenAIBase: "http://127.0.0.1:8080/v1",
		},
	}
}

// NewDiscoverer creates a loopback-only local AI provider discoverer.
func NewDiscoverer(targets []Target, timeout time.Duration, client *http.Client) *Discoverer {
	if len(targets) == 0 {
		targets = DefaultTargets()
	}
	if timeout <= 0 {
		timeout = DefaultProbeTimeout
	}
	if client == nil {
		client = NewLoopbackHTTPClient(timeout)
	}
	copied := make([]Target, len(targets))
	copy(copied, targets)
	return &Discoverer{
		targets:    copied,
		timeout:    timeout,
		httpClient: client,
	}
}

// IsLoopbackHostname returns true strictly for localhost, 127.0.0.1, and ::1.
func IsLoopbackHostname(hostname string) bool {
	cleaned := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(hostname)), ".")
	cleaned = strings.TrimPrefix(cleaned, "[")
	cleaned = strings.TrimSuffix(cleaned, "]")
	if cleaned == "" {
		return false
	}
	switch cleaned {
	case "localhost", "127.0.0.1", "::1":
		return true
	}
	parsedIP := net.ParseIP(cleaned)
	if parsedIP == nil {
		return false
	}
	if v4 := parsedIP.To4(); v4 != nil {
		return v4.Equal(net.IPv4(127, 0, 0, 1))
	}
	return parsedIP.Equal(net.IPv6loopback)
}

// ValidateLoopbackEndpointURL strictly validates that rawURL is an http/https
// loopback endpoint without userinfo or fragments.
func ValidateLoopbackEndpointURL(rawURL string) (*url.URL, error) {
	trimmed := strings.TrimSpace(rawURL)
	if trimmed == "" {
		return nil, errors.New("empty endpoint URL")
	}
	parsed, err := url.Parse(trimmed)
	if err != nil {
		return nil, fmt.Errorf("invalid endpoint URL: %w", err)
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "http" && scheme != "https" {
		return nil, fmt.Errorf("rejected non-http(s) endpoint scheme %q", parsed.Scheme)
	}
	if parsed.User != nil {
		return nil, errors.New("rejected endpoint URL containing userinfo")
	}
	if parsed.Fragment != "" {
		return nil, errors.New("rejected endpoint URL containing fragment")
	}
	if !IsLoopbackHostname(parsed.Hostname()) {
		return nil, fmt.Errorf("rejected non-loopback endpoint host %q", parsed.Hostname())
	}
	return parsed, nil
}

// IsLoopbackEndpointURL reports whether rawURL passes the strict loopback-only guard.
func IsLoopbackEndpointURL(rawURL string) bool {
	_, err := ValidateLoopbackEndpointURL(rawURL)
	return err == nil
}

// NewLoopbackHTTPClient returns an http.Client that enforces loopback-only dialing
// (blocking DNS rebinding to non-loopback IPs) and disables HTTP redirects.
func NewLoopbackHTTPClient(timeout time.Duration) *http.Client {
	if timeout <= 0 {
		timeout = DefaultProbeTimeout
	}
	dialer := &net.Dialer{Timeout: timeout}
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, _, splitErr := net.SplitHostPort(addr)
			if splitErr != nil {
				return nil, fmt.Errorf("invalid target address %q: %w", addr, splitErr)
			}
			if !IsLoopbackHostname(host) {
				return nil, fmt.Errorf("blocked non-loopback dial target %q", host)
			}
			ips, lookupErr := net.DefaultResolver.LookupIPAddr(ctx, host)
			if lookupErr != nil {
				return nil, lookupErr
			}
			if len(ips) == 0 {
				return nil, fmt.Errorf("no IP addresses resolved for loopback host %q", host)
			}
			for _, ipAddr := range ips {
				if !ipAddr.IP.IsLoopback() {
					return nil, fmt.Errorf("blocked non-loopback resolved IP %s for %q", ipAddr.IP.String(), host)
				}
			}
			return dialer.DialContext(ctx, network, addr)
		},
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

// Discover concurrently probes the configured loopback local AI runtimes.
func (d *Discoverer) Discover(ctx context.Context) DiscoveryResult {
	return DiscoverWithClient(ctx, d.targets, d.httpClient, d.timeout)
}

// Discover concurrently probes loopback local AI targets with the default timeout.
func Discover(ctx context.Context, targets []Target) DiscoveryResult {
	return DiscoverWithClient(ctx, targets, nil, DefaultProbeTimeout)
}

// DiscoverWithClient probes loopback targets using an optional custom HTTP client and timeout.
func DiscoverWithClient(
	ctx context.Context,
	targets []Target,
	customClient *http.Client,
	timeout time.Duration,
) DiscoveryResult {
	if len(targets) == 0 {
		targets = DefaultTargets()
	}
	if timeout <= 0 {
		timeout = DefaultProbeTimeout
	}
	client := customClient
	if client == nil {
		client = NewLoopbackHTTPClient(timeout)
	}

	statuses := make([]ProviderStatus, len(targets))
	var waitGroup sync.WaitGroup
	for idx, target := range targets {
		waitGroup.Add(1)
		go func(slot int, probeTarget Target) {
			defer waitGroup.Done()
			statuses[slot] = probeTargetEndpoint(ctx, probeTarget, timeout, client)
		}(idx, target)
	}
	waitGroup.Wait()

	onlineCount := 0
	allModels := make([]Model, 0)
	for _, status := range statuses {
		if status.Online {
			onlineCount++
		}
		allModels = append(allModels, status.Models...)
	}

	workspaceModel, devModel := selectRecommendedModels(allModels)
	return DiscoveryResult{
		OnlineCount:               onlineCount,
		Providers:                 statuses,
		Models:                    allModels,
		RecommendedWorkspaceModel: workspaceModel,
		RecommendedDevModel:       devModel,
	}
}

func probeTargetEndpoint(
	ctx context.Context,
	target Target,
	timeout time.Duration,
	client *http.Client,
) ProviderStatus {
	openAIBase := strings.TrimRight(strings.TrimSpace(target.OpenAIBase), "/")
	if openAIBase == "" {
		openAIBase = deriveOpenAIBase(target.ProbeURL)
	}
	status := ProviderStatus{
		ID:            target.ID,
		Name:          target.Name,
		Kind:          target.Kind,
		Endpoint:      target.ProbeURL,
		OpenAIBaseURL: openAIBase,
		Models:        []Model{},
	}

	if _, guardErr := ValidateLoopbackEndpointURL(target.ProbeURL); guardErr != nil {
		status.Error = guardErr.Error()
		return status
	}
	if openAIBase != "" {
		if _, baseErr := ValidateLoopbackEndpointURL(openAIBase); baseErr != nil {
			status.Error = baseErr.Error()
			return status
		}
	}

	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, reqErr := http.NewRequestWithContext(probeCtx, http.MethodGet, target.ProbeURL, nil)
	if reqErr != nil {
		status.Error = reqErr.Error()
		return status
	}
	req.Header.Set("Accept", "application/json")

	startedAt := time.Now()
	resp, doErr := client.Do(req)
	status.LatencyMs = time.Since(startedAt).Milliseconds()
	if doErr != nil {
		status.Error = doErr.Error()
		return status
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		status.Error = fmt.Sprintf("HTTP %d", resp.StatusCode)
		return status
	}

	rawBody, readErr := io.ReadAll(io.LimitReader(resp.Body, maxLocalAIResponseBytes))
	if readErr != nil {
		status.Error = readErr.Error()
		return status
	}

	normalizedTarget := target
	normalizedTarget.OpenAIBase = openAIBase

	var parsedModels []Model
	if strings.EqualFold(target.Kind, "ollama") {
		parsedModels = parseOllamaModels(rawBody, normalizedTarget)
	} else {
		parsedModels = parseOpenAICompatModels(rawBody, normalizedTarget)
		if len(parsedModels) == 0 {
			parsedModels = parseOllamaModels(rawBody, normalizedTarget)
		}
	}

	status.Online = true
	status.Models = parsedModels
	return status
}

func deriveOpenAIBase(probeURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(probeURL))
	if err != nil || parsed.Host == "" {
		return ""
	}
	return fmt.Sprintf("%s://%s/v1", parsed.Scheme, parsed.Host)
}

func parseOllamaModels(rawBody []byte, target Target) []Model {
	var payload struct {
		Models []struct {
			Name    string `json:"name"`
			Model   string `json:"model"`
			Size    int64  `json:"size"`
			Details struct {
				Family string `json:"family"`
			} `json:"details"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rawBody, &payload); err != nil {
		return []Model{}
	}

	models := make([]Model, 0, len(payload.Models))
	for _, entry := range payload.Models {
		modelID := strings.TrimSpace(entry.Name)
		if modelID == "" {
			modelID = strings.TrimSpace(entry.Model)
		}
		if modelID == "" {
			continue
		}
		models = append(models, Model{
			ID:            modelID,
			ProviderID:    target.ID,
			ProviderName:  target.Name,
			ProviderKind:  target.Kind,
			Endpoint:      target.ProbeURL,
			OpenAIBaseURL: target.OpenAIBase,
			SizeBytes:     entry.Size,
			Family:        strings.TrimSpace(entry.Details.Family),
		})
	}
	return models
}

func parseOpenAICompatModels(rawBody []byte, target Target) []Model {
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rawBody, &payload); err != nil {
		return []Model{}
	}

	models := make([]Model, 0, len(payload.Data))
	for _, entry := range payload.Data {
		modelID := strings.TrimSpace(entry.ID)
		if modelID == "" {
			continue
		}
		models = append(models, Model{
			ID:            modelID,
			ProviderID:    target.ID,
			ProviderName:  target.Name,
			ProviderKind:  target.Kind,
			Endpoint:      target.ProbeURL,
			OpenAIBaseURL: target.OpenAIBase,
		})
	}
	return models
}

func selectRecommendedModels(models []Model) (workspaceModel string, devModel string) {
	if len(models) == 0 {
		return "", ""
	}
	for _, model := range models {
		if devModel == "" && devModelPattern.MatchString(model.ID) {
			devModel = model.ID
		}
		if workspaceModel == "" && !devModelPattern.MatchString(model.ID) {
			workspaceModel = model.ID
		}
		if devModel != "" && workspaceModel != "" {
			break
		}
	}
	if devModel == "" {
		devModel = models[0].ID
	}
	if workspaceModel == "" {
		workspaceModel = models[0].ID
	}
	return workspaceModel, devModel
}
