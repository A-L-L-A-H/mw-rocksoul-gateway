package auth

import (
	"context"
	"crypto"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	DefaultRIDIssuer        = "http://127.0.0.1:4480"
	defaultJWKSFetchTimeout = 10 * time.Second
	defaultJWKSCacheTTL     = 5 * time.Minute
	maxAuthResponseBytes    = 1 << 20
)

type claimsContextKey struct{}

// RIDConfig configures the pure-Go fail-closed OIDC/JWKS verifier for mw-rid (:4480).
type RIDConfig struct {
	Issuer         string
	JWKSURL        string
	Audience       string
	RequiredScopes []string
	HTTPClient     *http.Client
	CacheTTL       time.Duration
	Now            func() time.Time
}

// RIDClaims holds validated claims extracted from an mw-rid RS256 access token.
type RIDClaims struct {
	Subject  string   `json:"sub"`
	Issuer   string   `json:"iss"`
	Audience []string `json:"aud,omitempty"`
	Scope    string   `json:"scope,omitempty"`
	ClientID string   `json:"client_id,omitempty"`
	TokenUse string   `json:"token_use,omitempty"`
	KeyID    string   `json:"kid,omitempty"`
	Expires  int64    `json:"exp"`
}

// RIDVerifier verifies mw-rid RS256 access tokens against OIDC discovery / JWKS endpoints.
// All failures are strictly fail-closed.
type RIDVerifier struct {
	issuer         string
	jwksURL        string
	audience       string
	requiredScopes []string
	httpClient     *http.Client
	cacheTTL       time.Duration
	nowFunc        func() time.Time

	mu             sync.RWMutex
	discoveredJWKS string
	keysByKID      map[string]*rsa.PublicKey
	fetchedAt      time.Time
}

type oidcDiscoveryDocument struct {
	Issuer  string `json:"issuer"`
	JWKSURI string `json:"jwks_uri"`
}

type jwtHeader struct {
	Alg string `json:"alg"`
	Typ string `json:"typ"`
	Kid string `json:"kid"`
}

type jwtPayload struct {
	Issuer   string          `json:"iss"`
	Subject  string          `json:"sub"`
	Audience json.RawMessage `json:"aud"`
	Exp      *float64        `json:"exp"`
	Nbf      *float64        `json:"nbf"`
	TokenUse string          `json:"token_use"`
	AtHash   string          `json:"at_hash"`
	CHash    string          `json:"c_hash"`
	Scope    string          `json:"scope"`
	ClientID string          `json:"client_id"`
}

type jwksDocument struct {
	Keys []jwkEntry `json:"keys"`
}

type jwkEntry struct {
	Kty string `json:"kty"`
	Kid string `json:"kid"`
	Use string `json:"use"`
	Alg string `json:"alg"`
	N   string `json:"n"`
	E   string `json:"e"`
}

// NewRIDVerifier constructs a fail-closed RS256 OIDC/JWKS verifier for mw-rid.
// At least Issuer or JWKSURL must be provided; when JWKSURL is omitted, jwks_uri is
// resolved via <Issuer>/.well-known/openid-configuration on demand.
func NewRIDVerifier(cfg RIDConfig) (*RIDVerifier, error) {
	issuer := strings.TrimRight(strings.TrimSpace(cfg.Issuer), "/")
	jwksURL := strings.TrimSpace(cfg.JWKSURL)
	if issuer == "" && jwksURL == "" {
		return nil, errors.New("rid verifier: issuer or jwks url is required")
	}
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: defaultJWKSFetchTimeout}
	}
	cacheTTL := cfg.CacheTTL
	if cacheTTL <= 0 {
		cacheTTL = defaultJWKSCacheTTL
	}
	nowFn := cfg.Now
	if nowFn == nil {
		nowFn = time.Now
	}

	scopes := make([]string, 0, len(cfg.RequiredScopes))
	for _, scope := range cfg.RequiredScopes {
		if trimmed := strings.TrimSpace(scope); trimmed != "" {
			scopes = append(scopes, trimmed)
		}
	}

	return &RIDVerifier{
		issuer:         issuer,
		jwksURL:        jwksURL,
		audience:       strings.TrimSpace(cfg.Audience),
		requiredScopes: scopes,
		httpClient:     client,
		cacheTTL:       cacheTTL,
		nowFunc:        nowFn,
		keysByKID:      make(map[string]*rsa.PublicKey),
	}, nil
}

// ClaimsFromContext retrieves verified RID claims attached by Middleware.
func ClaimsFromContext(ctx context.Context) (RIDClaims, bool) {
	if ctx == nil {
		return RIDClaims{}, false
	}
	claims, ok := ctx.Value(claimsContextKey{}).(RIDClaims)
	return claims, ok
}

// ExtractBearerToken extracts a Bearer token from an Authorization header.
func ExtractBearerToken(authHeader string) (string, error) {
	trimmed := strings.TrimSpace(authHeader)
	if trimmed == "" {
		return "", errors.New("missing authorization header")
	}
	parts := strings.SplitN(trimmed, " ", 2)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
		return "", errors.New("authorization header must use Bearer scheme")
	}
	token := strings.TrimSpace(parts[1])
	if token == "" {
		return "", errors.New("empty bearer token")
	}
	return token, nil
}

// VerifyAccessToken validates a compact RS256 JWS access token with fail-closed checks.
func (v *RIDVerifier) VerifyAccessToken(ctx context.Context, rawToken string) (RIDClaims, error) {
	if v == nil {
		return RIDClaims{}, errors.New("rid verifier is not initialized")
	}
	trimmedToken := strings.TrimSpace(rawToken)
	if trimmedToken == "" {
		return RIDClaims{}, errors.New("empty bearer token")
	}

	segments := strings.Split(trimmedToken, ".")
	if len(segments) != 3 {
		return RIDClaims{}, errors.New("malformed jwt: expected 3 segments")
	}

	headerBytes, err := base64.RawURLEncoding.DecodeString(segments[0])
	if err != nil {
		return RIDClaims{}, fmt.Errorf("decode jwt header: %w", err)
	}
	var hdr jwtHeader
	if err := json.Unmarshal(headerBytes, &hdr); err != nil {
		return RIDClaims{}, fmt.Errorf("parse jwt header: %w", err)
	}

	// ADR-001: strictly allow RS256 only (blocks HS256, none, algorithm confusion).
	if hdr.Alg != "RS256" {
		return RIDClaims{}, fmt.Errorf("unsupported jwt alg %q: only RS256 is permitted", hdr.Alg)
	}
	if normalizedTyp := strings.ToLower(strings.TrimSpace(hdr.Typ)); normalizedTyp != "" {
		if normalizedTyp != "at+jwt" && normalizedTyp != "application/at+jwt" && normalizedTyp != "jwt" {
			return RIDClaims{}, fmt.Errorf("invalid access token typ header %q", hdr.Typ)
		}
	}

	payloadBytes, err := base64.RawURLEncoding.DecodeString(segments[1])
	if err != nil {
		return RIDClaims{}, fmt.Errorf("decode jwt payload: %w", err)
	}
	var payload jwtPayload
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return RIDClaims{}, fmt.Errorf("parse jwt payload: %w", err)
	}

	// ADR-001: reject OIDC ID tokens.
	if strings.EqualFold(strings.TrimSpace(payload.TokenUse), "id") {
		return RIDClaims{}, errors.New("oidc id tokens are rejected; access token required")
	}
	if strings.TrimSpace(payload.AtHash) != "" || strings.TrimSpace(payload.CHash) != "" {
		return RIDClaims{}, errors.New("oidc id tokens containing at_hash or c_hash are rejected")
	}

	signatureBytes, err := base64.RawURLEncoding.DecodeString(segments[2])
	if err != nil {
		return RIDClaims{}, fmt.Errorf("decode jwt signature: %w", err)
	}

	pubKey, err := v.resolveKey(ctx, strings.TrimSpace(hdr.Kid))
	if err != nil {
		return RIDClaims{}, err
	}

	signingInput := segments[0] + "." + segments[1]
	digest := sha256.Sum256([]byte(signingInput))
	if err := rsa.VerifyPKCS1v15(pubKey, crypto.SHA256, digest[:], signatureBytes); err != nil {
		return RIDClaims{}, errors.New("invalid jwt signature")
	}

	nowUnix := v.nowFunc().UTC().Unix()
	if payload.Exp == nil {
		return RIDClaims{}, errors.New("jwt missing required exp claim")
	}
	expUnix := int64(*payload.Exp)
	if nowUnix >= expUnix {
		return RIDClaims{}, errors.New("jwt expired")
	}
	if payload.Nbf != nil && nowUnix < int64(*payload.Nbf) {
		return RIDClaims{}, errors.New("jwt not yet valid (nbf)")
	}
	if strings.TrimSpace(payload.Subject) == "" {
		return RIDClaims{}, errors.New("jwt missing required sub claim")
	}

	if v.issuer != "" && strings.TrimRight(strings.TrimSpace(payload.Issuer), "/") != v.issuer {
		return RIDClaims{}, fmt.Errorf("jwt issuer mismatch: got %q", payload.Issuer)
	}

	audiences, err := parseAudienceClaim(payload.Audience)
	if err != nil {
		return RIDClaims{}, fmt.Errorf("parse jwt aud claim: %w", err)
	}
	if v.audience != "" && !containsString(audiences, v.audience) {
		return RIDClaims{}, fmt.Errorf("jwt audience mismatch: expected %q", v.audience)
	}

	if len(v.requiredScopes) > 0 {
		grantedScopes := strings.Fields(payload.Scope)
		for _, required := range v.requiredScopes {
			if !containsString(grantedScopes, required) {
				return RIDClaims{}, fmt.Errorf("jwt missing required scope %q", required)
			}
		}
	}

	return RIDClaims{
		Subject:  payload.Subject,
		Issuer:   payload.Issuer,
		Audience: audiences,
		Scope:    payload.Scope,
		ClientID: payload.ClientID,
		TokenUse: payload.TokenUse,
		KeyID:    hdr.Kid,
		Expires:  expUnix,
	}, nil
}

// Middleware enforces fail-closed mw-rid Bearer authentication on all non-exempt paths.
func (v *RIDVerifier) Middleware(exemptPaths ...string) func(http.Handler) http.Handler {
	exemptSet := make(map[string]bool, len(exemptPaths))
	for _, path := range exemptPaths {
		exemptSet[path] = true
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if exemptSet[r.URL.Path] {
				next.ServeHTTP(w, r)
				return
			}
			if v == nil {
				writeUnauthorized(w, "rid verifier unavailable")
				return
			}
			rawToken, err := ExtractBearerToken(r.Header.Get("Authorization"))
			if err != nil {
				writeUnauthorized(w, err.Error())
				return
			}
			claims, err := v.VerifyAccessToken(r.Context(), rawToken)
			if err != nil {
				writeUnauthorized(w, err.Error())
				return
			}
			ctx := context.WithValue(r.Context(), claimsContextKey{}, claims)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func writeUnauthorized(w http.ResponseWriter, detail string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("WWW-Authenticate", `Bearer realm="mw-rocksoul-gateway", error="invalid_token"`)
	w.WriteHeader(http.StatusUnauthorized)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{
			"type":    "authentication_error",
			"code":    "rid_auth_failed",
			"message": detail,
		},
	})
}

func (v *RIDVerifier) resolveKey(ctx context.Context, kid string) (*rsa.PublicKey, error) {
	if cached := v.lookupCachedKey(kid); cached != nil {
		return cached, nil
	}
	if err := v.refreshJWKS(ctx); err != nil {
		return nil, err
	}
	if refreshed := v.lookupCachedKey(kid); refreshed != nil {
		return refreshed, nil
	}
	return nil, fmt.Errorf("signing key not found in jwks for kid %q", kid)
}

func (v *RIDVerifier) lookupCachedKey(kid string) *rsa.PublicKey {
	v.mu.RLock()
	defer v.mu.RUnlock()

	if v.fetchedAt.IsZero() || v.nowFunc().Sub(v.fetchedAt) > v.cacheTTL {
		return nil
	}
	if kid != "" {
		return v.keysByKID[kid]
	}
	if len(v.keysByKID) == 1 {
		for _, onlyKey := range v.keysByKID {
			return onlyKey
		}
	}
	return nil
}

func (v *RIDVerifier) resolveJWKSURL(ctx context.Context) (string, error) {
	if v.jwksURL != "" {
		return v.jwksURL, nil
	}

	v.mu.RLock()
	cached := v.discoveredJWKS
	v.mu.RUnlock()
	if cached != "" {
		return cached, nil
	}

	discoveryURL := v.issuer + "/.well-known/openid-configuration"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, discoveryURL, nil)
	if err != nil {
		return "", fmt.Errorf("build oidc discovery request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := v.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch oidc discovery %s: %w", discoveryURL, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("oidc discovery %s returned HTTP %d", discoveryURL, resp.StatusCode)
	}
	bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, maxAuthResponseBytes))
	if err != nil {
		return "", fmt.Errorf("read oidc discovery: %w", err)
	}

	var doc oidcDiscoveryDocument
	if err := json.Unmarshal(bodyBytes, &doc); err != nil {
		return "", fmt.Errorf("parse oidc discovery json: %w", err)
	}
	if doc.Issuer != "" && strings.TrimRight(strings.TrimSpace(doc.Issuer), "/") != v.issuer {
		return "", fmt.Errorf("oidc discovery issuer mismatch: got %q expected %q", doc.Issuer, v.issuer)
	}
	jwksURI := strings.TrimSpace(doc.JWKSURI)
	if jwksURI == "" {
		return "", errors.New("oidc discovery document missing jwks_uri")
	}

	v.mu.Lock()
	v.discoveredJWKS = jwksURI
	v.mu.Unlock()
	return jwksURI, nil
}

func (v *RIDVerifier) refreshJWKS(ctx context.Context) error {
	targetJWKSURL, err := v.resolveJWKSURL(ctx)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetJWKSURL, nil)
	if err != nil {
		return fmt.Errorf("build jwks request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := v.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("fetch jwks: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fetch jwks: unexpected status %d", resp.StatusCode)
	}
	bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, maxAuthResponseBytes))
	if err != nil {
		return fmt.Errorf("read jwks response: %w", err)
	}

	var doc jwksDocument
	if err := json.Unmarshal(bodyBytes, &doc); err != nil {
		return fmt.Errorf("parse jwks json: %w", err)
	}

	parsedKeys := make(map[string]*rsa.PublicKey, len(doc.Keys))
	for _, entry := range doc.Keys {
		if entry.Kty != "RSA" {
			continue
		}
		if entry.Use != "" && entry.Use != "sig" {
			continue
		}
		if entry.Alg != "" && entry.Alg != "RS256" {
			continue
		}
		pubKey, parseErr := jwkToRSAPublicKey(entry.N, entry.E)
		if parseErr != nil {
			continue
		}
		keyID := strings.TrimSpace(entry.Kid)
		parsedKeys[keyID] = pubKey
	}
	if len(parsedKeys) == 0 {
		return errors.New("jwks contains no usable RS256 RSA signing keys")
	}

	v.mu.Lock()
	v.keysByKID = parsedKeys
	v.fetchedAt = v.nowFunc()
	v.mu.Unlock()
	return nil
}

func jwkToRSAPublicKey(modulusB64, exponentB64 string) (*rsa.PublicKey, error) {
	modulusBytes, err := base64.RawURLEncoding.DecodeString(strings.TrimSpace(modulusB64))
	if err != nil || len(modulusBytes) == 0 {
		return nil, errors.New("invalid jwk modulus")
	}
	exponentBytes, err := base64.RawURLEncoding.DecodeString(strings.TrimSpace(exponentB64))
	if err != nil || len(exponentBytes) == 0 {
		return nil, errors.New("invalid jwk exponent")
	}

	modulus := new(big.Int).SetBytes(modulusBytes)
	exponentBig := new(big.Int).SetBytes(exponentBytes)
	if !exponentBig.IsInt64() {
		return nil, errors.New("jwk exponent overflow")
	}
	exponent := int(exponentBig.Int64())
	if exponent < 3 {
		return nil, errors.New("invalid jwk exponent")
	}
	return &rsa.PublicKey{N: modulus, E: exponent}, nil
}

func parseAudienceClaim(raw json.RawMessage) ([]string, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if strings.TrimSpace(single) == "" {
			return nil, nil
		}
		return []string{single}, nil
	}
	var multiple []string
	if err := json.Unmarshal(raw, &multiple); err == nil {
		return multiple, nil
	}
	return nil, errors.New("aud must be string or array of strings")
}

func containsString(items []string, expected string) bool {
	for _, candidate := range items {
		if candidate == expected {
			return true
		}
	}
	return false
}
