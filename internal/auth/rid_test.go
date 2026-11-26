package auth_test

import (
	"context"
	"crypto"
	"crypto/hmac"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/auth"
)

func TestRIDVerifierOIDCDiscoveryAndFailClosedRules(t *testing.T) {
	key1, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate rsa key 1: %v", err)
	}
	key2, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate rsa key 2: %v", err)
	}

	var rotated atomic.Bool
	var ridServer *httptest.Server
	ridServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/.well-known/openid-configuration":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"issuer":   ridServer.URL,
				"jwks_uri": ridServer.URL + "/jwks",
			})
		case "/jwks":
			keys := []map[string]any{rsaToJWK("kid-1", key1)}
			if rotated.Load() {
				keys = append(keys, rsaToJWK("kid-2", key2))
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"keys": keys})
		default:
			http.NotFound(w, r)
		}
	}))
	defer ridServer.Close()

	verifier, err := auth.NewRIDVerifier(auth.RIDConfig{
		Issuer:         ridServer.URL,
		Audience:       "mw-rocksoul-gateway",
		RequiredScopes: []string{"ai:invoke"},
	})
	if err != nil {
		t.Fatalf("new rid verifier: %v", err)
	}

	validToken := mustSignRS256(t, map[string]any{
		"alg": "RS256",
		"typ": "at+jwt",
		"kid": "kid-1",
	}, map[string]any{
		"iss":       ridServer.URL,
		"sub":       "rid_user_42",
		"aud":       []string{"mw-rocksoul-gateway"},
		"exp":       time.Now().Add(time.Hour).Unix(),
		"nbf":       time.Now().Add(-time.Minute).Unix(),
		"scope":     "openid ai:invoke",
		"client_id": "rocksoul-cli",
		"token_use": "access",
	}, key1)

	claims, err := verifier.VerifyAccessToken(context.Background(), validToken)
	if err != nil {
		t.Fatalf("valid token failed verification: %v", err)
	}
	if claims.Subject != "rid_user_42" || claims.ClientID != "rocksoul-cli" {
		t.Fatalf("unexpected claims: %+v", claims)
	}

	// Key rotation triggers JWKS refresh on unknown kid.
	rotated.Store(true)
	rotatedToken := mustSignRS256(t, map[string]any{
		"alg": "RS256",
		"typ": "JWT",
		"kid": "kid-2",
	}, map[string]any{
		"iss":   ridServer.URL,
		"sub":   "rid_user_99",
		"aud":   "mw-rocksoul-gateway",
		"exp":   time.Now().Add(time.Hour).Unix(),
		"scope": "ai:invoke",
	}, key2)
	if _, err := verifier.VerifyAccessToken(context.Background(), rotatedToken); err != nil {
		t.Fatalf("rotated key token failed verification: %v", err)
	}

	// Fail-closed negative cases.
	badCases := []struct {
		name  string
		token string
	}{
		{"empty", ""},
		{"malformed", "not-a-jwt"},
		{"hs256", signHS256(map[string]any{"alg": "HS256", "typ": "JWT"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "scope": "ai:invoke"}, []byte("secret"))},
		{"none_alg", signNone(map[string]any{"alg": "none", "typ": "JWT"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "scope": "ai:invoke"})},
		{"oidc_id_token", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "scope": "ai:invoke", "token_use": "id"}, key1)},
		{"oidc_at_hash", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "scope": "ai:invoke", "at_hash": "abc"}, key1)},
		{"expired", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(-time.Minute).Unix(), "scope": "ai:invoke"}, key1)},
		{"missing_exp", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "scope": "ai:invoke"}, key1)},
		{"future_nbf", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "nbf": time.Now().Add(30 * time.Minute).Unix(), "scope": "ai:invoke"}, key1)},
		{"missing_sub", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "scope": "ai:invoke"}, key1)},
		{"wrong_issuer", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": "https://evil.example", "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "scope": "ai:invoke"}, key1)},
		{"wrong_audience", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "other-service", "exp": time.Now().Add(time.Hour).Unix(), "scope": "ai:invoke"}, key1)},
		{"missing_scope", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "scope": "read:only"}, key1)},
		{"wrong_signature_key", mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "kid-1"}, map[string]any{"iss": ridServer.URL, "sub": "u1", "aud": "mw-rocksoul-gateway", "exp": time.Now().Add(time.Hour).Unix(), "scope": "ai:invoke"}, key2)},
	}

	for _, tc := range badCases {
		if _, err := verifier.VerifyAccessToken(context.Background(), tc.token); err == nil {
			t.Errorf("case %s: expected rejection, got nil error", tc.name)
		}
	}
}

func TestRIDMiddlewareFailClosedAndHealthExempt(t *testing.T) {
	privKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate rsa key: %v", err)
	}

	var jwksUp atomic.Bool
	jwksUp.Store(false)
	jwksSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !jwksUp.Load() {
			http.Error(w, "rid unavailable", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"keys": []map[string]any{rsaToJWK("k1", privKey)},
		})
	}))
	defer jwksSrv.Close()

	verifier, err := auth.NewRIDVerifier(auth.RIDConfig{
		Issuer:   "http://127.0.0.1:4480",
		JWKSURL:  jwksSrv.URL,
		Audience: "mw-rocksoul-gateway",
	})
	if err != nil {
		t.Fatalf("new verifier: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		claims, ok := auth.ClaimsFromContext(r.Context())
		if !ok || claims.Subject == "" {
			http.Error(w, "missing claims", http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"sub":"` + claims.Subject + `"}`))
	})

	protected := verifier.Middleware("/health")(mux)

	// 1. Exempt /health passes without auth.
	healthRec := httptest.NewRecorder()
	protected.ServeHTTP(healthRec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if healthRec.Code != http.StatusOK {
		t.Fatalf("expected /health 200, got %d", healthRec.Code)
	}

	// 2. Missing Bearer token fails closed with 401.
	noAuthRec := httptest.NewRecorder()
	protected.ServeHTTP(noAuthRec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if noAuthRec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without token, got %d", noAuthRec.Code)
	}

	validTok := mustSignRS256(t, map[string]any{"alg": "RS256", "kid": "k1"}, map[string]any{
		"iss": "http://127.0.0.1:4480",
		"sub": "rid_agent_1",
		"aud": "mw-rocksoul-gateway",
		"exp": time.Now().Add(time.Hour).Unix(),
	}, privKey)

	// 3. Valid token STILL fails closed with 401 when JWKS/mw-rid is down.
	downReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	downReq.Header.Set("Authorization", "Bearer "+validTok)
	downRec := httptest.NewRecorder()
	protected.ServeHTTP(downRec, downReq)
	if downRec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 fail-closed when JWKS unavailable, got %d", downRec.Code)
	}

	// 4. Once JWKS is online, valid token passes with claims attached.
	jwksUp.Store(true)
	okReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	okReq.Header.Set("Authorization", "Bearer "+validTok)
	okRec := httptest.NewRecorder()
	protected.ServeHTTP(okRec, okReq)
	if okRec.Code != http.StatusOK {
		t.Fatalf("expected 200 with valid token and healthy JWKS, got %d: %s", okRec.Code, okRec.Body.String())
	}
}

func rsaToJWK(kid string, priv *rsa.PrivateKey) map[string]any {
	return map[string]any{
		"kty": "RSA",
		"kid": kid,
		"use": "sig",
		"alg": "RS256",
		"n":   base64.RawURLEncoding.EncodeToString(priv.N.Bytes()),
		"e":   base64.RawURLEncoding.EncodeToString(big.NewInt(int64(priv.E)).Bytes()),
	}
}

func mustSignRS256(t *testing.T, header, payload map[string]any, priv *rsa.PrivateKey) string {
	t.Helper()
	hBytes, _ := json.Marshal(header)
	pBytes, _ := json.Marshal(payload)
	input := base64.RawURLEncoding.EncodeToString(hBytes) + "." + base64.RawURLEncoding.EncodeToString(pBytes)
	digest := sha256.Sum256([]byte(input))
	sig, err := rsa.SignPKCS1v15(rand.Reader, priv, crypto.SHA256, digest[:])
	if err != nil {
		t.Fatalf("sign rs256: %v", err)
	}
	return input + "." + base64.RawURLEncoding.EncodeToString(sig)
}

func signHS256(header, payload map[string]any, secret []byte) string {
	hBytes, _ := json.Marshal(header)
	pBytes, _ := json.Marshal(payload)
	input := base64.RawURLEncoding.EncodeToString(hBytes) + "." + base64.RawURLEncoding.EncodeToString(pBytes)
	mac := hmac.New(sha256.New, secret)
	_, _ = mac.Write([]byte(input))
	return input + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func signNone(header, payload map[string]any) string {
	hBytes, _ := json.Marshal(header)
	pBytes, _ := json.Marshal(payload)
	return base64.RawURLEncoding.EncodeToString(hBytes) + "." + base64.RawURLEncoding.EncodeToString(pBytes) + "."
}
