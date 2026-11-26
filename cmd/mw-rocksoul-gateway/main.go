package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/auth"
	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/gateway"
	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/localai"
)

const (
	serviceName = "mw-rocksoul-gateway"
	defaultPort = "4510"
)

type serverConfig struct {
	ListenAddr            string
	Port                  string
	DataDir               string
	AuthMode              string
	RIDIssuer             string
	RIDJWKSURL            string
	RIDAudience           string
	DefaultClassification gateway.DataClassification
}

func defaultDataDir() string {
	if envDir := strings.TrimSpace(os.Getenv("ROCKSOUL_GATEWAY_DATA_DIR")); envDir != "" {
		return envDir
	}
	if localAppData := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); localAppData != "" {
		return filepath.Join(localAppData, serviceName)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "./data"
	}
	return filepath.Join(home, "."+serviceName)
}

func firstEnv(keys []string, fallback string) string {
	for _, key := range keys {
		if val := strings.TrimSpace(os.Getenv(key)); val != "" {
			return val
		}
	}
	return fallback
}

func resolveListenAddr(host, port string) (string, string) {
	cleanPort := strings.TrimPrefix(strings.TrimSpace(port), ":")
	if cleanPort == "" {
		cleanPort = defaultPort
	}
	cleanHost := strings.TrimSpace(host)
	if cleanHost == "" {
		return ":" + cleanPort, cleanPort
	}
	return cleanHost + ":" + cleanPort, cleanPort
}

func newHTTPHandler(gw *gateway.Gateway, verifier *auth.RIDVerifier, cfg serverConfig) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":                "ok",
			"service":               serviceName,
			"port":                  cfg.Port,
			"authMode":              cfg.AuthMode,
			"defaultClassification": string(gw.DefaultClassification()),
			"dataDir":               gw.DataDir(),
		})
	})
	gw.RegisterRoutes(mux)

	if strings.EqualFold(cfg.AuthMode, "disabled") {
		return mux
	}
	// Fail-closed RID authentication on all routes except GET /health.
	return verifier.Middleware("/health")(mux)
}

func main() {
	portFlag := flag.String("port", firstEnv([]string{"ROCKSOUL_GATEWAY_PORT", "PORT"}, defaultPort), "TCP listen port")
	hostFlag := flag.String("host", strings.TrimSpace(os.Getenv("ROCKSOUL_GATEWAY_HOST")), "Optional bind host (empty binds all interfaces)")
	dataDirFlag := flag.String("data-dir", defaultDataDir(), "SQLite and encrypted key state directory")
	authModeFlag := flag.String("auth-mode", firstEnv([]string{"ROCKSOUL_AUTH_MODE"}, "rid"), "Authentication mode: rid (fail-closed) or disabled")
	ridIssuerFlag := flag.String("rid-issuer", firstEnv([]string{"MW_RID_ISSUER", "MW_RID_URL"}, auth.DefaultRIDIssuer), "mw-rid OIDC issuer URL")
	ridJWKSFlag := flag.String("rid-jwks-url", firstEnv([]string{"ROCKSOUL_RID_JWKS_URL", "MW_RID_JWKS_URL"}, ""), "Optional explicit mw-rid JWKS URL")
	ridAudienceFlag := flag.String("rid-audience", firstEnv([]string{"ROCKSOUL_RID_AUDIENCE"}, serviceName), "Expected JWT aud claim")
	defaultClassFlag := flag.String("default-classification", firstEnv([]string{"ROCKSOUL_DEFAULT_CLASSIFICATION"}, string(gateway.ClassificationCloudEligible)), "Default data classification (local_only|cloud_eligible)")
	flag.Parse()

	parsedClass, err := gateway.ParseDataClassification(*defaultClassFlag)
	if err != nil {
		log.Fatalf("[%s] invalid default classification: %v", serviceName, err)
	}
	if parsedClass == "" {
		parsedClass = gateway.ClassificationCloudEligible
	}

	listenAddr, cleanPort := resolveListenAddr(*hostFlag, *portFlag)
	cfg := serverConfig{
		ListenAddr:            listenAddr,
		Port:                  cleanPort,
		DataDir:               *dataDirFlag,
		AuthMode:              strings.ToLower(strings.TrimSpace(*authModeFlag)),
		RIDIssuer:             *ridIssuerFlag,
		RIDJWKSURL:            *ridJWKSFlag,
		RIDAudience:           *ridAudienceFlag,
		DefaultClassification: parsedClass,
	}

	discoverer := localai.NewDiscoverer(localai.DefaultTargets(), localai.DefaultProbeTimeout, nil)
	gw, err := gateway.Open(gateway.Options{
		DataDir:               cfg.DataDir,
		DefaultClassification: cfg.DefaultClassification,
		LocalDiscovery:        discoverer,
	})
	if err != nil {
		log.Fatalf("[%s] initialize gateway: %v", serviceName, err)
	}
	defer func() { _ = gw.Close() }()

	geminiEnabled, geminiErr := gw.ConfigureGemini(context.Background(), gateway.GeminiConfigFromEnv(os.Getenv))
	if geminiErr != nil {
		log.Fatalf("[%s] configure google ai studio: %v", serviceName, geminiErr)
	}
	if geminiEnabled {
		log.Printf("[%s] google ai studio (gemini free tier) enabled", serviceName)
	}
	freeRegistered, freeErr := gw.ConfigureFreeProviders(context.Background(), gateway.FreeProvidersConfigFromEnv(os.Getenv))
	if freeErr != nil {
		log.Fatalf("[%s] configure free fallback providers: %v", serviceName, freeErr)
	}
	if len(freeRegistered) > 0 {
		log.Printf("[%s] free fallback providers enabled: %s", serviceName, strings.Join(freeRegistered, ", "))
	}

	var verifier *auth.RIDVerifier
	if cfg.AuthMode != "disabled" {
		verifier, err = auth.NewRIDVerifier(auth.RIDConfig{
			Issuer:   cfg.RIDIssuer,
			JWKSURL:  cfg.RIDJWKSURL,
			Audience: cfg.RIDAudience,
		})
		if err != nil {
			log.Fatalf("[%s] initialize RID auth verifier (fail-closed): %v", serviceName, err)
		}
	} else {
		log.Printf("[%s] WARNING: ROCKSOUL_AUTH_MODE=disabled (local development bypass active)", serviceName)
	}

	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           newHTTPHandler(gw, verifier, cfg),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		log.Printf("[%s] listening on %s (auth=%s, defaultClassification=%s, dataDir=%s)",
			serviceName, cfg.ListenAddr, cfg.AuthMode, cfg.DefaultClassification, gw.DataDir())
		if serveErr := srv.ListenAndServe(); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			log.Fatalf("[%s] server error: %v", serviceName, serveErr)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	<-sigCh

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if shutdownErr := srv.Shutdown(ctx); shutdownErr != nil {
		log.Printf("[%s] shutdown error: %v", serviceName, shutdownErr)
	}
}
