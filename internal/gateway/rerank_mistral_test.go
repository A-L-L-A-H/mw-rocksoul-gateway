package gateway_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestOTLPUsageTelemetryEmitted(t *testing.T) {
	_, handler := setupTestGateway(t)
	got := make(chan string, 1)
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		got <- r.URL.Path + string(raw)
	}))
	defer collector.Close()
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.URL)
	t.Setenv("COHERE_API_KEY", "")
	body := `{"query":"x","documents":["x y"]}`
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/rerank", strings.NewReader(body)))
	select {
	case payload := <-got:
		if !strings.HasPrefix(payload, "/v1/traces") || !strings.Contains(payload, "gateway.rerank") {
			t.Fatalf("bad otlp payload: %s", payload)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no otlp span emitted")
	}
}
