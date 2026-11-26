package gateway

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	// DefaultGroqWhisperBaseURL is the Groq OpenAI-compatible root hosting whisper.
	DefaultGroqWhisperBaseURL = "https://api.groq.com/openai/v1"
	// DefaultDeepgramBaseURL is the optional Deepgram fallback root.
	DefaultDeepgramBaseURL = "https://api.deepgram.com/v1"
	// DefaultWhisperModel is the Groq free-tier transcription model.
	DefaultWhisperModel = "whisper-large-v3-turbo"

	maxAudioUploadBytes = 25 << 20
)

// handleAudioTranscriptions proxies multipart audio to Groq whisper-large-v3-turbo,
// with optional Deepgram fallback. local_only classification is strictly blocked
// because audio would leave the loopback boundary.
func (g *Gateway) handleAudioTranscriptions(w http.ResponseWriter, r *http.Request) {
	classification, err := g.resolveClassification(r.Header.Get(headerClassification), r.URL.Query().Get("data_classification"))
	if err != nil {
		writeGatewayError(w, &upstreamCallError{StatusCode: http.StatusBadRequest, Code: "invalid_data_classification", Body: err.Error()})
		return
	}
	if classification == ClassificationLocalOnly {
		writeGatewayError(w, &upstreamCallError{
			StatusCode: http.StatusForbidden,
			Code:       "local_only_egress_blocked",
			Body:       "data classification local_only strictly blocks external audio egress",
		})
		return
	}
	groqKey := strings.TrimSpace(os.Getenv("GROQ_API_KEY"))
	deepgramKey := strings.TrimSpace(os.Getenv("DEEPGRAM_API_KEY"))
	if groqKey == "" && deepgramKey == "" {
		writeGatewayError(w, &upstreamCallError{StatusCode: http.StatusServiceUnavailable, Code: "no_healthy_providers", Body: "no transcription provider configured"})
		return
	}
	payload, err := io.ReadAll(io.LimitReader(r.Body, maxAudioUploadBytes+1))
	_ = r.Body.Close()
	if err != nil || len(payload) == 0 || len(payload) > maxAudioUploadBytes {
		writeGatewayError(w, &upstreamCallError{StatusCode: http.StatusBadRequest, Code: "invalid_audio_payload", Body: "audio payload empty, unreadable, or exceeds 25MiB"})
		return
	}
	contentType := r.Header.Get("Content-Type")
	client := &http.Client{Timeout: 60 * time.Second}

	if groqKey != "" {
		base := strings.TrimRight(firstNonEmpty(os.Getenv("GROQ_BASE_URL"), DefaultGroqWhisperBaseURL), "/")
		status, body, ctype, callErr := forwardAudio(r, client, base+"/audio/transcriptions", "Bearer "+groqKey, contentType, payload)
		if callErr == nil && status < 400 {
			relayAudio(w, status, body, ctype, "groq-whisper")
			return
		}
		if deepgramKey == "" {
			if callErr != nil {
				writeGatewayError(w, &upstreamCallError{StatusCode: http.StatusBadGateway, Code: "upstream_unreachable", Body: callErr.Error()})
				return
			}
			relayAudio(w, status, body, ctype, "groq-whisper")
			return
		}
	}
	base := strings.TrimRight(firstNonEmpty(os.Getenv("DEEPGRAM_BASE_URL"), DefaultDeepgramBaseURL), "/")
	status, body, ctype, callErr := forwardAudio(r, client, base+"/listen?model=nova-2", "Token "+deepgramKey, contentType, payload)
	if callErr != nil {
		writeGatewayError(w, &upstreamCallError{StatusCode: http.StatusBadGateway, Code: "upstream_unreachable", Body: callErr.Error()})
		return
	}
	relayAudio(w, status, body, ctype, "deepgram")
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func forwardAudio(r *http.Request, client *http.Client, target, authorization, contentType string, payload []byte) (int, []byte, string, error) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target, bytes.NewReader(payload))
	if err != nil {
		return 0, nil, "", err
	}
	req.Header.Set("Authorization", authorization)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, "", err
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxIncomingRequestBytes))
	return resp.StatusCode, body, resp.Header.Get("Content-Type"), err
}

func relayAudio(w http.ResponseWriter, status int, body []byte, contentType, provider string) {
	if contentType == "" {
		contentType = "application/json"
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set(headerProvider, provider)
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
