package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

const (
	rtkToolCompressThreshold = 600
	rtkMaxLinesAfterCap      = 90
	rtkMaxCharsAfterCap      = 3600
	rtkHeadKeepLines         = 30
	rtkTailKeepLines         = 30
	rtkMaxMiddleErrorLines   = 20
)

var (
	ansiCSIRegex    = regexp.MustCompile(`\x1b\[[0-9;?]*[ -/]*[@-~]`)
	ansiOSCRegex    = regexp.MustCompile(`\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)`)
	ansiSingleRegex = regexp.MustCompile(`\x1b[@-Z\\\-_]`)

	separatorLineRegex = regexp.MustCompile(`^[\s=\-~*_#─━┄┈═+|]{6,}$`)
	gitNoiseLineRegex  = regexp.MustCompile(`^(index [0-9a-f]{6,}\.\.[0-9a-f]{6,}|old mode \d+|new mode \d+|similarity index \d+%|dissimilarity index \d+%|rename from |rename to |new file mode \d+|deleted file mode \d+)`)
	lsPermPrefixRegex  = regexp.MustCompile(`^([drwx\-lcbsp@+]{10,12})\s+\d+\s+\S+\s+\S+\s+(\d+[KMG]?)\s+[A-Z][a-z]{2}\s+\d+\s+[\d:]+\s+(.+)$`)
	timestampLeadRegex = regexp.MustCompile(`^(?:\[\d{2,4}[-/:T.\dZ\s]+\]|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?|\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+`)
	progressNoiseRegex = regexp.MustCompile(`(?i)^(downloading|extracting|pulling fs layer|waiting|verifying checksum|download complete|already exists|resolving deltas|receiving objects|counting objects|compressing objects|remote:\s*(counting|compressing|total)|=== RUN\s+|--- PASS:\s+|PASS\s*$|ok\s+\S+\s+\d|added \d+ packages|audited \d+ packages|\d+ packages are looking for funding)`)
)

// CompressMessages runs the RTK token-saving compressor across a conversation
// without mutating the caller's slice. It strips ANSI sequences, collapses noisy
// tool output (role == "tool" and Anthropic tool_result blocks) above 600 chars,
// deduplicates consecutive repeated lines with [xN repeated], and caps extreme
// dumps while preserving head, diagnostic error lines, and tail.
func CompressMessages(messages []ChatMessage) ([]ChatMessage, int) {
	if len(messages) == 0 {
		return nil, 0
	}

	compressed := make([]ChatMessage, len(messages))
	rawChars := 0
	compressedChars := 0

	for idx, msg := range messages {
		copied := msg
		isToolRole := strings.EqualFold(strings.TrimSpace(msg.Role), "tool")

		switch contentVal := msg.Content.(type) {
		case string:
			rawChars += len(contentVal)
			cleaned := compressTextContent(contentVal, isToolRole)
			copied.Content = cleaned
			compressedChars += len(cleaned)

		case []any:
			nextBlocks, blockRaw, blockCompressed := compressContentBlocks(contentVal, isToolRole)
			copied.Content = nextBlocks
			rawChars += blockRaw
			compressedChars += blockCompressed

		case []map[string]any:
			genericSlice := make([]any, len(contentVal))
			for blockIdx, blockMap := range contentVal {
				genericSlice[blockIdx] = blockMap
			}
			nextBlocks, blockRaw, blockCompressed := compressContentBlocks(genericSlice, isToolRole)
			copied.Content = nextBlocks
			rawChars += blockRaw
			compressedChars += blockCompressed
		}

		compressed[idx] = copied
	}

	if rawChars <= compressedChars {
		return compressed, 0
	}
	return compressed, (rawChars - compressedChars) / 4
}

func compressContentBlocks(blocks []any, parentIsTool bool) ([]any, int, int) {
	outBlocks := make([]any, len(blocks))
	rawChars := 0
	compressedChars := 0

	for idx, rawBlock := range blocks {
		blockMap, ok := rawBlock.(map[string]any)
		if !ok {
			outBlocks[idx] = rawBlock
			continue
		}

		cloned := make(map[string]any, len(blockMap))
		for key, val := range blockMap {
			cloned[key] = val
		}

		blockType, _ := cloned["type"].(string)
		isToolBlock := parentIsTool || strings.EqualFold(blockType, "tool_result")

		switch blockType {
		case "tool_result":
			switch inner := cloned["content"].(type) {
			case string:
				rawChars += len(inner)
				cleaned := compressTextContent(inner, true)
				cloned["content"] = cleaned
				compressedChars += len(cleaned)
			case []any:
				subBlocks, subRaw, subComp := compressContentBlocks(inner, true)
				cloned["content"] = subBlocks
				rawChars += subRaw
				compressedChars += subComp
			}
		case "text":
			if textStr, isStr := cloned["text"].(string); isStr {
				rawChars += len(textStr)
				cleaned := compressTextContent(textStr, isToolBlock)
				cloned["text"] = cleaned
				compressedChars += len(cleaned)
			}
		default:
			if textStr, isStr := cloned["text"].(string); isStr && isToolBlock {
				rawChars += len(textStr)
				cleaned := compressTextContent(textStr, true)
				cloned["text"] = cleaned
				compressedChars += len(cleaned)
			}
		}

		outBlocks[idx] = cloned
	}

	return outBlocks, rawChars, compressedChars
}

func compressTextContent(raw string, aggressiveToolMode bool) string {
	if raw == "" {
		return ""
	}
	cleaned := stripANSIAndProgressCR(raw)
	if !aggressiveToolMode {
		return lightNormalizeWhitespace(cleaned)
	}
	if len(raw) <= rtkToolCompressThreshold && len(cleaned) <= rtkToolCompressThreshold {
		return lightNormalizeWhitespace(cleaned)
	}
	return compressVerboseToolDump(cleaned)
}

func stripANSIAndProgressCR(raw string) string {
	withoutANSI := ansiOSCRegex.ReplaceAllString(raw, "")
	withoutANSI = ansiCSIRegex.ReplaceAllString(withoutANSI, "")
	withoutANSI = ansiSingleRegex.ReplaceAllString(withoutANSI, "")

	normalized := strings.ReplaceAll(withoutANSI, "\r\n", "\n")
	if !strings.ContainsRune(normalized, '\r') {
		return normalized
	}

	rawLines := strings.Split(normalized, "\n")
	for idx, line := range rawLines {
		if strings.ContainsRune(line, '\r') {
			segments := strings.Split(line, "\r")
			kept := ""
			for segIdx := len(segments) - 1; segIdx >= 0; segIdx-- {
				if strings.TrimSpace(segments[segIdx]) != "" {
					kept = segments[segIdx]
					break
				}
			}
			rawLines[idx] = kept
		}
	}
	return strings.Join(rawLines, "\n")
}

func lightNormalizeWhitespace(input string) string {
	rawLines := strings.Split(input, "\n")
	outLines := make([]string, 0, len(rawLines))
	blankRun := 0

	for _, line := range rawLines {
		trimmedRight := strings.TrimRight(line, " \t")
		if trimmedRight == "" {
			blankRun++
			if blankRun <= 1 {
				outLines = append(outLines, "")
			}
			continue
		}
		blankRun = 0
		outLines = append(outLines, trimmedRight)
	}
	return strings.TrimSpace(strings.Join(outLines, "\n"))
}

func compressVerboseToolDump(input string) string {
	rawLines := strings.Split(input, "\n")
	filtered := make([]string, 0, len(rawLines))
	blankRun := 0
	isLongDump := len(rawLines) > 25

	for _, rawLine := range rawLines {
		line := strings.TrimRight(rawLine, " \t")
		trimmed := strings.TrimSpace(line)

		if trimmed == "" {
			blankRun++
			if blankRun <= 1 {
				filtered = append(filtered, "")
			}
			continue
		}

		if separatorLineRegex.MatchString(trimmed) {
			continue
		}

		if !isDiagnosticLine(trimmed) {
			if gitNoiseLineRegex.MatchString(trimmed) {
				continue
			}
			if isLongDump && progressNoiseRegex.MatchString(trimmed) {
				continue
			}
			if strings.HasPrefix(trimmed, "total ") && len(trimmed) < 20 && isLongDump {
				continue
			}
		}

		if lsMatches := lsPermPrefixRegex.FindStringSubmatch(trimmed); len(lsMatches) == 4 {
			line = lsMatches[1][:1] + " " + lsMatches[2] + " " + lsMatches[3]
		}

		blankRun = 0
		filtered = append(filtered, line)
	}

	deduped := deduplicateConsecutiveLines(filtered)

	if len(deduped) > rtkMaxLinesAfterCap {
		deduped = capLinesPreserveErrors(deduped)
	}

	joined := strings.TrimSpace(strings.Join(deduped, "\n"))
	if len(joined) > rtkMaxCharsAfterCap {
		joined = capCharsPreserveHeadTail(joined, rtkMaxCharsAfterCap)
	}
	return joined
}

func deduplicateConsecutiveLines(lines []string) []string {
	if len(lines) == 0 {
		return nil
	}

	out := make([]string, 0, len(lines))
	idx := 0
	for idx < len(lines) {
		current := lines[idx]
		normKey := normalizeForDedup(current)
		if normKey == "" {
			out = append(out, current)
			idx++
			continue
		}

		runCount := 1
		scanIdx := idx + 1
		for scanIdx < len(lines) {
			nextNonBlank := scanIdx
			for nextNonBlank < len(lines) && strings.TrimSpace(lines[nextNonBlank]) == "" {
				nextNonBlank++
			}
			if nextNonBlank >= len(lines) || normalizeForDedup(lines[nextNonBlank]) != normKey {
				break
			}
			runCount++
			scanIdx = nextNonBlank + 1
		}

		if runCount >= 2 {
			out = append(out, fmt.Sprintf("%s [x%d repeated]", current, runCount))
		} else {
			out = append(out, current)
		}
		idx = scanIdx
	}
	return out
}

func normalizeForDedup(line string) string {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return ""
	}
	if isDiagnosticLine(trimmed) {
		return trimmed
	}
	stripped := timestampLeadRegex.ReplaceAllString(trimmed, "")
	return strings.TrimSpace(stripped)
}

func isDiagnosticLine(line string) bool {
	lower := strings.ToLower(line)
	markers := []string{
		"error", "fail", "panic", "fatal", "exception",
		"traceback", "warn", "denied", "timeout", "assert",
		"exit status", "segfault", "undefined", "cannot find",
	}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func capLinesPreserveErrors(lines []string) []string {
	total := len(lines)
	if total <= rtkHeadKeepLines+rtkTailKeepLines+5 {
		return lines
	}

	head := lines[:rtkHeadKeepLines]
	middle := lines[rtkHeadKeepLines : total-rtkTailKeepLines]
	tail := lines[total-rtkTailKeepLines:]

	middleErrors := make([]string, 0, rtkMaxMiddleErrorLines)
	for _, middleLine := range middle {
		if isDiagnosticLine(middleLine) && len(middleErrors) < rtkMaxMiddleErrorLines {
			middleErrors = append(middleErrors, middleLine)
		}
	}

	omittedCount := len(middle) - len(middleErrors)
	resultLines := make([]string, 0, len(head)+len(middleErrors)+len(tail)+2)
	resultLines = append(resultLines, head...)
	if omittedCount > 0 {
		resultLines = append(resultLines, fmt.Sprintf("[... RTK omitted %d verbose lines ...]", omittedCount))
	}
	resultLines = append(resultLines, middleErrors...)
	resultLines = append(resultLines, tail...)
	return resultLines
}

func capCharsPreserveHeadTail(text string, maxChars int) string {
	if len(text) <= maxChars {
		return text
	}
	headBudget := maxChars / 2
	tailBudget := maxChars - headBudget - 48
	if tailBudget < 200 {
		tailBudget = 200
	}
	omittedChars := len(text) - headBudget - tailBudget
	if omittedChars <= 0 {
		return text
	}
	return text[:headBudget] + fmt.Sprintf("\n[... RTK trimmed %d chars ...]\n", omittedChars) + text[len(text)-tailBudget:]
}

var (
	cavemanPolitePhraseRegex = regexp.MustCompile(`(?i)\b(could you please|would you please|would you mind|i would like you to|can you please|please help me|thank you very much|thank you so much|thanks a lot|thank you|thanks|please|kindly|actually|basically|literally|terima kasih banyak|terima kasih|terimakasih|selamat pagi|selamat siang|selamat sore|selamat malam|mohon bantuannya|mohon bantuan|tolong bantu|tolong|mohon|halo|hai|baiklah|siap|pada dasarnya|sebenarnya)\b[,!.:;]*`)
	cavemanStopwordSet       = map[string]bool{
		"the": true, "a": true, "an": true, "that": true, "which": true,
		"is": true, "are": true, "was": true, "were": true, "be": true,
		"been": true, "being": true, "just": true, "really": true, "very": true,
		"quite": true, "rather": true, "somewhat": true, "simply": true,
		"yang": true, "adalah": true, "tersebut": true, "sangat": true,
		"sekali": true, "dong": true, "deh": true, "nih": true, "sih": true,
		"yah": true, "juga": true, "telah": true, "sudah": true,
	}
	cavemanMultiSpaceRegex = regexp.MustCompile(`[ \t]{2,}`)
)

// CompressCaveman runs ultra-terse compression across a conversation without
// mutating the caller's slice. It applies standard RTK tool compression,
// compacts JSON payloads, and strips English/Indonesian polite filler & safe
// stopwords from prose while preserving fenced code blocks and diagnostics.
func CompressCaveman(messages []ChatMessage) ([]ChatMessage, int) {
	if len(messages) == 0 {
		return nil, 0
	}

	compressed := make([]ChatMessage, len(messages))
	rawChars := 0
	compressedChars := 0

	for idx, msg := range messages {
		copied := msg
		isToolRole := strings.EqualFold(strings.TrimSpace(msg.Role), "tool")

		switch contentVal := msg.Content.(type) {
		case string:
			rawChars += len(contentVal)
			cleaned := compressCavemanText(contentVal, isToolRole)
			copied.Content = cleaned
			compressedChars += len(cleaned)

		case []any:
			nextBlocks, blockRaw, blockCompressed := compressCavemanContentBlocks(contentVal, isToolRole)
			copied.Content = nextBlocks
			rawChars += blockRaw
			compressedChars += blockCompressed

		case []map[string]any:
			genericSlice := make([]any, len(contentVal))
			for blockIdx, blockMap := range contentVal {
				genericSlice[blockIdx] = blockMap
			}
			nextBlocks, blockRaw, blockCompressed := compressCavemanContentBlocks(genericSlice, isToolRole)
			copied.Content = nextBlocks
			rawChars += blockRaw
			compressedChars += blockCompressed
		}

		compressed[idx] = copied
	}

	if rawChars <= compressedChars {
		return compressed, 0
	}
	return compressed, (rawChars - compressedChars) / 4
}

func compressCavemanContentBlocks(blocks []any, parentIsTool bool) ([]any, int, int) {
	outBlocks := make([]any, len(blocks))
	rawChars := 0
	compressedChars := 0

	for idx, rawBlock := range blocks {
		blockMap, ok := rawBlock.(map[string]any)
		if !ok {
			outBlocks[idx] = rawBlock
			continue
		}

		cloned := make(map[string]any, len(blockMap))
		for key, val := range blockMap {
			cloned[key] = val
		}

		blockType, _ := cloned["type"].(string)
		isToolBlock := parentIsTool || strings.EqualFold(blockType, "tool_result")

		switch blockType {
		case "tool_result":
			switch inner := cloned["content"].(type) {
			case string:
				rawChars += len(inner)
				cleaned := compressCavemanText(inner, true)
				cloned["content"] = cleaned
				compressedChars += len(cleaned)
			case []any:
				subBlocks, subRaw, subComp := compressCavemanContentBlocks(inner, true)
				cloned["content"] = subBlocks
				rawChars += subRaw
				compressedChars += subComp
			}
		case "text":
			if textStr, isStr := cloned["text"].(string); isStr {
				rawChars += len(textStr)
				cleaned := compressCavemanText(textStr, isToolBlock)
				cloned["text"] = cleaned
				compressedChars += len(cleaned)
			}
		default:
			if textStr, isStr := cloned["text"].(string); isStr && isToolBlock {
				rawChars += len(textStr)
				cleaned := compressCavemanText(textStr, true)
				cloned["text"] = cleaned
				compressedChars += len(cleaned)
			}
		}

		outBlocks[idx] = cloned
	}

	return outBlocks, rawChars, compressedChars
}

func compressCavemanText(raw string, aggressiveToolMode bool) string {
	if raw == "" {
		return ""
	}
	base := compressTextContent(raw, aggressiveToolMode)
	trimmed := strings.TrimSpace(base)
	if trimmed == "" {
		return ""
	}

	if (strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}")) ||
		(strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]")) {
		var compacted bytes.Buffer
		if err := json.Compact(&compacted, []byte(trimmed)); err == nil {
			return compacted.String()
		}
	}

	lines := strings.Split(base, "\n")
	outLines := make([]string, 0, len(lines))
	inCodeFence := false

	for idx := 0; idx < len(lines); idx++ {
		line := lines[idx]
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "```") {
			inCodeFence = !inCodeFence
			outLines = append(outLines, trimmedLine)
			continue
		}
		if inCodeFence {
			if trimmedLine != "" {
				outLines = append(outLines, strings.TrimRight(line, " \t"))
			}
			continue
		}
		if trimmedLine == "" {
			continue
		}

		if strings.HasPrefix(trimmedLine, "{") || strings.HasPrefix(trimmedLine, "[") {
			var candidateBuilder strings.Builder
			compactedJSON := ""
			endIdx := idx
			for scanIdx := idx; scanIdx < len(lines) && scanIdx-idx < 200; scanIdx++ {
				if scanIdx > idx {
					candidateBuilder.WriteByte('\n')
				}
				candidateBuilder.WriteString(strings.TrimSpace(lines[scanIdx]))
				var compacted bytes.Buffer
				if err := json.Compact(&compacted, []byte(candidateBuilder.String())); err == nil {
					compactedJSON = compacted.String()
					endIdx = scanIdx
					break
				}
			}
			if compactedJSON != "" {
				outLines = append(outLines, compactedJSON)
				idx = endIdx
				continue
			}
		}

		terse := cavemanStripProse(trimmedLine)
		if terse != "" {
			outLines = append(outLines, terse)
		}
	}

	return strings.TrimSpace(strings.Join(outLines, "\n"))
}

func cavemanStripProse(line string) string {
	withoutFiller := cavemanPolitePhraseRegex.ReplaceAllString(line, " ")
	tokens := strings.Fields(withoutFiller)
	if len(tokens) == 0 {
		return ""
	}
	kept := make([]string, 0, len(tokens))
	for _, tok := range tokens {
		cleanWord := strings.ToLower(strings.Trim(tok, ".,;:!?()[]\"'"))
		if cleanWord != "" && cavemanStopwordSet[cleanWord] {
			continue
		}
		kept = append(kept, tok)
	}
	joined := strings.Join(kept, " ")
	return strings.TrimSpace(cavemanMultiSpaceRegex.ReplaceAllString(joined, " "))
}
