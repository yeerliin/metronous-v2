package tui

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/kiosvantra/metronous/internal/benchmark"
	"github.com/kiosvantra/metronous/internal/discovery"
	"github.com/kiosvantra/metronous/internal/store"
)

// maxBenchmarkRows is the maximum number of rows to fetch per page in the benchmark tab.
const maxBenchmarkRows = 20

// benchmarkRefreshInterval is the auto-refresh period for the benchmark tab,
// matching the tracking tab's cadence.
const benchmarkRefreshInterval = 2 * time.Second

// benchmarkTickMsg is sent by the auto-refresh ticker.
type benchmarkTickMsg struct{ t time.Time }

// BenchmarkDataMsg carries fetched benchmark runs.
type BenchmarkDataMsg struct {
	Runs      []store.BenchmarkRun
	TypeByID  map[string]string   // agentID → type label (primary/subagent/built-in/all)
	TrendByID map[string][]string // "agentID\tmodel" → verdict trend (oldest first)
	Err       error
}

// trendKey builds the composite key for trendByID lookups.
func trendKey(agentID, model string) string {
	return agentID + "\t" + model
}

// Verdict colour styles.
var (
	verdictKeep             = lipgloss.NewStyle().Foreground(lipgloss.Color("82"))  // green
	verdictSwitch           = lipgloss.NewStyle().Foreground(lipgloss.Color("196")) // red
	verdictUrgent           = lipgloss.NewStyle().Foreground(lipgloss.Color("196")) // red
	verdictInsufficientData = lipgloss.NewStyle().Foreground(lipgloss.Color("226")) // yellow/dim
	verdictOther            = lipgloss.NewStyle().Foreground(lipgloss.Color("240")) // grey
)

// detailPanelStyle styles the decision rationale detail panel.
var detailPanelStyle = lipgloss.NewStyle().
	Foreground(lipgloss.Color("252"))

// Score color styles for the composite score column.
var (
	scoreGreen  = lipgloss.NewStyle().Foreground(lipgloss.Color("82"))  // >= 0.80
	scoreYellow = lipgloss.NewStyle().Foreground(lipgloss.Color("226")) // >= 0.50
	scoreRed    = lipgloss.NewStyle().Foreground(lipgloss.Color("196")) // < 0.50
)

// scoreStyle returns the lipgloss style appropriate for the given composite score value.
func scoreStyle(s float64) lipgloss.Style {
	if s >= 0.80 {
		return scoreGreen
	}
	if s >= 0.50 {
		return scoreYellow
	}
	return scoreRed
}

// detailLabelStyle styles the label keys in the detail panel.
var detailLabelStyle = lipgloss.NewStyle().
	Bold(true).
	Foreground(lipgloss.Color("33"))

// benchColWidths / benchColNames describe the benchmark history table.
// Columns: Time | Agent | Type | Score | Accuracy | P95 Latency | Verdict | → Model | Savings
// "Time" shows full date+time (YYYY-MM-DD HH:MM) so width is 17 to avoid truncation.
// Score column (index 3) was added for the composite score feature.
var (
	benchColWidths = []int{17, 16, 9, 6, 10, 12, 18, 16, 8}
	benchColNames  = []string{"Time", "Agent", "Type", "Score", "Accuracy", "P95 Latency", "Verdict", "Model", "Savings"}
)

// verdictColIdx is the index of the Verdict column in benchColNames/benchColWidths.
// Defined as a constant so the rendering code stays in sync with the column layout.
// Updated from 5 to 6 when the Score column was inserted at index 3.
const verdictColIdx = 6

// modelPricingSection mirrors the JSON structure of the "model_pricing" key in thresholds.json.
type modelPricingSection struct {
	Models map[string]float64 `json:"models"`
}

// loadModelPricing reads the "model_pricing.models" section from thresholds.json located
// in the parent directory of dataDir (i.e. dataDir/../thresholds.json).
// Returns an empty map if the file cannot be read or the section is absent — callers
// treat an empty map as "pricing unknown" and display "-" for savings.
func loadModelPricing(dataDir string) map[string]float64 {
	if dataDir == "" {
		return map[string]float64{}
	}
	thresholdsPath := filepath.Join(dataDir, "..", "thresholds.json")
	data, err := os.ReadFile(thresholdsPath)
	if err != nil {
		return map[string]float64{}
	}
	var raw struct {
		ModelPricing *modelPricingSection `json:"model_pricing"`
	}
	if err := json.Unmarshal(data, &raw); err != nil || raw.ModelPricing == nil {
		return map[string]float64{}
	}
	return raw.ModelPricing.Models
}

// BenchmarkModel is the Bubble Tea sub-model for the benchmark history tab.
type BenchmarkModel struct {
	bs        store.BenchmarkStore
	runs      []store.BenchmarkRun
	agents    []discovery.AgentInfo
	typeByID  map[string]string   // agentID → type label (primary/subagent/built-in/all)
	trendByID map[string][]string // "agentID\tmodel" → verdict trend (oldest first)
	err       error
	// cursor is the local row index within the current page (0..maxBenchmarkRows-1).
	cursor  int
	loading bool
	// pageOffset is the number of runs skipped from the top (run_at DESC).
	// PgDn increases pageOffset (moves toward older runs).
	// PgUp decreases pageOffset (moves toward newer runs).
	pageOffset int
	// detailFrozen indicates whether the detail panel is locked to frozenRun/frozenTrend.
	// When true, the detail does not update even if the background refresh changes m.runs.
	detailFrozen bool
	// frozenRun is the run whose detail panel is displayed when detailFrozen == true.
	frozenRun store.BenchmarkRun
	// frozenTrend is the verdict trend for frozenRun, captured at freeze time.
	frozenTrend []string
	pricing     map[string]float64
	workDir     string
	// comparing is true when the comparison panel is active.
	comparing bool
	// comparisonResult holds the pairwise comparison between the top-2 models.
	comparisonResult benchmark.ModelComparison
	// comparisonRuns holds all runs for the selected agent sorted by CompositeScore DESC.
	// Used by the ranked comparison panel to show all models with visual bars.
	comparisonRuns []store.BenchmarkRun
	// statusMsg is a transient message shown in the view (e.g. "Comparison requires 2+ models").
	statusMsg string
	// showNoData controls whether agents with no benchmark data are shown.
	// Default false — hidden to reduce clutter. Toggle with 'h'.
	showNoData bool
}

// NewBenchmarkModel creates a BenchmarkModel wired to the given BenchmarkStore.
// dataDir is the Metronous data directory (e.g. ~/.metronous/data); pricing is
// loaded from dataDir/../thresholds.json. Pass an empty string to disable pricing.
// workDir is used for project-level agent discovery; pass os.Getwd() from the caller.
func NewBenchmarkModel(bs store.BenchmarkStore, dataDir string, workDir string) BenchmarkModel {
	return BenchmarkModel{
		bs:      bs,
		loading: true,
		pricing: loadModelPricing(dataDir),
		agents:  discovery.DiscoverAgents(workDir),
		workDir: workDir,
	}
}

// Init returns the initial fetch command and starts the auto-refresh ticker.
func (m BenchmarkModel) Init() tea.Cmd {
	return tea.Batch(
		m.fetchRuns(),
		tea.Tick(benchmarkRefreshInterval, func(t time.Time) tea.Msg {
			return benchmarkTickMsg{t: t}
		}),
	)
}

// Update handles data, tick, and key messages.
func (m BenchmarkModel) Update(msg tea.Msg) (BenchmarkModel, tea.Cmd) {
	switch msg := msg.(type) {
	case benchmarkTickMsg:
		// Schedule next tick and refresh data.
		return m, tea.Batch(
			tea.Tick(benchmarkRefreshInterval, func(t time.Time) tea.Msg {
				return benchmarkTickMsg{t: t}
			}),
			m.fetchRuns(),
		)

	case BenchmarkDataMsg:
		m.loading = false
		m.err = msg.Err
		if msg.Err == nil {
			// Enforce page size — the view always renders at most maxBenchmarkRows rows.
			runs := msg.Runs
			if len(runs) > maxBenchmarkRows {
				runs = runs[:maxBenchmarkRows]
			}
			m.runs = runs
			if msg.TypeByID != nil {
				m.typeByID = msg.TypeByID
			}
			if msg.TrendByID != nil {
				m.trendByID = msg.TrendByID
			}
			// Clamp cursor to actual result size.
			if m.cursor >= len(m.runs) {
				if len(m.runs) > 0 {
					m.cursor = len(m.runs) - 1
				} else {
					m.cursor = 0
				}
			}
		}
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			// Move selection one row up within the current page.
			if m.cursor > 0 {
				m.cursor--
			}
			// Unfreeze detail so it follows the cursor.
			m.detailFrozen = false
		case "down", "j":
			// Move selection one row down within the current page.
			if m.cursor < len(m.runs)-1 {
				m.cursor++
			}
			// Unfreeze detail so it follows the cursor.
			m.detailFrozen = false
		case "pgdown":
			// Slide window toward older runs (increase pageOffset by one full page).
			m.pageOffset += maxBenchmarkRows
			m.cursor = 0
			m.detailFrozen = false
			return m, m.fetchRuns()
		case "pgup":
			// Slide window toward newer runs (decrease pageOffset by one full page).
			if m.pageOffset >= maxBenchmarkRows {
				m.pageOffset -= maxBenchmarkRows
			} else {
				m.pageOffset = 0
			}
			m.cursor = 0
			m.detailFrozen = false
			return m, m.fetchRuns()
		case "enter":
			// Freeze the detail panel on the currently selected run.
			if m.cursor >= 0 && m.cursor < len(m.runs) {
				m.detailFrozen = true
				m.frozenRun = m.runs[m.cursor]
				m.frozenTrend = m.trendByID[trendKey(m.frozenRun.AgentID, m.frozenRun.Model)]
			}
		case "esc", "escape":
			if m.comparing {
				// Close comparison panel first.
				m.comparing = false
			} else {
				// Unfreeze the detail panel.
				m.detailFrozen = false
			}
		case "h":
			// Toggle NO DATA row visibility.
			m.showNoData = !m.showNoData
			m.cursor = 0
			m.pageOffset = 0
			return m, m.fetchRuns()
		case "c":
			m.statusMsg = ""
			if m.cursor < 0 || m.cursor >= len(m.runs) {
				break
			}
			curRun := m.runs[m.cursor]
			// Collect all runs for this agent.
			var agentRuns []store.BenchmarkRun
			for _, r := range m.runs {
				if r.AgentID == curRun.AgentID && !isNoData(r) {
					agentRuns = append(agentRuns, r)
				}
			}
			if len(agentRuns) < 2 {
				m.comparing = false
				m.statusMsg = "Comparison requires 2+ models for this agent"
			} else {
				// Sort by composite score DESC.
				sort.Slice(agentRuns, func(i, j int) bool {
					return agentRuns[i].CompositeScore > agentRuns[j].CompositeScore
				})
				m.comparisonRuns = agentRuns
				// Pairwise comparison between top-2 for the recommendation sentence.
				m.comparisonResult = benchmark.CompareModels(agentRuns[0], agentRuns[1])
				m.comparing = true
			}
		}
	}
	return m, nil
}

// agentTypeOrder returns a sort priority for the given agent type.
// Primary agents come first (0), then subagent (1), then all (2), then built-in (3).
// Unknown types sort last (4).
func agentTypeOrder(t string) int {
	switch t {
	case "primary":
		return 0
	case "subagent":
		return 1
	case "all":
		return 2
	case "built-in":
		return 3
	default:
		return 4
	}
}

// fetchRuns returns a command that queries all discovered agents' latest runs,
// producing one row per (agent_id, model) combination.
// Agents with no data are included as placeholder rows (Verdict == "").
// The pageOffset field controls which window of sorted rows is returned.
func (m BenchmarkModel) fetchRuns() tea.Cmd {
	if m.bs == nil {
		return nil
	}
	// Snapshot the agent list, pageOffset, and showNoData at scheduling time
	// so the closure is self-contained.
	agents := m.agents
	pageOffset := m.pageOffset
	showNoData := m.showNoData
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// Build a map of agent types from the discovered agent list.
		typeByID := make(map[string]string, len(agents))
		for _, a := range agents {
			typeByID[a.ID] = a.Type
		}

		// Also pick up any agents that have runs in the DB but were not
		// discovered via config files (e.g. agents from old sessions).
		dbAgents, err := m.bs.ListAgents(ctx)
		if err != nil {
			return BenchmarkDataMsg{Err: err}
		}
		for _, id := range dbAgents {
			if _, found := typeByID[id]; !found {
				typeByID[id] = "primary" // default type for DB-only agents
			}
		}

		// Collect all known agent IDs (discovered + DB).
		knownAgents := make(map[string]bool)
		for _, a := range agents {
			knownAgents[a.ID] = true
		}
		for _, id := range dbAgents {
			knownAgents[id] = true
		}

		// Fetch distinct (agent_id, model) pairs from the DB to build per-model rows.
		agentModelPairs, err := m.bs.ListAgentModels(ctx)
		if err != nil {
			return BenchmarkDataMsg{Err: err}
		}

		// Track which agents have at least one (agent, model) row in DB.
		agentsWithData := make(map[string]bool, len(agentModelPairs))
		var all []store.BenchmarkRun
		for _, pair := range agentModelPairs {
			agentID, model := pair[0], pair[1]
			agentsWithData[agentID] = true
			run, err := m.bs.GetLatestRunByAgentModel(ctx, agentID, model)
			if err != nil || run == nil {
				continue
			}
			// Ensure the agent type is known.
			if _, found := typeByID[agentID]; !found {
				typeByID[agentID] = "primary"
			}
			all = append(all, *run)
		}

		// Add placeholder rows for discovered agents that have NO data at all.
		// Only shown when showNoData is true (toggled by 'h' key).
		if showNoData {
			for agentID := range knownAgents {
				if !agentsWithData[agentID] {
					all = append(all, store.BenchmarkRun{AgentID: agentID})
				}
			}
		}

		// Sort rows: primary → subagent → all → built-in,
		// then by agent_id, then by model within each agent.
		sort.Slice(all, func(i, j int) bool {
			ti := agentTypeOrder(typeByID[all[i].AgentID])
			tj := agentTypeOrder(typeByID[all[j].AgentID])
			if ti != tj {
				return ti < tj
			}
			if all[i].AgentID != all[j].AgentID {
				return all[i].AgentID < all[j].AgentID
			}
			return all[i].Model < all[j].Model
		})

		// Apply sliding-window pagination: slice out the current page.
		start := pageOffset
		if start > len(all) {
			start = len(all)
		}
		end := start + maxBenchmarkRows
		if end > len(all) {
			end = len(all)
		}
		page := all[start:end]

		// Fetch verdict trends for each (agent, model) in the current page (last 8 weeks).
		trendByID := make(map[string][]string, len(page))
		for _, run := range page {
			if isNoData(run) {
				continue
			}
			trend, err := m.bs.GetVerdictTrendByModel(ctx, run.AgentID, run.Model, 8)
			if err == nil {
				trendByID[trendKey(run.AgentID, run.Model)] = trend
			}
		}

		return BenchmarkDataMsg{Runs: page, TypeByID: typeByID, TrendByID: trendByID}
	}
}

// View renders the benchmark history tab.
func (m BenchmarkModel) View() string {
	var sb strings.Builder

	sb.WriteString(titleStyle.Render("Benchmark History") + "\n\n")

	if m.loading {
		sb.WriteString(dimStyle.Render("  Loading…") + "\n")
		return sb.String()
	}
	if m.err != nil {
		sb.WriteString(errStyle.Render(fmt.Sprintf("  Error: %v", m.err)) + "\n")
		return sb.String()
	}
	if len(m.runs) == 0 && len(m.agents) == 0 {
		sb.WriteString(dimStyle.Render("  No agents discovered and no benchmark runs yet.") + "\n")
		return sb.String()
	}
	if len(m.runs) == 0 {
		sb.WriteString(dimStyle.Render("  No benchmark runs yet. Run a benchmark to see history here.") + "\n")
		return sb.String()
	}

	// Header.
	sb.WriteString(renderRow(benchColNames, benchColWidths, headerStyle))
	sb.WriteString("\n")
	sb.WriteString(strings.Repeat("─", totalWidth(benchColWidths)) + "\n")

	// Data rows — m.runs already contains only the current page (maxBenchmarkRows rows max).
	// The cursor is a local index within this page.
	for i, run := range m.runs {
		agentType := m.typeByID[run.AgentID]
		row := formatBenchmarkRow(run, agentType, m.pricing)
		baseStyle := lipgloss.NewStyle()
		if i == m.cursor {
			baseStyle = cursorStyle
		}
		// Render columns before Score without special colour.
		// Score column is at index 3; render [0:3] plain, then score with color, then [4:verdictColIdx] plain.
		rendered := renderRow(row[:3], benchColWidths[:3], baseStyle)
		// Score column (index 3) with color coding.
		var scoreStyled string
		if !isNoData(run) && run.CompositeScore > 0 {
			scoreStyled = scoreStyle(run.CompositeScore).Inherit(baseStyle).Render(
				fmt.Sprintf("%-*s", benchColWidths[3], row[3]))
		} else {
			scoreStyled = baseStyle.Render(fmt.Sprintf("%-*s", benchColWidths[3], row[3]))
		}
		rendered += " " + scoreStyled
		// Columns between Score and Verdict (Accuracy, P95 Latency = indices 4..verdictColIdx-1).
		rendered += renderRow(row[4:verdictColIdx], benchColWidths[4:verdictColIdx], baseStyle)
		// Verdict column with colour — combine cursor background with verdict foreground.
		// verdictColIdx = 6 (Time, Agent, Type, Score, Accuracy, P95 Latency, Verdict, → Model, Savings)
		verdictCell := verdictStyle(run.Verdict).Inherit(baseStyle).Render(
			fmt.Sprintf("%-*s", benchColWidths[verdictColIdx], row[verdictColIdx]))
		rendered += verdictCell
		// → Model column (index 7).
		rendered += " " + baseStyle.Render(fmt.Sprintf("%-*s", benchColWidths[7], row[7]))
		// Savings column (index 8).
		rendered += " " + baseStyle.Render(fmt.Sprintf("%-*s", benchColWidths[8], row[8]))
		// Write the row directly — do NOT re-wrap with baseStyle.Render() as that
		// would strip the inner ANSI colour codes (verdict colour, etc.).
		sb.WriteString(rendered)
		sb.WriteString("\n")
	}

	// Pagination footer.
	sb.WriteString("\n")
	pageNum := m.pageOffset/maxBenchmarkRows + 1
	var noDataHint string
	if m.showNoData {
		noDataHint = "h: hide empty"
	} else {
		noDataHint = "h: show all"
	}
	footer := fmt.Sprintf("  %d entries shown  |  page %d  (↑↓ select, Enter freeze, c compare, %s)",
		len(m.runs), pageNum, noDataHint)
	sb.WriteString(dimStyle.Render(footer))
	sb.WriteString("\n")

	// Status message (e.g. single-model comparison attempt).
	if m.statusMsg != "" {
		sb.WriteString("\n")
		sb.WriteString(verdictInsufficientData.Render("  "+m.statusMsg) + "\n")
	}

	// Comparison panel (replaces detail panel when active).
	if m.comparing {
		sb.WriteString("\n")
		sb.WriteString(renderRankedComparison(m.comparisonRuns, m.comparisonResult))
		return sb.String()
	}

	// Detail panel for the selected run.
	// When detailFrozen, show the frozen snapshot — it won't change on background refresh.
	if m.cursor >= 0 && m.cursor < len(m.runs) {
		sb.WriteString("\n")
		var detailRun store.BenchmarkRun
		var trend []string
		if m.detailFrozen {
			detailRun = m.frozenRun
			trend = m.frozenTrend
			sb.WriteString(dimStyle.Render("  [Detail frozen — press Esc to unfreeze]") + "\n")
		} else {
			detailRun = m.runs[m.cursor]
			trend = m.trendByID[trendKey(detailRun.AgentID, detailRun.Model)]
		}
		sb.WriteString(renderDetailPanel(detailRun, m.pricing, trend))
	}

	return sb.String()
}

// barMaxWidth is the maximum number of bar characters (█) for the top-scored model.
const barMaxWidth = 20

// renderBar produces a visual bar string proportional to score relative to maxScore.
func renderBar(score, maxScore float64) string {
	if maxScore <= 0 {
		return ""
	}
	ratio := score / maxScore
	if ratio > 1 {
		ratio = 1
	}
	width := int(ratio * barMaxWidth)
	if width < 1 && score > 0 {
		width = 1
	}
	return strings.Repeat("█", width)
}

// renderRankedComparison renders the ranked model comparison panel with visual bars.
// runs must be sorted by CompositeScore DESC. cmp is the pairwise comparison between top-2.
func renderRankedComparison(runs []store.BenchmarkRun, cmp benchmark.ModelComparison) string {
	if len(runs) == 0 {
		return ""
	}

	var sb strings.Builder
	agentID := runs[0].AgentID

	divider := strings.Repeat("─", totalWidth(benchColWidths))
	sb.WriteString(dimStyle.Render(divider) + "\n")
	sb.WriteString(detailLabelStyle.Render(fmt.Sprintf(
		"Model Ranking: %s (%d models evaluated)", agentID, len(runs))) + "\n")
	sb.WriteString(dimStyle.Render(divider) + "\n")

	// The top model's score defines 100% bar width.
	maxScore := runs[0].CompositeScore

	// Best model's cost for computing cost deltas.
	bestCost := runs[0].TotalCostUSD

	for i, r := range runs {
		rank := i + 1
		model := shortenModel(r.Model)
		bar := renderBar(r.CompositeScore, maxScore)

		// Build the label after the bar.
		var label string
		if rank == 1 {
			label = "BEST"
		} else if r.Verdict == store.VerdictSwitch || r.Verdict == store.VerdictUrgentSwitch {
			label = "SWITCH"
		} else if r.Verdict == store.VerdictInsufficientData {
			label = "⚠ insufficient"
		} else if r.Verdict == store.VerdictKeep {
			label = "KEEP"
		}

		// Cost delta relative to best model.
		var costNote string
		if rank > 1 && bestCost > 0 && r.TotalCostUSD > 0 {
			costDeltaPct := ((r.TotalCostUSD - bestCost) / bestCost) * 100
			if costDeltaPct < -1 {
				costNote = fmt.Sprintf("%.0f%% cost", costDeltaPct)
			} else if costDeltaPct > 1 {
				costNote = fmt.Sprintf("+%.0f%% cost", costDeltaPct)
			}
		}

		// Combine label and cost note.
		var suffix string
		switch {
		case label != "" && costNote != "":
			suffix = label + "  " + costNote
		case label != "":
			suffix = label
		case costNote != "":
			suffix = costNote
		}

		// Colorize the rank line based on verdict.
		line := fmt.Sprintf("  #%-2d %-20s %4.2f  %-20s  %s",
			rank, model, r.CompositeScore, bar, suffix)

		switch {
		case rank == 1:
			sb.WriteString(verdictKeep.Render(line) + "\n")
		case r.Verdict == store.VerdictSwitch || r.Verdict == store.VerdictUrgentSwitch:
			sb.WriteString(verdictSwitch.Render(line) + "\n")
		case r.Verdict == store.VerdictInsufficientData:
			sb.WriteString(verdictInsufficientData.Render(line) + "\n")
		case r.Verdict == store.VerdictKeep:
			sb.WriteString(verdictKeep.Render(line) + "\n")
		default:
			sb.WriteString(verdictOther.Render(line) + "\n")
		}
	}

	sb.WriteString("\n")

	// Recommendation sentence from pairwise comparison of top-2.
	sb.WriteString(detailPanelStyle.Render("  "+cmp.Recommendation) + "\n")
	sb.WriteString("\n")
	sb.WriteString(dimStyle.Render("  Press Esc to return to detail view.") + "\n")

	return sb.String()
}

// renderDetailPanel renders the decision rationale panel for the selected run.
// trend is the verdict history for the agent (oldest first); pass nil if unavailable.
func renderDetailPanel(run store.BenchmarkRun, pricing map[string]float64, trend []string) string {
	var sb strings.Builder

	divider := strings.Repeat("─", totalWidth(benchColWidths))
	sb.WriteString(dimStyle.Render(divider) + "\n")
	sb.WriteString(detailLabelStyle.Render("Decision Rationale") + "\n")
	sb.WriteString(dimStyle.Render(divider) + "\n")

	// Handle NO_DATA placeholder rows.
	if isNoData(run) {
		writeDetailField(&sb, "Agent", run.AgentID)
		writeDetailField(&sb, "Status", "No benchmark runs recorded yet for this agent.")
		return sb.String()
	}

	// Verdict line: show switch arrow if applicable.
	verdictLine := string(run.Verdict)
	if (run.Verdict == store.VerdictSwitch || run.Verdict == store.VerdictUrgentSwitch) && run.RecommendedModel != "" {
		verdictLine = fmt.Sprintf("%s → %s", run.Verdict, run.RecommendedModel)
	}

	// Recommendation sentence based on verdict.
	var recommendation string
	switch run.Verdict {
	case store.VerdictKeep:
		recommendation = "This agent performs well with this model. Keep it."
	case store.VerdictSwitch:
		recommendation = "This agent underperforms with this model. Consider switching."
	case store.VerdictUrgentSwitch:
		recommendation = "CRITICAL: This agent is failing with this model. Switch immediately."
	case store.VerdictInsufficientData:
		recommendation = fmt.Sprintf("Not enough data yet (%d/50 samples). Keep using to gather more.", run.SampleSize)
	}

	// Cost savings for detail panel.
	_, savingsStr := computeSavings(run.Model, run.RecommendedModel, run.Verdict, pricing)

	// Format fields with aligned labels.
	writeDetailField(&sb, "Agent", run.AgentID)
	writeDetailField(&sb, "Model", run.Model)
	if run.CompositeScore > 0 {
		writeDetailField(&sb, "Score", fmt.Sprintf("%.2f", run.CompositeScore))
	}
	writeDetailField(&sb, "Verdict", verdictLine)
	if recommendation != "" {
		// Render recommendation with verdict-appropriate style.
		recStyle := verdictStyle(run.Verdict)
		var recPrefix string
		switch run.Verdict {
		case store.VerdictKeep:
			recPrefix = "  OK "
		case store.VerdictSwitch, store.VerdictUrgentSwitch:
			recPrefix = "  !! "
		case store.VerdictInsufficientData:
			recPrefix = "  .. "
		}
		sb.WriteString(recStyle.Render(recPrefix + recommendation))
		sb.WriteString("\n")
	}
	writeDetailField(&sb, "Cost", fmt.Sprintf("$%.2f  Savings: %s", run.TotalCostUSD, savingsStr))
	writeDetailField(&sb, "Samples", fmt.Sprintf("%d events", run.SampleSize))
	sb.WriteString("\n")
	writeDetailField(&sb, "Reason", run.DecisionReason)
	writeDetailField(&sb, "Context", evaluateAgentContext(run))

	// Trend line: show last N verdicts with direction indicator.
	if len(trend) > 0 {
		trendStr := formatVerdictTrend(trend)
		writeDetailField(&sb, "Trend", trendStr)
	}

	return sb.String()
}

// formatVerdictTrend formats a slice of verdict strings into a human-readable trend line.
// e.g. "SWITCH → SWITCH → KEEP → KEEP  (↑ improving)"
func formatVerdictTrend(trend []string) string {
	if len(trend) == 0 {
		return "-"
	}
	trendLine := strings.Join(trend, " → ")
	direction := trendDirection(trend)
	return fmt.Sprintf("%s  (%s)", trendLine, direction)
}

// verdictSeverity returns a numeric severity for a verdict (lower = better).
func verdictSeverity(v string) int {
	switch store.VerdictType(v) {
	case store.VerdictKeep:
		return 0
	case store.VerdictSwitch:
		return 2
	case store.VerdictUrgentSwitch:
		return 3
	default:
		return 1
	}
}

// trendDirection returns a direction indicator string for a slice of verdict strings.
// INSUFFICIENT_DATA at either endpoint is treated as a neutral sentinel — it does not
// imply improvement or degradation from data gaps.
func trendDirection(verdicts []string) string {
	if len(verdicts) < 2 {
		return "→ stable"
	}
	first := verdicts[0]
	last := verdicts[len(verdicts)-1]

	// Data gaps are neutral — don't signal improvement or degradation.
	if first == string(store.VerdictInsufficientData) || last == string(store.VerdictInsufficientData) {
		return "→ stable"
	}

	firstSev := verdictSeverity(first)
	lastSev := verdictSeverity(last)

	if lastSev < firstSev {
		return "↑ improving"
	}
	if lastSev > firstSev {
		return "↓ degrading"
	}
	return "→ stable"
}

// evaluateAgentContext returns a short qualitative assessment of whether the agent
// fulfilled its mission, based on its known role and available telemetry metrics.
func evaluateAgentContext(run store.BenchmarkRun) string {
	switch run.AgentID {
	case "sdd-orchestrator":
		// Mission: coordinate, never do work inline
		// Good: high tool_success (delegates correctly)
		// Bad: if tool success < 0.8, likely doing inline work
		if run.ToolSuccessRate >= 0.9 {
			return "Coordinating effectively — delegations succeeding at expected rate"
		} else if run.ToolSuccessRate >= 0.7 {
			return "Some delegation failures detected — may be attempting inline work"
		}
		return "High failure rate — orchestrator may be bypassing delegation pattern"

	case "sdd-apply":
		// Mission: implement code changes
		// Good: high tool success (edits, writes working)
		// Bad: low success means broken implementations
		if run.ToolSuccessRate >= 0.9 {
			return "Implementations landing correctly — code changes applied successfully"
		} else if run.ToolSuccessRate >= 0.7 {
			return "Some implementation failures — review task definitions for clarity"
		}
		return "High implementation failure rate — task definitions may be incomplete"

	case "sdd-explore":
		// Mission: investigate codebase and think through ideas
		// Good: high tool success (reads, searches working)
		// Check: sample size indicates depth of exploration
		if run.SampleSize >= 50 && run.ToolSuccessRate >= 0.9 {
			return "Deep exploration with high read success — investigations thorough"
		} else if run.ToolSuccessRate >= 0.8 {
			return "Adequate exploration — consider deeper codebase analysis"
		}
		return "Shallow exploration detected — may be missing critical context"

	case "sdd-verify":
		// Mission: validate implementation against specs
		// Good: high tool success (reads, comparisons working)
		if run.ToolSuccessRate >= 0.9 {
			return "Validation passing — spec compliance checks executing correctly"
		} else if run.ToolSuccessRate >= 0.7 {
			return "Some validation failures — specs may need clarification"
		}
		return "Validation failing frequently — implementation may not match specs"

	case "sdd-spec":
		if run.ToolSuccessRate >= 0.9 {
			return "Spec writing succeeding — requirements captured correctly"
		}
		return "Spec generation issues — proposal inputs may be incomplete"

	case "sdd-design":
		if run.ToolSuccessRate >= 0.9 {
			return "Design artifacts generated successfully"
		}
		return "Design generation issues — proposal may need more detail"

	case "sdd-propose":
		if run.ToolSuccessRate >= 0.9 {
			return "Proposals being created from explorations correctly"
		}
		return "Proposal failures — exploration output may be insufficient"

	case "sdd-tasks":
		if run.ToolSuccessRate >= 0.9 {
			return "Task breakdown succeeding — specs and designs well-structured"
		}
		return "Task breakdown failures — specs may be ambiguous"

	case "sdd-init":
		if run.ToolSuccessRate >= 0.9 {
			return "Bootstrap executing correctly"
		}
		return "Bootstrap failures — check project configuration"

	case "sdd-archive":
		if run.ToolSuccessRate >= 0.9 {
			return "Archiving completing correctly"
		}
		return "Archive failures — verify change artifacts are complete"

	default:
		if run.ToolSuccessRate >= 0.9 {
			return "Agent performing within normal parameters"
		}
		return "Performance below expected thresholds for this agent role"
	}
}

// writeDetailField writes a single label: value line to the string builder.
func writeDetailField(sb *strings.Builder, label, value string) {
	sb.WriteString(detailLabelStyle.Render(fmt.Sprintf("%-9s", label+":")))
	sb.WriteString(" ")
	sb.WriteString(detailPanelStyle.Render(value))
	sb.WriteString("\n")
}

// isNoData returns true when a BenchmarkRun is a placeholder (no real run data).
// A run is considered NO_DATA when RunAt is the zero time (never been run).
func isNoData(run store.BenchmarkRun) bool {
	return run.RunAt.IsZero()
}

// shortenModel abbreviates a provider/model string for compact display.
// "anthropic/claude-opus-4-6" → "opus-4-6", "openai/gpt-5.4" → "gpt-5.4".
func shortenModel(model string) string {
	if model == "" {
		return "-"
	}
	// Strip the provider prefix (everything up to and including the last "/").
	if idx := strings.LastIndex(model, "/"); idx >= 0 {
		model = model[idx+1:]
	}
	// Strip common prefixes that add noise.
	model = strings.TrimPrefix(model, "claude-")
	return model
}

// formatBenchmarkRow converts a BenchmarkRun into display columns.
// agentType is the type label for the Type column (primary/subagent/built-in/all).
// For NO_DATA rows, metric fields are rendered as "-".
func formatBenchmarkRow(run store.BenchmarkRun, agentType string, pricing map[string]float64) []string {
	if agentType == "" {
		agentType = "-"
	}

	// Handle placeholder rows (agent discovered but no runs yet).
	if isNoData(run) {
		return []string{"-", run.AgentID, agentType, "-", "-", "-", "NO DATA", "-", "-"}
	}

	date := run.RunAt.Local().Format("2006-01-02 15:04")
	accuracy := fmt.Sprintf("%.1f%%", run.Accuracy*100)
	p95 := fmt.Sprintf("%.0fms", run.P95LatencyMs)

	// Score column: show "—" when CompositeScore == 0 (not yet calculated).
	var scoreStr string
	if run.CompositeScore == 0 {
		scoreStr = "—"
	} else {
		scoreStr = fmt.Sprintf("%.2f", run.CompositeScore)
	}

	// Model column: always show the current model (shortened for display).
	modelCol := shortenModel(run.Model)

	// Savings column.
	_, savingsStr := computeSavings(run.Model, run.RecommendedModel, run.Verdict, pricing)

	return []string{date, run.AgentID, agentType, scoreStr, accuracy, p95, string(run.Verdict), modelCol, savingsStr}
}

// computeSavings returns the savings ratio (0.0–1.0) and a formatted string
// (e.g. "~45%") given the current and recommended model names.
// Returns (0, "-") when the calculation is not applicable or pricing is unknown.
func computeSavings(currentModel, recommendedModel string, verdict store.VerdictType, pricing map[string]float64) (float64, string) {
	if verdict != store.VerdictSwitch && verdict != store.VerdictUrgentSwitch {
		return 0, "-"
	}
	if recommendedModel == "" {
		return 0, "-"
	}
	if len(pricing) == 0 {
		return 0, "-"
	}
	currentPrice, ok1 := pricing[currentModel]
	recommendedPrice, ok2 := pricing[recommendedModel]
	if !ok1 || !ok2 || currentPrice <= 0 || recommendedPrice <= 0 {
		return 0, "-"
	}
	savings := (1 - recommendedPrice/currentPrice) * 100
	if savings <= 0 {
		return 0, "-"
	}
	return savings, fmt.Sprintf("~%.0f%%", savings)
}

// verdictStyle returns the lipgloss style for a verdict.
func verdictStyle(v store.VerdictType) lipgloss.Style {
	switch v {
	case store.VerdictKeep:
		return verdictKeep
	case store.VerdictSwitch:
		return verdictSwitch
	case store.VerdictUrgentSwitch:
		return verdictUrgent
	case store.VerdictInsufficientData:
		return verdictInsufficientData
	default:
		return verdictOther
	}
}
