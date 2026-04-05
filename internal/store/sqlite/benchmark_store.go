package sqlite

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/kiosvantra/metronous/internal/store"
)

// benchmarkSchema defines the DDL for benchmark.db.
const benchmarkSchema = `
-- Core benchmark run table (one row per run per agent per model)
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id                    TEXT PRIMARY KEY,
    run_at                INTEGER NOT NULL,
    run_kind              TEXT NOT NULL DEFAULT 'weekly',
    window_start          INTEGER NOT NULL DEFAULT 0,
    window_end            INTEGER NOT NULL DEFAULT 0,
    window_days           INTEGER NOT NULL DEFAULT 7,
    agent_id              TEXT NOT NULL,
    model                 TEXT NOT NULL,
    raw_model             TEXT NOT NULL DEFAULT '',
    accuracy              REAL NOT NULL DEFAULT 0.0,
    avg_latency_ms        REAL NOT NULL DEFAULT 0.0,
    p50_latency_ms        REAL NOT NULL DEFAULT 0.0,
    p95_latency_ms        REAL NOT NULL DEFAULT 0.0,
    p99_latency_ms        REAL NOT NULL DEFAULT 0.0,
    tool_success_rate     REAL NOT NULL DEFAULT 0.0,
    roi_score             REAL NOT NULL DEFAULT 0.0,
    total_cost_usd        REAL NOT NULL DEFAULT 0.0,
    sample_size           INTEGER NOT NULL DEFAULT 0,
    verdict               TEXT NOT NULL,
    recommended_model     TEXT NOT NULL DEFAULT '',
    decision_reason       TEXT NOT NULL DEFAULT '',
    artifact_path         TEXT NOT NULL DEFAULT '',
    avg_quality_score     REAL NOT NULL DEFAULT 0.0,
    avg_prompt_tokens     REAL NOT NULL DEFAULT 0.0,
    avg_completion_tokens REAL NOT NULL DEFAULT 0.0,
    avg_turn_ms           REAL NOT NULL DEFAULT 0.0,
    p95_turn_ms           REAL NOT NULL DEFAULT 0.0,
    composite_score       REAL NOT NULL DEFAULT 0.0,
    run_status            TEXT NOT NULL DEFAULT 'active'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_benchmark_agent_ts ON benchmark_runs(agent_id, run_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_run_at ON benchmark_runs(run_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_verdict ON benchmark_runs(verdict, run_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_agent_model ON benchmark_runs(agent_id, model, run_at DESC);
`

// benchmarkMigrations contains ALTER TABLE statements to apply to existing databases.
// Each migration is guarded by checking for the column's existence first (SQLite
// returns an error on duplicate column add, which we ignore).
const addAvgQualityScoreColumn = `ALTER TABLE benchmark_runs ADD COLUMN avg_quality_score REAL NOT NULL DEFAULT 0.0`
const addCompositeScoreColumn = `ALTER TABLE benchmark_runs ADD COLUMN composite_score REAL NOT NULL DEFAULT 0.0`
const addRunKindColumn = `ALTER TABLE benchmark_runs ADD COLUMN run_kind TEXT NOT NULL DEFAULT 'weekly'`
const addWindowStartColumn = `ALTER TABLE benchmark_runs ADD COLUMN window_start INTEGER NOT NULL DEFAULT 0`
const addWindowEndColumn = `ALTER TABLE benchmark_runs ADD COLUMN window_end INTEGER NOT NULL DEFAULT 0`
const addRawModelColumn = `ALTER TABLE benchmark_runs ADD COLUMN raw_model TEXT NOT NULL DEFAULT ''`
const addAvgPromptTokensColumn = `ALTER TABLE benchmark_runs ADD COLUMN avg_prompt_tokens REAL NOT NULL DEFAULT 0.0`
const addAvgCompletionTokensColumn = `ALTER TABLE benchmark_runs ADD COLUMN avg_completion_tokens REAL NOT NULL DEFAULT 0.0`
const addAvgTurnMsColumn = `ALTER TABLE benchmark_runs ADD COLUMN avg_turn_ms REAL NOT NULL DEFAULT 0.0`
const addP95TurnMsColumn = `ALTER TABLE benchmark_runs ADD COLUMN p95_turn_ms REAL NOT NULL DEFAULT 0.0`
const addRunStatusColumn = `ALTER TABLE benchmark_runs ADD COLUMN run_status TEXT NOT NULL DEFAULT 'active'`

// BenchmarkStore is a SQLite-backed implementation of store.BenchmarkStore.
type BenchmarkStore struct {
	writeDB *sql.DB
	readDB  *sql.DB
	path    string
}

// Compile-time interface check.
var _ store.BenchmarkStore = (*BenchmarkStore)(nil)

// NewBenchmarkStore opens (or creates) the benchmark SQLite database at path,
// applies WAL pragmas, and runs schema migrations.
func NewBenchmarkStore(path string) (*BenchmarkStore, error) {
	writeDB, err := openDB(path)
	if err != nil {
		return nil, fmt.Errorf("open benchmark write connection: %w", err)
	}

	// Apply schema migrations.
	if err := ApplyBenchmarkMigrations(context.Background(), writeDB); err != nil {
		_ = writeDB.Close()
		return nil, err
	}

	// Open a separate read-pool connection.
	// For in-memory databases (:memory:) used in tests, share the write connection.
	var readDB *sql.DB
	if path == ":memory:" {
		readDB = writeDB
	} else {
		readDB, err = openReadDB(path)
		if err != nil {
			_ = writeDB.Close()
			return nil, fmt.Errorf("open benchmark read connection: %w", err)
		}
	}

	return &BenchmarkStore{
		writeDB: writeDB,
		readDB:  readDB,
		path:    path,
	}, nil
}

// ApplyBenchmarkMigrations creates all tables and indexes for benchmark.db,
// then applies any additive column migrations for existing databases.
// It is idempotent and safe to call at startup.
func ApplyBenchmarkMigrations(ctx context.Context, db *sql.DB) error {
	if _, err := db.ExecContext(ctx, benchmarkSchema); err != nil {
		return fmt.Errorf("apply benchmark schema: %w", err)
	}
	// Apply additive column migration — ignore "duplicate column name" errors from
	// databases that already have the column (e.g. newly created with the full schema).
	if _, err := db.ExecContext(ctx, addAvgQualityScoreColumn); err != nil {
		// SQLite returns "duplicate column name" as an error; this is expected for
		// fresh databases where the CREATE TABLE already includes the column.
		if !strings.Contains(err.Error(), "duplicate column name") {
			return fmt.Errorf("apply avg_quality_score migration: %w", err)
		}
	}
	if _, err := db.ExecContext(ctx, addCompositeScoreColumn); err != nil {
		if !strings.Contains(err.Error(), "duplicate column name") {
			return fmt.Errorf("apply composite_score migration: %w", err)
		}
	}
	// New column migrations for upstream alignment.
	for _, migration := range []struct {
		sql  string
		name string
	}{
		{addRunKindColumn, "run_kind"},
		{addWindowStartColumn, "window_start"},
		{addWindowEndColumn, "window_end"},
		{addRawModelColumn, "raw_model"},
		{addAvgPromptTokensColumn, "avg_prompt_tokens"},
		{addAvgCompletionTokensColumn, "avg_completion_tokens"},
		{addAvgTurnMsColumn, "avg_turn_ms"},
		{addP95TurnMsColumn, "p95_turn_ms"},
		{addRunStatusColumn, "run_status"},
	} {
		if _, err := db.ExecContext(ctx, migration.sql); err != nil {
			if !strings.Contains(err.Error(), "duplicate column name") {
				return fmt.Errorf("apply %s migration: %w", migration.name, err)
			}
		}
	}
	return nil
}

// SaveRun persists a benchmark run. If run.ID is empty, a UUID is generated.
func (bs *BenchmarkStore) SaveRun(ctx context.Context, run store.BenchmarkRun) error {
	if run.ID == "" {
		run.ID = uuid.New().String()
	}

	// Default RunKind for backward compatibility.
	runKind := run.RunKind
	if runKind == "" {
		runKind = store.RunKindWeekly
	}
	status := run.Status
	if status == "" {
		status = store.RunStatusActive
	}

	const q = `
		INSERT INTO benchmark_runs (
			id, run_at, run_kind, window_start, window_end, window_days,
			agent_id, model, raw_model,
			accuracy, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
			tool_success_rate, roi_score, total_cost_usd, sample_size,
			verdict, recommended_model, decision_reason, artifact_path,
			avg_quality_score, avg_prompt_tokens, avg_completion_tokens,
			avg_turn_ms, p95_turn_ms, composite_score, run_status
		) VALUES (
			?, ?, ?, ?, ?, ?,
			?, ?, ?,
			?, ?, ?, ?, ?,
			?, ?, ?, ?,
			?, ?, ?, ?,
			?, ?, ?,
			?, ?, ?, ?
		)`

	_, err := bs.writeDB.ExecContext(ctx, q,
		run.ID,
		run.RunAt.UTC().UnixMilli(),
		string(runKind),
		run.WindowStart.UTC().UnixMilli(),
		run.WindowEnd.UTC().UnixMilli(),
		run.WindowDays,
		run.AgentID,
		run.Model,
		run.RawModel,
		run.Accuracy,
		run.AvgLatencyMs,
		run.P50LatencyMs,
		run.P95LatencyMs,
		run.P99LatencyMs,
		run.ToolSuccessRate,
		run.ROIScore,
		run.TotalCostUSD,
		run.SampleSize,
		string(run.Verdict),
		run.RecommendedModel,
		run.DecisionReason,
		run.ArtifactPath,
		run.AvgQualityScore,
		run.AvgPromptTokens,
		run.AvgCompletionTokens,
		run.AvgTurnMs,
		run.P95TurnMs,
		run.CompositeScore,
		string(status),
	)
	if err != nil {
		return fmt.Errorf("save benchmark run: %w", err)
	}
	return nil
}

// MaxQueryLimit prevents OOM crashes from unbounded queries
const MaxQueryLimit = 10000

// GetRuns returns up to limit benchmark runs for the given agent, ordered by run_at DESC.
// If agentID is empty, runs for all agents are returned.
// Pass limit=0 for default cap (MaxQueryLimit), enforced to prevent OOM.
func (bs *BenchmarkStore) GetRuns(ctx context.Context, agentID string, limit int) ([]store.BenchmarkRun, error) {
	// Enforce maximum limit to prevent OOM with millions of rows
	if limit == 0 || limit > MaxQueryLimit {
		limit = MaxQueryLimit
	}

	var (
		conditions []string
		args       []interface{}
	)

	if agentID != "" {
		conditions = append(conditions, "agent_id = ?")
		args = append(args, agentID)
	}

	q := `SELECT id, run_at, run_kind, window_start, window_end, window_days,
		agent_id, model, raw_model,
		accuracy, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
		tool_success_rate, roi_score, total_cost_usd, sample_size,
		verdict, recommended_model, decision_reason, artifact_path,
		avg_quality_score, avg_prompt_tokens, avg_completion_tokens,
		avg_turn_ms, p95_turn_ms, composite_score, run_status
		FROM benchmark_runs`

	if len(conditions) > 0 {
		q += " WHERE " + strings.Join(conditions, " AND ")
	}
	q += " ORDER BY run_at DESC"
	q += " LIMIT ?"
	args = append(args, limit)

	rows, err := bs.readDB.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("get benchmark runs: %w", err)
	}
	defer rows.Close()

	return scanBenchmarkRuns(rows)
}

// QueryRuns retrieves benchmark runs matching the supplied filter criteria.
// Results are ordered by run_at DESC. Supports Offset/Limit for sliding-window pagination.
func (bs *BenchmarkStore) QueryRuns(ctx context.Context, query store.BenchmarkQuery) ([]store.BenchmarkRun, error) {
	limit := query.Limit
	if limit == 0 || limit > MaxQueryLimit {
		limit = MaxQueryLimit
	}

	var (
		conditions []string
		args       []interface{}
	)

	if query.AgentID != "" {
		conditions = append(conditions, "agent_id = ?")
		args = append(args, query.AgentID)
	}

	q := `SELECT id, run_at, run_kind, window_start, window_end, window_days,
		agent_id, model, raw_model,
		accuracy, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
		tool_success_rate, roi_score, total_cost_usd, sample_size,
		verdict, recommended_model, decision_reason, artifact_path,
		avg_quality_score, avg_prompt_tokens, avg_completion_tokens,
		avg_turn_ms, p95_turn_ms, composite_score, run_status
		FROM benchmark_runs`

	if len(conditions) > 0 {
		q += " WHERE " + strings.Join(conditions, " AND ")
	}
	q += " ORDER BY run_at DESC"
	q += " LIMIT ?"
	args = append(args, limit)
	if query.Offset > 0 {
		q += " OFFSET ?"
		args = append(args, query.Offset)
	}

	rows, err := bs.readDB.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("query benchmark runs: %w", err)
	}
	defer rows.Close()

	return scanBenchmarkRuns(rows)
}

// CountRuns returns the total number of benchmark runs matching the supplied filter.
func (bs *BenchmarkStore) CountRuns(ctx context.Context, query store.BenchmarkQuery) (int, error) {
	var (
		conditions []string
		args       []interface{}
	)

	if query.AgentID != "" {
		conditions = append(conditions, "agent_id = ?")
		args = append(args, query.AgentID)
	}

	q := `SELECT COUNT(*) FROM benchmark_runs`
	if len(conditions) > 0 {
		q += " WHERE " + strings.Join(conditions, " AND ")
	}

	var count int
	if err := bs.readDB.QueryRowContext(ctx, q, args...).Scan(&count); err != nil {
		return 0, fmt.Errorf("count benchmark runs: %w", err)
	}
	return count, nil
}

// GetLatestRun returns the most recent benchmark run for the agent, or nil if none exists.
func (bs *BenchmarkStore) GetLatestRun(ctx context.Context, agentID string) (*store.BenchmarkRun, error) {
	const q = `SELECT id, run_at, run_kind, window_start, window_end, window_days,
		agent_id, model, raw_model,
		accuracy, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
		tool_success_rate, roi_score, total_cost_usd, sample_size,
		verdict, recommended_model, decision_reason, artifact_path,
		avg_quality_score, avg_prompt_tokens, avg_completion_tokens,
		avg_turn_ms, p95_turn_ms, composite_score, run_status
		FROM benchmark_runs
		WHERE agent_id = ?
		ORDER BY run_at DESC
		LIMIT 1`

	row := bs.readDB.QueryRowContext(ctx, q, agentID)
	run, err := scanBenchmarkRun(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get latest benchmark run for %q: %w", agentID, err)
	}
	return run, nil
}

// ListAgents returns the distinct agent IDs that have at least one benchmark run.
func (bs *BenchmarkStore) ListAgents(ctx context.Context) ([]string, error) {
	const q = `SELECT DISTINCT agent_id FROM benchmark_runs ORDER BY agent_id`

	rows, err := bs.readDB.QueryContext(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("list benchmark agents: %w", err)
	}
	defer rows.Close()

	var agents []string
	for rows.Next() {
		var agentID string
		if err := rows.Scan(&agentID); err != nil {
			return nil, fmt.Errorf("scan agent id: %w", err)
		}
		agents = append(agents, agentID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate agent rows: %w", err)
	}
	return agents, nil
}

// Checkpoint performs a WAL checkpoint to prevent unbounded WAL file growth.
// This should be called before Close during graceful shutdown.
func (bs *BenchmarkStore) Checkpoint() error {
	if bs.writeDB == nil {
		return nil
	}
	_, err := bs.writeDB.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	return err
}

// Close releases all database connections.
func (bs *BenchmarkStore) Close() error {
	var errs []string
	if bs.readDB != nil && bs.readDB != bs.writeDB {
		if err := bs.readDB.Close(); err != nil {
			errs = append(errs, "read db: "+err.Error())
		}
	}
	if bs.writeDB != nil {
		if err := bs.writeDB.Close(); err != nil {
			errs = append(errs, "write db: "+err.Error())
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("close benchmark store: %s", strings.Join(errs, "; "))
	}
	return nil
}

// scanBenchmarkRuns reads rows from a query into a slice of BenchmarkRun structs.
func scanBenchmarkRuns(rows *sql.Rows) ([]store.BenchmarkRun, error) {
	var runs []store.BenchmarkRun
	for rows.Next() {
		var (
			runAtMs      int64
			windowStartMs int64
			windowEndMs  int64
			runKind      string
			verdict      string
			status       string
			run          store.BenchmarkRun
		)
		err := rows.Scan(
			&run.ID,
			&runAtMs,
			&runKind,
			&windowStartMs,
			&windowEndMs,
			&run.WindowDays,
			&run.AgentID,
			&run.Model,
			&run.RawModel,
			&run.Accuracy,
			&run.AvgLatencyMs,
			&run.P50LatencyMs,
			&run.P95LatencyMs,
			&run.P99LatencyMs,
			&run.ToolSuccessRate,
			&run.ROIScore,
			&run.TotalCostUSD,
			&run.SampleSize,
			&verdict,
			&run.RecommendedModel,
			&run.DecisionReason,
			&run.ArtifactPath,
			&run.AvgQualityScore,
			&run.AvgPromptTokens,
			&run.AvgCompletionTokens,
			&run.AvgTurnMs,
			&run.P95TurnMs,
			&run.CompositeScore,
			&status,
		)
		if err != nil {
			return nil, fmt.Errorf("scan benchmark run row: %w", err)
		}
		run.RunAt = time.UnixMilli(runAtMs).UTC()
		run.RunKind = store.RunKindType(runKind)
		if windowStartMs > 0 {
			run.WindowStart = time.UnixMilli(windowStartMs).UTC()
		}
		if windowEndMs > 0 {
			run.WindowEnd = time.UnixMilli(windowEndMs).UTC()
		}
		run.Verdict = store.VerdictType(verdict)
		run.Status = store.RunStatus(status)
		runs = append(runs, run)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate benchmark run rows: %w", err)
	}
	return runs, nil
}

// rowScanner is a common interface for *sql.Row and *sql.Rows to allow reuse of scan logic.
type rowScanner interface {
	Scan(dest ...interface{}) error
}

// scanBenchmarkRun reads a single row into a BenchmarkRun.
func scanBenchmarkRun(row rowScanner) (*store.BenchmarkRun, error) {
	var (
		runAtMs      int64
		windowStartMs int64
		windowEndMs  int64
		runKind      string
		verdict      string
		status       string
		run          store.BenchmarkRun
	)
	err := row.Scan(
		&run.ID,
		&runAtMs,
		&runKind,
		&windowStartMs,
		&windowEndMs,
		&run.WindowDays,
		&run.AgentID,
		&run.Model,
		&run.RawModel,
		&run.Accuracy,
		&run.AvgLatencyMs,
		&run.P50LatencyMs,
		&run.P95LatencyMs,
		&run.P99LatencyMs,
		&run.ToolSuccessRate,
		&run.ROIScore,
		&run.TotalCostUSD,
		&run.SampleSize,
		&verdict,
		&run.RecommendedModel,
		&run.DecisionReason,
		&run.ArtifactPath,
		&run.AvgQualityScore,
		&run.AvgPromptTokens,
		&run.AvgCompletionTokens,
		&run.AvgTurnMs,
		&run.P95TurnMs,
		&run.CompositeScore,
		&status,
	)
	if err != nil {
		return nil, err
	}
	run.RunAt = time.UnixMilli(runAtMs).UTC()
	run.RunKind = store.RunKindType(runKind)
	if windowStartMs > 0 {
		run.WindowStart = time.UnixMilli(windowStartMs).UTC()
	}
	if windowEndMs > 0 {
		run.WindowEnd = time.UnixMilli(windowEndMs).UTC()
	}
	run.Verdict = store.VerdictType(verdict)
	run.Status = store.RunStatus(status)
	return &run, nil
}

// ListAgentModels returns the distinct (agent_id, model) pairs that have at least one benchmark run.
func (bs *BenchmarkStore) ListAgentModels(ctx context.Context) ([][2]string, error) {
	const q = `SELECT DISTINCT agent_id, model FROM benchmark_runs ORDER BY agent_id, model`

	rows, err := bs.readDB.QueryContext(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("list agent-model pairs: %w", err)
	}
	defer rows.Close()

	var pairs [][2]string
	for rows.Next() {
		var agentID, model string
		if err := rows.Scan(&agentID, &model); err != nil {
			return nil, fmt.Errorf("scan agent-model pair: %w", err)
		}
		pairs = append(pairs, [2]string{agentID, model})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate agent-model rows: %w", err)
	}
	return pairs, nil
}

// GetLatestRunByAgentModel returns the most recent benchmark run for a specific
// (agent_id, model) combination, or nil if none exists.
func (bs *BenchmarkStore) GetLatestRunByAgentModel(ctx context.Context, agentID, model string) (*store.BenchmarkRun, error) {
	const q = `SELECT id, run_at, run_kind, window_start, window_end, window_days,
		agent_id, model, raw_model,
		accuracy, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
		tool_success_rate, roi_score, total_cost_usd, sample_size,
		verdict, recommended_model, decision_reason, artifact_path,
		avg_quality_score, avg_prompt_tokens, avg_completion_tokens,
		avg_turn_ms, p95_turn_ms, composite_score, run_status
		FROM benchmark_runs
		WHERE agent_id = ? AND model = ?
		ORDER BY run_at DESC
		LIMIT 1`

	row := bs.readDB.QueryRowContext(ctx, q, agentID, model)
	run, err := scanBenchmarkRun(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get latest benchmark run for %q/%q: %w", agentID, model, err)
	}
	return run, nil
}

// GetVerdictTrendByModel returns the last N weekly verdicts for a specific
// (agent_id, model) combination, ordered oldest first.
func (bs *BenchmarkStore) GetVerdictTrendByModel(ctx context.Context, agentID, model string, weeks int) ([]string, error) {
	if weeks <= 0 {
		return nil, nil
	}
	const q = `SELECT verdict FROM benchmark_runs
		WHERE agent_id = ? AND model = ?
		ORDER BY run_at DESC
		LIMIT ?`

	rows, err := bs.readDB.QueryContext(ctx, q, agentID, model, weeks)
	if err != nil {
		return nil, fmt.Errorf("get verdict trend for %q/%q: %w", agentID, model, err)
	}
	defer rows.Close()

	var verdicts []string
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			return nil, fmt.Errorf("scan verdict: %w", err)
		}
		verdicts = append(verdicts, v)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate verdict rows: %w", err)
	}

	// Reverse to get oldest-first order.
	for i, j := 0, len(verdicts)-1; i < j; i, j = i+1, j-1 {
		verdicts[i], verdicts[j] = verdicts[j], verdicts[i]
	}
	return verdicts, nil
}

// GetVerdictTrend returns the last N weekly verdicts for the given agent, ordered oldest first.
// Returns an empty slice if the agent has no runs or fewer than requested.
func (bs *BenchmarkStore) GetVerdictTrend(ctx context.Context, agentID string, weeks int) ([]string, error) {
	if weeks <= 0 {
		return nil, nil
	}
	// Fetch newest-first, then reverse for oldest-first order.
	const q = `SELECT verdict FROM benchmark_runs
		WHERE agent_id = ?
		ORDER BY run_at DESC
		LIMIT ?`

	rows, err := bs.readDB.QueryContext(ctx, q, agentID, weeks)
	if err != nil {
		return nil, fmt.Errorf("get verdict trend for %q: %w", agentID, err)
	}
	defer rows.Close()

	var verdicts []string
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			return nil, fmt.Errorf("scan verdict: %w", err)
		}
		verdicts = append(verdicts, v)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate verdict rows: %w", err)
	}

	// Reverse to get oldest-first order.
	for i, j := 0, len(verdicts)-1; i < j; i, j = i+1, j-1 {
		verdicts[i], verdicts[j] = verdicts[j], verdicts[i]
	}
	return verdicts, nil
}

// ListRunCycles returns the distinct week-start timestamps for all benchmark runs,
// ordered newest first.
func (bs *BenchmarkStore) ListRunCycles(ctx context.Context, loc *time.Location, limit, offset int) ([]time.Time, error) {
	if limit == 0 || limit > MaxQueryLimit {
		limit = MaxQueryLimit
	}
	const q = `SELECT DISTINCT run_at FROM benchmark_runs ORDER BY run_at DESC`
	rows, err := bs.readDB.QueryContext(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("list run cycles: %w", err)
	}
	defer rows.Close()

	seen := make(map[time.Time]struct{})
	var cycles []time.Time
	for rows.Next() {
		var runAtMs int64
		if err := rows.Scan(&runAtMs); err != nil {
			return nil, fmt.Errorf("scan run_at: %w", err)
		}
		t := time.UnixMilli(runAtMs).In(loc)
		// Find the Sunday start of the week.
		weekday := int(t.Weekday())
		sunday := t.AddDate(0, 0, -weekday)
		sunday = time.Date(sunday.Year(), sunday.Month(), sunday.Day(), 0, 0, 0, 0, loc)
		if _, ok := seen[sunday]; !ok {
			seen[sunday] = struct{}{}
			cycles = append(cycles, sunday)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate run cycle rows: %w", err)
	}

	// Apply offset and limit.
	if offset >= len(cycles) {
		return nil, nil
	}
	cycles = cycles[offset:]
	if len(cycles) > limit {
		cycles = cycles[:limit]
	}
	return cycles, nil
}

// QueryModelSummaries returns one aggregated row per model across all benchmark runs.
func (bs *BenchmarkStore) QueryModelSummaries(ctx context.Context) ([]store.BenchmarkModelSummary, error) {
	const q = `
		SELECT model,
			COUNT(*) as runs,
			SUM(accuracy * sample_size) / NULLIF(SUM(sample_size), 0) as avg_accuracy,
			SUM(p95_latency_ms * sample_size) / NULLIF(SUM(sample_size), 0) as avg_p95,
			0.0 as total_cost_usd,
			'' as last_verdict,
			MAX(run_at) as last_run_at
		FROM benchmark_runs
		GROUP BY model
		ORDER BY model`

	rows, err := bs.readDB.QueryContext(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("query model summaries: %w", err)
	}
	defer rows.Close()

	var summaries []store.BenchmarkModelSummary
	for rows.Next() {
		var (
			s           store.BenchmarkModelSummary
			lastRunAtMs int64
			verdict     string
		)
		if err := rows.Scan(&s.Model, &s.Runs, &s.AvgAccuracy, &s.AvgP95Ms, &s.TotalCostUSD, &verdict, &lastRunAtMs); err != nil {
			return nil, fmt.Errorf("scan model summary: %w", err)
		}
		s.LastVerdict = store.VerdictType(verdict)
		s.LastRunAt = time.UnixMilli(lastRunAtMs).UTC()
		summaries = append(summaries, s)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate model summary rows: %w", err)
	}

	// Fill in last verdict and cost from the most recent non-insufficient run per model.
	for i := range summaries {
		const vq = `SELECT verdict, total_cost_usd FROM benchmark_runs
			WHERE model = ? AND verdict != 'INSUFFICIENT_DATA'
			ORDER BY run_at DESC LIMIT 1`
		var v string
		var cost float64
		err := bs.readDB.QueryRowContext(ctx, vq, summaries[i].Model).Scan(&v, &cost)
		if err == nil {
			summaries[i].LastVerdict = store.VerdictType(v)
			summaries[i].TotalCostUSD = cost
		}
	}

	return summaries, nil
}

// QueryRunsInWindow returns all benchmark runs whose run_at falls within
// [since, until), ordered by run_at DESC.
func (bs *BenchmarkStore) QueryRunsInWindow(ctx context.Context, since, until time.Time) ([]store.BenchmarkRun, error) {
	q := `SELECT id, run_at, run_kind, window_start, window_end, window_days,
		agent_id, model, raw_model,
		accuracy, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
		tool_success_rate, roi_score, total_cost_usd, sample_size,
		verdict, recommended_model, decision_reason, artifact_path,
		avg_quality_score, avg_prompt_tokens, avg_completion_tokens,
		avg_turn_ms, p95_turn_ms, composite_score, run_status
		FROM benchmark_runs
		WHERE run_at >= ? AND run_at < ?
		ORDER BY run_at DESC
		LIMIT ?`

	rows, err := bs.readDB.QueryContext(ctx, q, since.UnixMilli(), until.UnixMilli(), MaxQueryLimit)
	if err != nil {
		return nil, fmt.Errorf("query runs in window: %w", err)
	}
	defer rows.Close()

	return scanBenchmarkRuns(rows)
}

// MarkSupersededRuns marks older intraweek runs of the same model as superseded.
func (bs *BenchmarkStore) MarkSupersededRuns(ctx context.Context, agentID string, newRunAt time.Time, newModel string, cycleStart, cycleEnd time.Time) error {
	const q = `UPDATE benchmark_runs SET run_status = 'superseded'
		WHERE agent_id = ?
		AND run_kind = 'intraweek'
		AND model = ?
		AND run_at < ?
		AND run_at >= ?
		AND run_at < ?`

	_, err := bs.writeDB.ExecContext(ctx, q,
		agentID, newModel,
		newRunAt.UnixMilli(),
		cycleStart.UnixMilli(),
		cycleEnd.UnixMilli(),
	)
	if err != nil {
		return fmt.Errorf("mark superseded runs: %w", err)
	}
	return nil
}
