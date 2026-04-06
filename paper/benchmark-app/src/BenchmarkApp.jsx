import { useEffect, useMemo, useRef, useState } from "react";
import {
  BENCHMARK_CASES,
  DEFAULT_SWEEP_CONFIGS,
  DEFAULT_SYNTHETIC_CONFIG,
} from "./benchmarkCases";

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function useBenchmarkWorker() {
  const workerRef = useRef(null);

  useEffect(() => {
    workerRef.current = new Worker(new URL("./benchmarkWorker.js", import.meta.url), {
      type: "module",
    });
    return () => workerRef.current?.terminate();
  }, []);

  return workerRef;
}

function compareValue(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${value.toFixed(3)}${suffix}`;
}

function aggregateSummary(resultSet) {
  const rows = [];

  for (const sweepConfig of resultSet.sweepConfigs ?? []) {
    const matching = (resultSet.results ?? [])
      .flatMap((entry) => entry.sweepResults)
      .filter((entry) => entry.cacheConfig.id === sweepConfig.id);

    if (matching.length === 0) continue;

    const avg = (items, getter) =>
      items.reduce((sum, item) => sum + getter(item), 0) / Math.max(items.length, 1);

    rows.push({
      id: sweepConfig.id,
      label: sweepConfig.label,
      cases: matching.length,
      avgSpeedRatio: avg(matching, (entry) => entry.comparison.speedRatio),
      avgCompressionRatio: avg(
        matching,
        (entry) => entry.comparison.compressionRatio ?? 1,
      ),
      avgPrefixAgreementRatio: avg(
        matching,
        (entry) => entry.comparison.prefixAgreementRatio ?? 0,
      ),
      exactMatches: matching.filter((entry) => entry.comparison.exactMatch).length,
    });
  }

  return rows;
}

function SummaryTable({ resultSet }) {
  const rows = useMemo(() => aggregateSummary(resultSet), [resultSet]);

  if (rows.length === 0) return null;

  return (
    <section className="benchmark-card table-card">
      <h2>Sweep Summary</h2>
      <div className="table-wrap">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Config</th>
              <th>Cases</th>
              <th>Avg Speed Ratio</th>
              <th>Avg Compression</th>
              <th>Avg Prefix Agreement</th>
              <th>Exact Matches</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>{row.label}</td>
                <td>{row.cases}</td>
                <td>{compareValue(row.avgSpeedRatio, "x")}</td>
                <td>{compareValue(row.avgCompressionRatio, "x")}</td>
                <td>{compareValue(row.avgPrefixAgreementRatio * 100, "%")}</td>
                <td>{row.exactMatches}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CaseCard({ entry }) {
  return (
    <section className="benchmark-card case-card">
      <div className="case-header">
        <div>
          <h3>{entry.case.label}</h3>
          <p className="meta-line">{entry.case.description}</p>
        </div>
        <div className="case-chip">{entry.case.maxNewTokens} max new tokens</div>
      </div>

      <details className="prompt-details">
        <summary>Prompt</summary>
        <pre>{entry.case.prompt}</pre>
      </details>

      <div className="table-wrap">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Config</th>
              <th>Dynamic Avg</th>
              <th>Turbo Avg</th>
              <th>TTFT</th>
              <th>Decode Tok/s</th>
              <th>Compression</th>
              <th>Prefix Agree</th>
              <th>Exact</th>
            </tr>
          </thead>
          <tbody>
            {entry.sweepResults.map((sweep) => (
              <tr key={sweep.cacheConfig.id}>
                <td>{sweep.cacheConfig.label}</td>
                <td>{compareValue(sweep.dynamic.averageMs, " ms")}</td>
                <td>{compareValue(sweep.turboquant.averageMs, " ms")}</td>
                <td>{compareValue(sweep.turboquant.averageTtftMs, " ms")}</td>
                <td>{compareValue(sweep.turboquant.decodeTokensPerSecond)}</td>
                <td>{compareValue(sweep.comparison.compressionRatio, "x")}</td>
                <td>{compareValue(sweep.comparison.prefixAgreementRatio * 100, "%")}</td>
                <td>{sweep.comparison.exactMatch ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SyntheticRowsTable({ resultSet }) {
  const materialize = (resultSet.rows ?? []).filter((row) => row.benchmark === "materialize_only");
  const sweep = (resultSet.rows ?? []).filter((row) => row.benchmark === "cache_sweep");

  return (
    <div className="benchmark-results benchmark-results-stacked">
      <section className="benchmark-card table-card">
        <h2>Materialize-only</h2>
        <div className="table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Backend</th>
                <th>Seq</th>
                <th>Avg Materialize</th>
                <th>Median</th>
                <th>Compression</th>
                <th>Blocks</th>
              </tr>
            </thead>
            <tbody>
              {materialize.map((row) => (
                <tr key={`${row.backend}-${row.seq_len}`}>
                  <td>{row.backend}</td>
                  <td>{row.seq_len}</td>
                  <td>{compareValue(row.avg_materialize_ms, " ms")}</td>
                  <td>{compareValue(row.median_materialize_ms, " ms")}</td>
                  <td>{compareValue(row.compression_ratio, "x")}</td>
                  <td>{row.compressed_blocks ?? "n/a"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="benchmark-card table-card">
        <h2>Cache Sweep</h2>
        <div className="table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Backend</th>
                <th>Seq</th>
                <th>Prefill Update</th>
                <th>Prefill Materialize</th>
                <th>Decode Update/step</th>
                <th>Decode Materialize/step</th>
                <th>Compression</th>
              </tr>
            </thead>
            <tbody>
              {sweep.map((row) => (
                <tr key={`${row.backend}-${row.seq_len}`}>
                  <td>{row.backend}</td>
                  <td>{row.seq_len}</td>
                  <td>{compareValue(row.prefill_update_ms, " ms")}</td>
                  <td>{compareValue(row.prefill_materialize_ms, " ms")}</td>
                  <td>{compareValue(row.decode_update_per_step_ms, " ms")}</td>
                  <td>{compareValue(row.decode_materialize_per_step_ms, " ms")}</td>
                  <td>{compareValue(row.compression_ratio, "x")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default function BenchmarkApp() {
  const workerRef = useBenchmarkWorker();

  const [mode, setMode] = useState("browser");
  const [modelId, setModelId] = useState("onnx-community/gemma-4-E2B-it-ONNX");
  const [runs, setRuns] = useState(2);
  const [status, setStatus] = useState("Idle");
  const [loading, setLoading] = useState(false);
  const [resultSet, setResultSet] = useState(null);
  const [error, setError] = useState("");
  const [events, setEvents] = useState([]);
  const [caseSelection, setCaseSelection] = useState(
    Object.fromEntries(BENCHMARK_CASES.map((entry) => [entry.id, true])),
  );
  const [sweepText, setSweepText] = useState(JSON.stringify(DEFAULT_SWEEP_CONFIGS, null, 2));
  const [syntheticConfig, setSyntheticConfig] = useState(DEFAULT_SYNTHETIC_CONFIG);

  useEffect(() => {
    const worker = workerRef.current;
    if (!worker) {
      return undefined;
    }

    const onMessage = (event) => {
      const {
        status: nextStatus,
        message,
        info,
        result,
        error: nextError,
        modelId: readyModelId,
      } = event.data ?? {};

      switch (nextStatus) {
        case "progress":
          setLoading(true);
          setStatus(
            info?.status === "progress_total"
              ? `Loading ${Math.round(info.progress ?? 0)}%`
              : `${message ?? info?.status ?? "Loading model..."}`,
          );
          setEvents((current) =>
            [
              {
                id: crypto.randomUUID(),
                text: `${info?.status ?? "progress"} ${info?.file ?? info?.name ?? ""}`.trim(),
              },
              ...current,
            ].slice(0, 12),
          );
          break;
        case "phase":
          setLoading(true);
          setStatus(message);
          setEvents((current) =>
            [{ id: crypto.randomUUID(), text: message }, ...current].slice(0, 12),
          );
          break;
        case "ready":
          setLoading(false);
          setStatus(`Ready: ${readyModelId}`);
          break;
        case "complete":
          setLoading(false);
          setStatus("Benchmark complete");
          setResultSet(result);
          setError("");
          break;
        case "error":
          setLoading(false);
          setError(nextError);
          setStatus("Benchmark failed");
          break;
        default:
          break;
      }
    };

    worker.addEventListener("message", onMessage);
    return () => worker.removeEventListener("message", onMessage);
  }, [workerRef]);

  const selectedCases = useMemo(
    () => BENCHMARK_CASES.filter((entry) => caseSelection[entry.id]),
    [caseSelection],
  );

  const parsedSweepConfigs = useMemo(() => {
    try {
      const parsed = JSON.parse(sweepText);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return null;
    }
  }, [sweepText]);

  const parsedSyntheticConfigs = useMemo(() => {
    try {
      const parsed = JSON.parse(syntheticConfig.turboConfigs);
      if (!Array.isArray(parsed)) return null;
      const seqLens = syntheticConfig.seqLens
        .split(",")
        .map((value) => Number(value.trim()))
        .filter(Number.isFinite);
      return {
        seqLens,
        decodeSteps: Number(syntheticConfig.decodeSteps),
        layers: Number(syntheticConfig.layers),
        numKvHeads: Number(syntheticConfig.numKvHeads),
        headDim: Number(syntheticConfig.headDim),
        runs: Number(syntheticConfig.runs),
        warmupRuns: Number(syntheticConfig.warmupRuns),
        turboConfigs: parsed,
      };
    } catch {
      return null;
    }
  }, [syntheticConfig]);

  const requestLoad = () => {
    setError("");
    setLoading(true);
    setStatus("Loading benchmark model...");
    workerRef.current?.postMessage({
      type: "load",
      data: { modelId },
    });
  };

  const runBrowserBenchmark = () => {
    if (!parsedSweepConfigs || selectedCases.length === 0) {
      setError("Select at least one case and provide valid JSON sweep configs.");
      return;
    }
    setError("");
    setLoading(true);
    setStatus("Starting browser benchmark suite...");
    setResultSet(null);
    workerRef.current?.postMessage({
      type: "benchmark",
      data: {
        modelId,
        runs: Number(runs),
        cases: selectedCases,
        sweepConfigs: parsedSweepConfigs,
      },
    });
  };

  const runSyntheticBenchmark = () => {
    if (!parsedSyntheticConfigs) {
      setError("Synthetic benchmark config is invalid.");
      return;
    }
    setError("");
    setLoading(true);
    setStatus("Starting synthetic cache benchmarks...");
    setResultSet(null);
    workerRef.current?.postMessage({
      type: "synthetic",
      data: parsedSyntheticConfigs,
    });
  };

  return (
    <div className="benchmark-shell">
      <div className="benchmark-hero">
        <p className="eyebrow">Paper Harness</p>
        <h1>TurboQuant Gemma 4 Benchmark App</h1>
        <p className="subhead">
          Browser suite for Gemma 4 WebGPU plus synthetic cache benchmarks that isolate
          update and rematerialization costs. Everything needed for the paper now lives in
          this repo.
        </p>
      </div>

      <div className="benchmark-mode-toggle">
        <button
          className={mode === "browser" ? "primary-button compact" : "glass-button"}
          onClick={() => setMode("browser")}
        >
          Browser Suite
        </button>
        <button
          className={mode === "synthetic" ? "primary-button compact" : "glass-button"}
          onClick={() => setMode("synthetic")}
        >
          Synthetic Cache
        </button>
      </div>

      <div className="benchmark-grid">
        <section className="benchmark-card controls-card">
          {mode === "browser" ? (
            <>
              <label>
                <span>Model</span>
                <input value={modelId} onChange={(event) => setModelId(event.target.value)} />
              </label>
              <div className="control-row two-up">
                <label>
                  <span>Runs per point</span>
                  <input
                    type="number"
                    min="1"
                    value={runs}
                    onChange={(event) => setRuns(event.target.value)}
                  />
                </label>
              </div>

              <div className="case-picker">
                <h2>Benchmark Cases</h2>
                {BENCHMARK_CASES.map((entry) => (
                  <label className="check-row" key={entry.id}>
                    <input
                      type="checkbox"
                      checked={Boolean(caseSelection[entry.id])}
                      onChange={(event) =>
                        setCaseSelection((current) => ({
                          ...current,
                          [entry.id]: event.target.checked,
                        }))
                      }
                    />
                    <span>
                      <strong>{entry.label}</strong>
                      <small>{entry.description}</small>
                    </span>
                  </label>
                ))}
              </div>

              <label>
                <span>Sweep Config JSON</span>
                <textarea
                  rows={12}
                  value={sweepText}
                  onChange={(event) => setSweepText(event.target.value)}
                />
              </label>

              <div className="benchmark-actions">
                <button className="glass-button" onClick={requestLoad} disabled={loading}>
                  Load Model
                </button>
                <button className="primary-button" onClick={runBrowserBenchmark} disabled={loading}>
                  {loading ? "Running..." : "Run Browser Suite"}
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="control-row">
                <label>
                  <span>Seq Lens</span>
                  <input
                    value={syntheticConfig.seqLens}
                    onChange={(event) =>
                      setSyntheticConfig((current) => ({
                        ...current,
                        seqLens: event.target.value,
                      }))
                    }
                  />
                </label>
                <label>
                  <span>Decode Steps</span>
                  <input
                    type="number"
                    min="1"
                    value={syntheticConfig.decodeSteps}
                    onChange={(event) =>
                      setSyntheticConfig((current) => ({
                        ...current,
                        decodeSteps: event.target.value,
                      }))
                    }
                  />
                </label>
                <label>
                  <span>Runs</span>
                  <input
                    type="number"
                    min="1"
                    value={syntheticConfig.runs}
                    onChange={(event) =>
                      setSyntheticConfig((current) => ({
                        ...current,
                        runs: event.target.value,
                      }))
                    }
                  />
                </label>
              </div>
              <div className="control-row">
                <label>
                  <span>Warmup Runs</span>
                  <input
                    type="number"
                    min="0"
                    value={syntheticConfig.warmupRuns}
                    onChange={(event) =>
                      setSyntheticConfig((current) => ({
                        ...current,
                        warmupRuns: event.target.value,
                      }))
                    }
                  />
                </label>
                <label>
                  <span>Layers</span>
                  <input
                    type="number"
                    min="1"
                    value={syntheticConfig.layers}
                    onChange={(event) =>
                      setSyntheticConfig((current) => ({
                        ...current,
                        layers: event.target.value,
                      }))
                    }
                  />
                </label>
                <label>
                  <span>KV Heads</span>
                  <input
                    type="number"
                    min="1"
                    value={syntheticConfig.numKvHeads}
                    onChange={(event) =>
                      setSyntheticConfig((current) => ({
                        ...current,
                        numKvHeads: event.target.value,
                      }))
                    }
                  />
                </label>
              </div>
              <label>
                <span>Turbo Config JSON</span>
                <textarea
                  rows={14}
                  value={syntheticConfig.turboConfigs}
                  onChange={(event) =>
                    setSyntheticConfig((current) => ({
                      ...current,
                      turboConfigs: event.target.value,
                    }))
                  }
                />
              </label>
              <div className="benchmark-actions">
                <button className="primary-button" onClick={runSyntheticBenchmark} disabled={loading}>
                  {loading ? "Running..." : "Run Synthetic Benchmarks"}
                </button>
              </div>
            </>
          )}

          <div className="benchmark-actions">
            <button
              className="glass-button"
              onClick={() => resultSet && downloadJson("turboquant-benchmark.json", resultSet)}
              disabled={!resultSet}
            >
              Export JSON
            </button>
          </div>
          <p className="benchmark-status">{status}</p>
          {error ? <pre className="error-box">{error}</pre> : null}
        </section>

        <section className="benchmark-card telemetry-card">
          <h2>Recent Worker Events</h2>
          <div className="event-list">
            {events.length === 0 ? <div className="event-item muted">No events yet.</div> : null}
            {events.map((item) => (
              <div key={item.id} className="event-item">
                {item.text}
              </div>
            ))}
          </div>
          <p className="meta-line">
            Open this app in Chrome or Chromium with WebGPU enabled for the Gemma 4 browser suite.
          </p>
        </section>
      </div>

      {resultSet?.kind === "browser_suite" ? (
        <>
          <SummaryTable resultSet={resultSet} />
          <div className="benchmark-results benchmark-results-stacked">
            {resultSet.results.map((entry) => (
              <CaseCard key={entry.case.id} entry={entry} />
            ))}
          </div>
        </>
      ) : null}

      {resultSet?.kind === "synthetic_cache" ? <SyntheticRowsTable resultSet={resultSet} /> : null}
    </div>
  );
}
