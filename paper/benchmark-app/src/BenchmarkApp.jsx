import { useEffect, useMemo, useRef, useState } from "react";
import { BENCHMARK_CASES, DEFAULT_SWEEP_CONFIGS } from "./benchmarkCases";

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
    workerRef.current.postMessage({ type: "ping" });
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

      {entry.sweepResults.map((sweep) => (
        <details className="result-details" key={`${entry.case.id}-${sweep.cacheConfig.id}`}>
          <summary>{sweep.cacheConfig.label} outputs</summary>
          <div className="detail-grid">
            <div>
              <strong>Dynamic</strong>
              <pre>{sweep.dynamic.output}</pre>
            </div>
            <div>
              <strong>TurboQuant</strong>
              <pre>{sweep.turboquant.output}</pre>
            </div>
          </div>
        </details>
      ))}
    </section>
  );
}

export default function BenchmarkApp() {
  const workerRef = useBenchmarkWorker();

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

    const onError = (event) => {
      setLoading(false);
      setStatus("Worker failed");
      const message = event.message || "Unknown worker error";
      setError(message);
      setEvents((current) =>
        [{ id: crypto.randomUUID(), text: `worker error: ${message}` }, ...current].slice(0, 12),
      );
    };

    const onMessageError = () => {
      setLoading(false);
      setStatus("Worker message error");
      setError("Worker message could not be deserialized.");
    };

    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    worker.addEventListener("messageerror", onMessageError);
    return () => {
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      worker.removeEventListener("messageerror", onMessageError);
    };
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

  const requestLoad = () => {
    setError("");
    setLoading(true);
    setStatus("Loading benchmark model...");
    workerRef.current?.postMessage({
      type: "load",
      data: { modelId },
    });
  };

  const runBenchmark = () => {
    if (!parsedSweepConfigs || selectedCases.length === 0) {
      setError("Select at least one case and provide valid JSON sweep configs.");
      return;
    }
    setError("");
    setLoading(true);
    setStatus("Starting benchmark suite...");
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

  return (
    <div className="benchmark-shell">
      <div className="benchmark-hero">
        <p className="eyebrow">Chrome Benchmark</p>
        <h1>TurboQuant Paper Harness</h1>
        <p className="subhead">
          Runs a reproducible case suite in a browser worker, sweeps multiple TurboQuant
          configurations, and reports latency, TTFT, compression, decode throughput, and
          output agreement against the dynamic baseline.
        </p>
      </div>

      <div className="benchmark-grid">
        <section className="benchmark-card controls-card">
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
            <button className="primary-button" onClick={runBenchmark} disabled={loading}>
              {loading ? "Running..." : "Run Suite"}
            </button>
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
            Open this page in Chrome or Chromium with WebGPU enabled.
          </p>
        </section>
      </div>

      {resultSet ? (
        <>
          <SummaryTable resultSet={resultSet} />
          <div className="benchmark-results benchmark-results-stacked">
            {resultSet.results.map((entry) => (
              <CaseCard key={entry.case.id} entry={entry} />
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
