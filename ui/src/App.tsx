import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// --- Types ---
type Mode = "simple_web_rag" | "mcp_basic" | "mcp_verified";
type Theme = "light" | "dark";

interface ResearchSource {
  title?: string;
  url?: string;
  snippet?: string;
  [key: string]: any;
}

interface ResearchMetrics {
  [key: string]: any;
}

interface ResearchResponse {
  answer: string;
  confidence?: number | null;
  sources: ResearchSource[];
  query: string;
  mode: string;
  metrics: ResearchMetrics;
  verification_details?: Record<string, any> | null;
}

interface CompareResponse {
  query: string;
  results: Record<string, ResearchResponse>;
}

// --- Benchmark types ---

interface BenchmarkSummary {
  mode: string;
  num_queries: number;
  avg_latency: number;
  min_latency: number;
  max_latency: number;
  avg_confidence: number | null;
  avg_sources: number;
  avg_answer_length: number;
  success_rate: number;
  total_time: number;
  // Accuracy metrics
  avg_f1?: number | null;
  avg_semantic?: number | null;
  avg_bert?: number | null;
  avg_rouge_l?: number | null;
}

interface BenchmarkResult {
  query: string;
  mode: string;
  latency_seconds: number;
  confidence: number | null;
  num_sources: number;
  answer_length: number;
  error?: string | null;

  // GT-related fields
  dataset?: string | null;
  example_id?: string | null;
  gt_f1?: number | null;
  gt_semantic?: number | null;
  gt_bert?: number | null;
  gt_rouge_l?: number | null;
}

interface BenchmarkResponse {
  summaries: Record<string, BenchmarkSummary>;
  results: BenchmarkResult[];
}

const API_BASE = "http://localhost:8000";

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"research" | "compare" | "benchmark">("research");
  const [theme, setTheme] = useState<Theme>("dark");

  const isDark = theme === "dark";

  // Refined Color Palette
  const colors = {
    background: isDark ? "#0f172a" : "#f8fafc",
    surface: isDark ? "#1e293b" : "#ffffff",
    surfaceSubtle: isDark ? "#334155" : "#f1f5f9",
    text: isDark ? "#f1f5f9" : "#0f172a",
    textMuted: isDark ? "#94a3b8" : "#64748b",
    border: isDark ? "#334155" : "#e2e8f0",
    primary: "#3b82f6",
    primaryHover: "#2563eb",
    primarySoft: isDark ? "rgba(59,130,246,0.15)" : "rgba(59,130,246,0.1)",
    codeBg: isDark ? "#020617" : "#f1f5f9",
    inputBg: isDark ? "#0f172a" : "#ffffff", // Darker input bg for contrast
  };

  const [query, setQuery] = useState("");
  const [maxPapers, setMaxPapers] = useState(10);
  const [selectedMode, setSelectedMode] = useState<Mode>("mcp_verified");

  // Results State
  const [researchResult, setResearchResult] = useState<ResearchResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);

  // Benchmark State
  const [benchmarkQueries, setBenchmarkQueries] = useState<string>(
    [
      "What are vision transformers?",
      "What are the latest advances in quantum error correction?",
      "How effective are GLP-1 agonists for weight loss?",
      "What are the current limitations of multimodal LLMs?",
    ].join("\n")
  );
  const [benchmarkModes, setBenchmarkModes] = useState<Mode[]>([
    "simple_web_rag",
    "mcp_basic",
    "mcp_verified",
  ]);
  const [benchmarkResponse, setBenchmarkResponse] = useState<BenchmarkResponse | null>(null);
  const [useAddHealthGT, setUseAddHealthGT] = useState<boolean>(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ================= API handlers =================

  const handleResearch = async () => {
    if (!query.trim()) {
      setError("Please enter a query.");
      return;
    }
    setLoading(true);
    setError(null);
    setResearchResult(null);

    try {
      const resp = await fetch(`${API_BASE}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          mode: selectedMode,
          max_papers: maxPapers,
        }),
      });

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed with ${resp.status}`);
      }

      const data: ResearchResponse = await resp.json();
      setResearchResult(data);
    } catch (e: any) {
      setError(e.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!query.trim()) {
      setError("Please enter a query.");
      return;
    }
    setLoading(true);
    setError(null);
    setCompareResult(null);

    try {
      const resp = await fetch(`${API_BASE}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          max_papers: maxPapers,
        }),
      });

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed with ${resp.status}`);
      }

      const data: CompareResponse = await resp.json();
      setCompareResult(data);
    } catch (e: any) {
      setError(e.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const handleBenchmark = async () => {
    const queries = benchmarkQueries
      .split("\n")
      .map((q) => q.trim())
      .filter(Boolean);

    if (!useAddHealthGT && !queries.length) {
      setError("Please enter at least one benchmark query.");
      return;
    }

    setLoading(true);
    setError(null);
    setBenchmarkResponse(null);

    const payload: any = {
      modes: benchmarkModes.length ? benchmarkModes : null,
      max_papers: maxPapers,
    };

    if (useAddHealthGT) {
      payload.eval_set = "addhealth";
    } else {
      payload.queries = queries;
    }

    try {
      const resp = await fetch(`${API_BASE}/benchmark`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed with ${resp.status}`);
      }

      const data: BenchmarkResponse = await resp.json();
      setBenchmarkResponse(data);
    } catch (e: any) {
      setError(e.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const toggleBenchmarkMode = (mode: Mode) =>
    setBenchmarkModes((prev) =>
      prev.includes(mode) ? prev.filter((m) => m !== mode) : [...prev, mode]
    );

  // ================= helpers =================

  const renderSources = (sources: ResearchSource[]) => {
    if (!sources?.length)
      return <p style={{ color: colors.textMuted }}>No sources returned.</p>;
    return (
      <ul style={{ paddingLeft: "1.2rem", marginTop: "0.4rem" }}>
        {sources.map((s, idx) => (
          <li key={idx} style={{ marginBottom: "0.5rem" }}>
            <strong>{s.title || `Source ${idx + 1}`}</strong>
            {s.url && (
              <>
                {" — "}
                <a
                  href={s.url}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: colors.primary }}
                >
                  {s.url}
                </a>
              </>
            )}
            {s.snippet && (
              <p style={{ margin: "0.25rem 0 0", fontSize: "0.9rem", color: colors.textMuted }}>
                {s.snippet}
              </p>
            )}
          </li>
        ))}
      </ul>
    );
  };

  const renderMetrics = (metrics?: ResearchMetrics) => {
    if (!metrics || Object.keys(metrics).length === 0) return null;
    return (
      <details style={{ marginTop: "1rem" }}>
        <summary style={{ cursor: "pointer", color: colors.textMuted, fontSize: "0.9rem" }}>
          Metrics
        </summary>
        <pre
          style={{
            marginTop: "0.5rem",
            padding: "0.7rem",
            background: colors.codeBg,
            borderRadius: "0.6rem",
            fontSize: "0.85rem",
            overflowX: "auto",
            color: colors.text,
          }}
        >
          {JSON.stringify(metrics, null, 2)}
        </pre>
      </details>
    );
  };

  const renderVerificationDetails = (verification?: Record<string, any> | null) => {
    if (!verification) return null;
    return (
      <details style={{ marginTop: "0.75rem" }}>
        <summary style={{ cursor: "pointer", color: colors.textMuted, fontSize: "0.9rem" }}>
          Verification details
        </summary>
        <pre
          style={{
            marginTop: "0.5rem",
            padding: "0.7rem",
            background: colors.codeBg,
            borderRadius: "0.6rem",
            fontSize: "0.85rem",
            overflowX: "auto",
            color: colors.text,
          }}
        >
          {JSON.stringify(verification, null, 2)}
        </pre>
      </details>
    );
  };

  const TabButton: React.FC<{ id: "research" | "compare" | "benchmark"; label: string }> = ({
    id,
    label,
  }) => {
    const active = activeTab === id;
    return (
      <button
        onClick={() => setActiveTab(id)}
        style={{
          padding: "0.6rem 1.2rem",
          border: "none",
          background: "transparent",
          cursor: "pointer",
          fontWeight: active ? 600 : 400,
          color: active ? colors.text : colors.textMuted,
          borderBottom: active ? `2px solid ${colors.primary}` : "2px solid transparent",
          transition: "all 0.2s",
        }}
      >
        {label}
      </button>
    );
  };

  const ThemeToggle = () => (
    <button
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.35rem",
        padding: "0.35rem 0.7rem",
        borderRadius: "999px",
        border: `1px solid ${colors.border}`,
        background: colors.surface,
        color: colors.text,
        fontSize: "0.8rem",
        cursor: "pointer",
      }}
    >
      {theme === "dark" ? "🌙 Dark" : "☀️ Light"}
    </button>
  );

  const ControlGroup: React.FC<{ label: string; children: React.ReactNode }> = ({
    label,
    children,
  }) => (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
      <label
        style={{
          fontSize: "0.75rem",
          fontWeight: 600,
          color: colors.textMuted,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        {label}
      </label>
      {children}
    </div>
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        backgroundColor: colors.background,
        color: colors.text,
        fontFamily:
          '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      }}
    >
      <div
        style={{
          maxWidth: "1100px",
          margin: "0 auto",
          padding: "2rem 1.5rem",
        }}
      >
        {/* Header */}
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: "2rem",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <h1
                style={{
                  fontSize: "1.8rem",
                  margin: 0,
                  fontWeight: 700,
                  letterSpacing: "-0.02em",
                }}
              >
                Multi-Mode MCP Research Assistant
              </h1>
            </div>
            <p style={{ marginTop: "0.5rem", color: colors.textMuted, fontSize: "0.95rem" }}>
              Advanced research pipeline comparison and benchmarking tool.
            </p>
          </div>
          <ThemeToggle />
        </header>

        {/* Tab Navigation */}
        <div
          style={{
            borderBottom: `1px solid ${colors.border}`,
            marginBottom: "2rem",
            display: "flex",
            gap: "1rem",
          }}
        >
          <TabButton id="research" label="Research" />
          <TabButton id="compare" label="Compare Modes" />
          <TabButton id="benchmark" label="Benchmark" />
        </div>

        {/* Main Input Area (Redesigned) */}
        <section
          style={{
            marginBottom: "2rem",
          }}
        >
          {activeTab === "benchmark" ? (
            // Benchmark UI (Different layout due to complexity)
            <div
              style={{
                backgroundColor: colors.surface,
                border: `1px solid ${colors.border}`,
                borderRadius: "1rem",
                padding: "1.5rem",
                boxShadow: isDark
                  ? "0 4px 6px -1px rgba(0, 0, 0, 0.3)"
                  : "0 4px 6px -1px rgba(0, 0, 0, 0.05)",
              }}
            >
              <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>
                <div style={{ flex: 2, minWidth: "300px" }}>
                  <ControlGroup label="Benchmark Queries (one per line)">
                    <textarea
                      key="benchmark-queries"
                      rows={6}
                      value={benchmarkQueries}
                      onChange={(e) => setBenchmarkQueries(e.target.value)}
                      placeholder="Enter your queries here..."
                      disabled={useAddHealthGT}
                      style={{
                        width: "100%",
                        padding: "0.75rem",
                        borderRadius: "0.5rem",
                        border: `1px solid ${colors.border}`,
                        background: colors.inputBg,
                        color: colors.text,
                        resize: "vertical",
                        outline: "none",
                        fontFamily: "inherit",
                        opacity: useAddHealthGT ? 0.6 : 1,
                        boxSizing: "border-box",
                      }}
                    />
                  </ControlGroup>
                </div>
                <div
                  style={{
                    flex: 1,
                    minWidth: "220px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "1rem",
                  }}
                >
                  <ControlGroup label="Active Modes">
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "0.5rem",
                        marginTop: "0.25rem",
                      }}
                    >
                      {(["simple_web_rag", "mcp_basic", "mcp_verified"] as Mode[]).map((m) => (
                        <label
                          key={m}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            cursor: "pointer",
                            padding: "0.5rem",
                            borderRadius: "0.4rem",
                            border: `1px solid ${
                              benchmarkModes.includes(m) ? colors.primary : colors.border
                            }`,
                            backgroundColor: benchmarkModes.includes(m)
                              ? colors.primarySoft
                              : "transparent",
                          }}
                        >
                          <input
                            type="checkbox"
                            checked={benchmarkModes.includes(m)}
                            onChange={() => toggleBenchmarkMode(m)}
                            style={{ marginRight: "0.6rem" }}
                          />
                          <span style={{ fontSize: "0.9rem" }}>{m}</span>
                        </label>
                      ))}
                    </div>
                  </ControlGroup>

                  <ControlGroup label="Evaluation Dataset">
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        cursor: "pointer",
                        gap: "0.5rem",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={useAddHealthGT}
                        onChange={(e) => setUseAddHealthGT(e.target.checked)}
                      />
                      <span style={{ fontSize: "0.9rem" }}>Use AddHealth ground-truth</span>
                    </label>
                    <p
                      style={{
                        fontSize: "0.75rem",
                        color: colors.textMuted,
                        marginTop: "0.35rem",
                      }}
                    >
                      {useAddHealthGT
                        ? "Ignores manual queries and evaluates on the AddHealth QA dataset (F1 + cosine + BERTScore + ROUGE-L)."
                        : "Uses the queries above for a standard latency/quality benchmark."}
                    </p>
                  </ControlGroup>

                  <ControlGroup label="Papers">
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={maxPapers}
                      onChange={(e) => setMaxPapers(Number(e.target.value) || 1)}
                      style={{
                        width: "100%",
                        padding: "0.6rem",
                        borderRadius: "0.5rem",
                        border: `1px solid ${colors.border}`,
                        background: colors.inputBg,
                        color: colors.text,
                        boxSizing: "border-box",
                      }}
                    />
                  </ControlGroup>
                </div>
              </div>

              <div style={{ marginTop: "1.5rem", display: "flex", justifyContent: "flex-end" }}>
                <button
                  onClick={handleBenchmark}
                  disabled={loading}
                  style={{
                    padding: "0.75rem 2rem",
                    borderRadius: "0.5rem",
                    border: "none",
                    background: colors.primary,
                    color: "#ffffff",
                    fontWeight: 600,
                    cursor: "pointer",
                    boxShadow: "0 4px 12px rgba(37, 99, 235, 0.3)",
                    opacity: loading ? 0.7 : 1,
                  }}
                >
                  {loading
                    ? "Benchmarking..."
                    : useAddHealthGT
                    ? "Run AddHealth Benchmark"
                    : "Run Benchmark"}
                </button>
              </div>
            </div>
          ) : (
            // Research & Compare UI (The unified card design)
            <div
              style={{
                backgroundColor: colors.surface,
                border: `1px solid ${colors.border}`,
                borderRadius: "1rem",
                boxShadow: isDark
                  ? "0 4px 6px -1px rgba(0, 0, 0, 0.3)"
                  : "0 4px 6px -1px rgba(0, 0, 0, 0.05)",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
              }}
            >
              {/* Text Area Area */}
              <div style={{ padding: "0.5rem" }}>
                <textarea
                  rows={4}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="What would you like to research today?"
                  style={{
                    width: "100%",
                    padding: "1.5rem",
                    fontSize: "1.1rem",
                    border: "none",
                    background: "transparent",
                    color: colors.text,
                    resize: "none",
                    outline: "none",
                    fontFamily: "inherit",
                    boxSizing: "border-box",
                  }}
                />
              </div>

              {/* Bottom Control Bar */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "0.75rem 1.5rem 1.5rem 1.5rem",
                  flexWrap: "wrap",
                  gap: "1rem",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    gap: "1.5rem",
                    alignItems: "center",
                    flexWrap: "wrap",
                  }}
                >
                  {activeTab === "research" && (
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      <label
                        style={{
                          fontSize: "0.7rem",
                          fontWeight: 700,
                          color: colors.textMuted,
                          textTransform: "uppercase",
                        }}
                      >
                        Research Mode
                      </label>
                      <div style={{ position: "relative" }}>
                        <select
                          value={selectedMode}
                          onChange={(e) => setSelectedMode(e.target.value as Mode)}
                          style={{
                            padding: "0.5rem 2rem 0.5rem 0.75rem",
                            borderRadius: "0.5rem",
                            border: `1px solid ${colors.border}`,
                            background: colors.inputBg,
                            color: colors.text,
                            cursor: "pointer",
                            fontSize: "0.9rem",
                            appearance: "none",
                            minWidth: "160px",
                          }}
                        >
                          <option value="simple_web_rag">Simple Web RAG</option>
                          <option value="mcp_basic">MCP Basic</option>
                          <option value="mcp_verified">MCP Verified</option>
                        </select>
                        {/* Custom Arrow for select */}
                        <div
                          style={{
                            position: "absolute",
                            right: "10px",
                            top: "50%",
                            transform: "translateY(-50%)",
                            pointerEvents: "none",
                            fontSize: "0.8rem",
                            color: colors.textMuted,
                          }}
                        >
                          ▼
                        </div>
                      </div>
                    </div>
                  )}

                  <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                    <label
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 700,
                        color: colors.textMuted,
                        textTransform: "uppercase",
                      }}
                    >
                      Papers
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={maxPapers}
                      onChange={(e) => setMaxPapers(Number(e.target.value) || 1)}
                      style={{
                        width: "80px",
                        padding: "0.5rem 0.75rem",
                        borderRadius: "0.5rem",
                        border: `1px solid ${colors.border}`,
                        background: colors.inputBg,
                        color: colors.text,
                        fontSize: "0.9rem",
                      }}
                    />
                  </div>
                </div>

                <button
                  onClick={activeTab === "research" ? handleResearch : handleCompare}
                  disabled={loading}
                  style={{
                    padding: "0.6rem 1.5rem",
                    borderRadius: "0.5rem",
                    border: "none",
                    background: colors.primary,
                    color: "#ffffff",
                    fontWeight: 600,
                    fontSize: "0.95rem",
                    cursor: "pointer",
                    boxShadow: "0 4px 12px rgba(37, 99, 235, 0.3)",
                    opacity: loading ? 0.7 : 1,
                    transition: "transform 0.1s",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                  }}
                >
                  {loading ? "Processing..." : activeTab === "research" ? "Run Research ↵" : "Compare Modes"}
                </button>
              </div>
            </div>
          )}
        </section>

        {error && (
          <div
            style={{
              marginBottom: "2rem",
              padding: "1rem",
              borderRadius: "0.5rem",
              background: isDark ? "rgba(220, 38, 38, 0.2)" : "#fee2e2",
              color: isDark ? "#fecaca" : "#b91c1c",
              border: `1px solid ${isDark ? "#7f1d1d" : "#fca5a5"}`,
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* ========== Research tab Results ========== */}
        {activeTab === "research" && (
          <section>
            {researchResult ? (
              <div
                style={{
                  borderRadius: "1rem",
                  padding: "2rem",
                  background: colors.surface,
                  border: `1px solid ${colors.border}`,
                  marginBottom: "1.25rem",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: "0.5rem",
                    marginBottom: "1.5rem",
                    fontSize: "0.85rem",
                    color: colors.textMuted,
                    borderBottom: `1px solid ${colors.border}`,
                    paddingBottom: "1rem",
                  }}
                >
                  <div style={{ display: "flex", gap: "1rem" }}>
                    <span>
                      Mode:{" "}
                      <strong style={{ color: colors.text, textTransform: "uppercase" }}>
                        {researchResult.mode}
                      </strong>
                    </span>
                    {typeof researchResult.confidence === "number" && (
                      <span>
                        Confidence:{" "}
                        <strong style={{ color: colors.text }}>
                          {(researchResult.confidence * 100).toFixed(1)}%
                        </strong>
                      </span>
                    )}
                  </div>
                </div>

                <div style={{ fontSize: "1rem", lineHeight: 1.7 }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{researchResult.answer}</ReactMarkdown>
                </div>

                <div
                  style={{
                    marginTop: "2rem",
                    paddingTop: "1rem",
                    borderTop: `1px solid ${colors.border}`,
                  }}
                >
                  <details>
                    <summary style={{ cursor: "pointer", fontWeight: 600 }}>
                      View Sources ({researchResult.sources.length})
                    </summary>
                    <div style={{ marginTop: "1rem" }}>
                      {renderSources(researchResult.sources)}
                    </div>
                  </details>
                  {renderMetrics(researchResult.metrics)}
                  {renderVerificationDetails(researchResult.verification_details)}
                </div>
              </div>
            ) : (
              !loading && (
                <div
                  style={{
                    textAlign: "center",
                    padding: "4rem",
                    color: colors.textMuted,
                    border: `2px dashed ${colors.border}`,
                    borderRadius: "1rem",
                  }}
                >
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🔬</div>
                  <p>Enter a query above to begin your research.</p>
                </div>
              )
            )}
          </section>
        )}

        {/* ========== Compare tab Results ========== */}
        {activeTab === "compare" && (
          <section>
            {compareResult ? (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(350px, 1fr))",
                  gap: "1.5rem",
                }}
              >
                {Object.entries(compareResult.results).map(([mode, result]) => (
                  <div
                    key={mode}
                    style={{
                      borderRadius: "1rem",
                      padding: "1.5rem",
                      background: colors.surface,
                      border: `1px solid ${colors.border}`,
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.75rem",
                    }}
                  >
                    {/* Header */}
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        paddingBottom: "0.5rem",
                        borderBottom: `1px solid ${colors.border}`,
                      }}
                    >
                      <strong
                        style={{
                          textTransform: "uppercase",
                          fontSize: "0.9rem",
                          color: colors.primary,
                        }}
                      >
                        {mode}
                      </strong>
                      {typeof result.confidence === "number" && (
                        <span
                          style={{
                            fontSize: "0.8rem",
                            padding: "0.2rem 0.5rem",
                            borderRadius: "4px",
                            background: colors.surfaceSubtle,
                          }}
                        >
                          {(result.confidence * 100).toFixed(0)}% Conf.
                        </span>
                      )}
                    </div>

                    {/* NEW: compact metrics row */}
                    {result.metrics && (
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          fontSize: "0.8rem",
                          color: colors.textMuted,
                        }}
                      >
                        <span>
                          Latency:{" "}
                          {typeof result.metrics.latency_seconds === "number"
                            ? `${result.metrics.latency_seconds.toFixed(2)}s`
                            : "—"}
                        </span>
                        <span>
                          Answer length:{" "}
                          {typeof result.metrics.answer_length === "number"
                            ? result.metrics.answer_length
                            : result.answer?.length ?? "—"}
                        </span>
                      </div>
                    )}

                    {/* Answer */}
                    <div
                      style={{
                        fontSize: "0.95rem",
                        lineHeight: 1.6,
                        maxHeight: "400px",
                        overflowY: "auto",
                      }}
                    >
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown>
                    </div>

                    {/* Sources */}
                    <div
                      style={{
                        marginTop: "auto",
                        paddingTop: "1rem",
                        borderTop: `1px solid ${colors.border}`,
                      }}
                    >
                      <details>
                        <summary
                          style={{
                            cursor: "pointer",
                            fontSize: "0.85rem",
                            color: colors.textMuted,
                          }}
                        >
                          Sources ({result.sources.length})
                        </summary>
                        <div style={{ marginTop: "0.5rem" }}>
                          {renderSources(result.sources)}
                        </div>
                      </details>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              !loading && (
                <div
                  style={{
                    textAlign: "center",
                    padding: "4rem",
                    color: colors.textMuted,
                    border: `2px dashed ${colors.border}`,
                    borderRadius: "1rem",
                  }}
                >
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>⚖️</div>
                  <p>Run a comparison to see how different modes handle your query.</p>
                </div>
              )
            )}
          </section>
        )}

        {/* ========== Benchmark tab Results ========== */}
        {activeTab === "benchmark" && (
          <section>
            {benchmarkResponse ? (
              <>
                <h3 style={{ marginBottom: "1rem" }}>Summary by mode</h3>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                    gap: "1rem",
                    marginBottom: "2rem",
                  }}
                >
                  {Object.entries(benchmarkResponse.summaries).map(([mode, summary]) => (
                    <div
                      key={mode}
                      style={{
                        borderRadius: "0.75rem",
                        padding: "1.25rem",
                        background: colors.surface,
                        border: `1px solid ${colors.border}`,
                        fontSize: "0.9rem",
                      }}
                    >
                      <div
                        style={{
                          marginBottom: "0.5rem",
                          color: colors.primary,
                          fontWeight: 700,
                          textTransform: "uppercase",
                        }}
                      >
                        {mode}
                      </div>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "0.25rem",
                        }}
                      >
                        <span style={{ color: colors.textMuted }}>Avg Latency:</span>
                        <span>{summary.avg_latency.toFixed(2)}s</span>
                      </div>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "0.25rem",
                        }}
                      >
                        <span style={{ color: colors.textMuted }}>Success Rate:</span>
                        <span>{(summary.success_rate * 100).toFixed(1)}%</span>
                      </div>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "0.25rem",
                        }}
                      >
                        <span style={{ color: colors.textMuted }}>Avg Sources:</span>
                        <span>{summary.avg_sources.toFixed(1)}</span>
                      </div>
                      {summary.avg_f1 != null && (
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            marginTop: "0.35rem",
                          }}
                        >
                          <span style={{ color: colors.textMuted }}>A · Token F1 (GT):</span>
                          <span>{(summary.avg_f1 * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {summary.avg_semantic != null && (
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            marginTop: "0.25rem",
                          }}
                        >
                          <span style={{ color: colors.textMuted }}>B · Cosine Sim (GT):</span>
                          <span>{(summary.avg_semantic * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {summary.avg_bert != null && (
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            marginTop: "0.25rem",
                          }}
                        >
                          <span style={{ color: colors.textMuted }}>C · BERTScore F1:</span>
                          <span>{(summary.avg_bert * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {summary.avg_rouge_l != null && (
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            marginTop: "0.25rem",
                          }}
                        >
                          <span style={{ color: colors.textMuted }}>ROUGE-L F1:</span>
                          <span>{(summary.avg_rouge_l * 100).toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                <h3 style={{ marginBottom: "1rem" }}>Per-query results</h3>
                <div
                  style={{
                    borderRadius: "0.75rem",
                    padding: "1rem",
                    background: colors.surface,
                    border: `1px solid ${colors.border}`,
                    overflow: "hidden",
                  }}
                >
                  <pre
                    style={{
                      margin: 0,
                      background: colors.codeBg,
                      borderRadius: "0.5rem",
                      padding: "1rem",
                      fontSize: "0.85rem",
                      overflowX: "auto",
                      color: colors.text,
                    }}
                  >
                    {JSON.stringify(benchmarkResponse.results, null, 2)}
                  </pre>
                </div>
              </>
            ) : (
              !loading && (
                <div
                  style={{
                    textAlign: "center",
                    padding: "4rem",
                    color: colors.textMuted,
                    border: `2px dashed ${colors.border}`,
                    borderRadius: "1rem",
                  }}
                >
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>⏱️</div>
                  <p>Configure queries above or enable AddHealth GT to run a performance benchmark.</p>
                </div>
              )
            )}
          </section>
        )}
      </div>
    </div>
  );
};

export default App;