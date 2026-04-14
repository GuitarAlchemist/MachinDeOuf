# Mathematical tools for analysing programming-language repositories

> The catalog ix's `ix_code_catalog` MCP tool serves. Read this to pick the right specialist — then call the catalog tool from your agent to confirm it exists, what technique it uses, and which languages it supports.

Analysing source code is not one problem but six. Different mathematics, different tools, different output shapes. Picking the wrong tool for your problem — say, using a cyclomatic-complexity linter when you need proof of memory safety — wastes effort and gives a false sense of coverage.

This guide is the map. It splits the space into the six categories the catalog uses, describes what each category is for, names the tools ix knows about, and says explicitly where ix's own `ix_code_analyze` overlaps and where it doesn't. Every tool listed here lives in [`crates/ix-code/src/catalog.rs`](../../crates/ix-code/src/catalog.rs) as a queryable constant, so agents can cross-reference this doc against live data.

---

## How the catalog is queried

### From Rust

```rust
use ix_code::catalog::{all, by_language, by_category, ToolCategory, by_technique};

// Everything
let every_tool = all();

// Everything useful for Rust (includes language-agnostic tools)
let rust = by_language("rust");

// Every formal verifier
let formal = by_category(ToolCategory::FormalVerification);

// Every tool whose technique mentions "abstract interpretation"
let ai = by_technique("abstract interpretation");
```

### From an MCP client

Call the `ix_code_catalog` tool on the ix MCP server:

```json
{
  "language": "rust",
  "category": "formal_verification"
}
```

All three filter fields (`language`, `category`, `technique`) are optional; they combine with AND semantics. Omitting all of them returns the full catalog.

---

## The seven categories

### 1. Static analysis and complexity metrics

**What it is.** Everything that reads source code (or its AST / CFG / call graph) at rest and produces measurements, lints, or queryable facts. The math is discrete: graph theory, automata, abstract interpretation.

**When to use.** You want to measure how complex a function is, detect common bugs from syntactic patterns, find unused code, or query the AST for specific shapes.

**Tools in the catalog.** `Radon` (Python), `gocyclo` (Go), `CodeQL` (12 languages), `Include-gardener` (C/C++), `Astrée` (C/C++ abstract interpretation), `Polyspace` (C/C++/Ada, formal proofs of absence of errors), `MIRAI` (Rust abstract interpretation), plus ix's own `ix_code_analyze` which computes 20 metrics per file in 10 languages via a lightweight Halstead implementation.

**When ix_code_analyze is enough.** You want numbers — cyclomatic, cognitive, Halstead, SLOC, maintainability index — as feature vectors for an ML pipeline or a quick baseline scan. ix_code_analyze runs in milliseconds and returns ML-ready output.

**When to reach for something else.** If you want *bug detection*, use CodeQL. If you need *proofs of absence of runtime errors*, use Astrée or Polyspace. If you want *Rust-specific panic detection*, use MIRAI. ix_code_analyze measures; these tools reason.

### 2. Formal verification and logic

**What it is.** Tools that prove code correct against a formal specification using mathematical logic. Theorem provers, proof assistants, model checkers, SMT solvers. The math is deep: dependent type theory, first-order logic, symbolic execution, bounded model checking.

**When to use.** You need mathematical certainty that a specific property holds — a cryptographic routine is memory-safe, a concurrent data structure is linearisable, an overflow cannot happen, a specification is refined correctly.

**Tools in the catalog.** General: `Lean`, `Coq`, `Z3`. C/C++ model checkers: `CBMC`, `CPAchecker`. Rust-specific: `Kani` (MIR model checking), `Verus` (SMT-based, write proofs in Rust), `Aeneas` (translates Rust to functional form for Lean/Coq/F*), `Creusot` (deductive verifier using the Pearlite spec language). Symbolic execution: `Klee` (C/C++/Rust), `Haybale` (LLVM bitcode).

**When ix is not enough — and will never be.** ix has no theorem prover and no model checker. If you need proof, you need one of these. ix's planned R8 (QED-MCP) would bridge ix pipelines to Lean 4 / Kani, but the underlying reasoning still lives outside ix.

### 3. Safety and memory analysis

**What it is.** Tools that find undefined behaviour, memory corruption, data races, and concurrency bugs — typically at runtime or via instrumented interpretation. The math is operational: happens-before relations, shadow memory, thread interleaving enumeration.

**When to use.** You've written unsafe Rust, or C/C++ with manual memory management, or concurrent code with shared state. You want the ground truth on whether your code obeys the language's memory model.

**Tools in the catalog.** Rust-specific: `Miri` (instrumented MIR interpreter, the canonical Rust undefined-behaviour detector), `Loom` (exhaustive thread-interleaving model checker). Cross-language: `AddressSanitizer` and `ThreadSanitizer` (LLVM instrumentation, available for C/C++/Rust/Go).

**When ix is not enough.** ix does not run your code, and cannot reason about concurrency. These are the tools.

### 4. Statistical and behavioural analysis

**What it is.** Tools that treat the repository as a time-varying data source — git history, issue trackers, CI results — and apply statistical or ML techniques to find hotspots, risky files, and productivity patterns. The math is familiar: time-series analysis, clustering, regression, hotspot detection.

**When to use.** You want to know where to focus refactoring effort, which files are at risk, how the team's velocity is trending, or which modules are churning unhealthily.

**Tools in the catalog.** `CodeScene` (commercial behavioural analysis across 11+ languages), `git-trend` (open-source git history miner), plus ix's own `ix_git_log` (normalised commit cadence time series) and `ix_cargo_deps` (workspace dependency graph + feature matrix for Rust workspaces).

**When ix is enough.** If you want a repo health score you can feed into a pipeline, `ix_cargo_deps` + `ix_git_log` + `ix_kmeans` is the refactor-oracle pattern and runs in milliseconds on a 54-crate workspace. See [`examples/canonical-showcase/05-adversarial-refactor-oracle/`](../../examples/canonical-showcase/05-adversarial-refactor-oracle/) for the worked example.

**When to reach for CodeScene.** Commercial support, rich UI, multi-language coverage, issue-tracker integration, co-change analysis across large teams. ix's source adapters are primitives you compose; CodeScene is a product.

### 5. Documentation generation

**What it is.** Tools that mechanically produce documentation — API references, user manuals, architecture diagrams — from source code, doc comments, or repository metadata.

**When to use.** You have a crate, a library, or an API you want to publish. You want the docs to update automatically when the code changes. You want architecture diagrams that don't drift.

**Tools in the catalog — Rust first-class.**
- **`rustdoc`** is the gold standard. Extracts `///` comments, compiles every code example as a test so your how-to guides can never silently break, runs on every `cargo doc`. For ix itself, `cargo doc --workspace --no-deps --open` is the authoritative API reference for every crate.
- **`mdBook`** is the canonical choice for guide-style documentation — markdown chapters, search, cross-linking. "The Rust Programming Language" book is written in mdBook. ix's [`docs/INDEX.md`](../INDEX.md) follows the same curriculum pattern without going full mdBook.
- **`Litho (deepwiki-rs)`** analyses crate structure to auto-generate C4 architecture diagrams (context / container / component) that stay in sync with the code.
- **`Auto-UML`** parses Rust via tree-sitter and emits Mermaid class diagrams for the whole repo.
- **`Oxdraw`** is diagram-as-code: declarative Mermaid linked to specific code segments, good for onboarding docs that point at real files.
- **`Utoipa`** auto-generates OpenAPI specs from Rust types + handlers via attribute macros — the standard choice for axum/actix web services.
- **`simple-mermaid`** embeds Mermaid diagrams into `#[doc]` attributes so they render in the HTML rustdoc output.

**Cross-language.** `Doxygen` (C/C++/many), `Sphinx` (Python/C/C++), `godoc` (Go built-in), `TypeDoc` (TypeScript), `Kodesage` (AI-powered multi-source synthesis across language-agnostic repos).

**The usual Rust recipe.** `rustdoc` for API references, `mdBook` for long-form guides, `Litho` for architecture diagrams, `Utoipa` if you have HTTP handlers. That's the "gold-standard" stack the Rust community converges on.

### 6. Numeric libraries

**What it is.** Not analysis tools but the math libraries you *use inside* analysis tools. Linear algebra, numerical methods, optimisation, statistics.

**When to use.** You're writing a new analysis tool and want the primitives rather than reinventing them.

**Tools in the catalog.** Rust: `nalgebra` (static + dynamic matrices, decompositions, geometry), `ndarray` (numpy-equivalent, BLAS-backed), `Peroxide` (numpy/MATLAB-like with autodiff). Python: `SciPy` (the canonical scientific computing library). Language-embedded: `MathOpt` (C++/Python LP and MIP), `MATLAB` (commercial), `GNU Octave` (free MATLAB-compatible).

**For ix's own code.** The workspace already uses `ndarray` 0.17 as its core `f64` matrix type. Per-crate conventions are in [`CLAUDE.md`](../../CLAUDE.md).

### 7. Machine-learning frameworks (Rust ecosystem)

**What it is.** Full-stack ML libraries in Rust — deep-learning frameworks, classical-ML toolkits, dataframe libraries, and edge-inference engines. The Python ML ecosystem dominates research, but Rust's advantage is performance-critical deployment, memory safety, and embedded AI. The math is the usual ML stack: tensors, backprop, autodiff, GBDT, SVM, k-means.

**When to use.** The analysis itself is an ML task rather than a single primitive. You want to train or serve a model, not just compute a statistic.

**Tools in the catalog.**

- **Deep learning.** `Burn` is the comprehensive full-stack framework — data loading, model definition, training, hyperparameter optimisation, custom kernels. `Candle` is Hugging Face's minimalist framework optimised for LLM inference on NVIDIA GPUs via cuTENSOR and cuDNN. `tch-rs` gives you high-performance Rust bindings to the PyTorch C++ API — the bridge for teams coming from the Python PyTorch ecosystem. `dfdx` is the declarative, functional-style differentiable-programming library with autodiff and CUDA support.
- **Classical ML.** `Linfa` is the "Scikit-learn of Rust": standardised API across SVM, k-means, logistic regression, Gaussian mixture models. `SmartCore` is lower-level, covers classification / regression / clustering, and prioritises interpretability (feature importance, decision paths).
- **Infrastructure.** `Polars` is a blazingly fast Arrow-backed DataFrame library similar to Pandas, used for preprocessing in both Rust and Python pipelines. `XGBoost-RS` provides Rust bindings for the canonical gradient-boosting library — the go-to for structured-data problems. `Tract` is the edge and embedded inference engine supporting ONNX, TensorFlow, and PyTorch models.

**When ix is enough.** `ix-supervised`, `ix-unsupervised`, `ix-ensemble`, `ix-nn`, `ix-rl`, and `ix-autograd` together cover the classical ML surface plus a trainable transformer stack. For small-to-medium pipelines where you want everything in-process, deterministic, and MCP-callable, ix's own crates are the right choice. The `ix_ml_pipeline` and `ix_ml_predict` tools wrap these for agent consumption.

**When to reach for Burn or Candle.** You're training a real deep-learning model (not a toy transformer), you need GPU acceleration beyond what `ix-gpu`'s compute shaders can do, or you want pretrained models from Hugging Face. Start with Candle for LLM inference and Burn for full-stack training. Use Tract if you're deploying to edge devices and need ONNX compatibility.

**Quick choice framework.** Full DL stack: Burn. LLM inference: Candle. PyTorch ecosystem access: tch-rs. Classical ML on tabular data: Linfa or SmartCore. Data preprocessing at scale: Polars. Edge inference: Tract.

---

## Quick decision table

| You want to… | Reach for |
|---|---|
| Measure cyclomatic / cognitive / Halstead complexity | `ix_code_analyze` or `Radon` / `gocyclo` / their per-language equivalents |
| Detect bugs by AST pattern | `CodeQL` |
| Prove a C program has no overflows | `Astrée` or `Polyspace` |
| Prove a Rust function is functionally correct | `Verus` or `Creusot` |
| Find all possible execution paths in a Rust function | `Kani` |
| Detect undefined behaviour in unsafe Rust | `Miri` |
| Find data races in concurrent Rust | `Loom` or `ThreadSanitizer` |
| Rank Rust crates by refactor priority | `ix_cargo_deps` + `ix_git_log` + `ix_kmeans` (the refactor-oracle pattern) or `CodeScene` |
| Treat git history as a time series for anomaly detection | `ix_git_log` + `ix_fft` + `ix_chaos_lyapunov` |
| Auto-generate API docs for a Rust crate | `rustdoc` |
| Auto-generate a user manual for a Rust crate | `mdBook` |
| Auto-generate architecture diagrams for a Rust crate | `Litho` or `Auto-UML` |
| Auto-generate OpenAPI specs for a Rust HTTP service | `Utoipa` |
| Get linear algebra primitives in Rust | `ndarray` or `nalgebra` |
| Train a full deep-learning model in Rust | `Burn` |
| Serve an LLM on NVIDIA GPUs in Rust | `Candle` |
| Reuse a PyTorch model in a Rust service | `tch-rs` |
| Classical ML in Rust (SVM, k-means, logreg) | `Linfa` or `SmartCore` (or `ix-supervised` / `ix-unsupervised`) |
| Pandas-like dataframe preprocessing in Rust | `Polars` |
| Gradient boosting on tabular data in Rust | `XGBoost-RS` (or `ix_gradient_boosting`) |
| ONNX / TensorFlow / PyTorch inference on edge devices | `Tract` |

---

## Contributing to the catalog

The catalog is a `const &[CodeAnalysisTool]` in [`crates/ix-code/src/catalog.rs`](../../crates/ix-code/src/catalog.rs). Adding a tool is a three-step PR:

1. Add a `CodeAnalysisTool { ... }` literal in the appropriate category section. Keep entries alphabetical within each category.
2. Update the `catalog_is_not_empty_and_covers_every_category` test's minimum count if you cross a round number.
3. If the tool is Rust-first-class, add it to the `rust_query_includes_the_rust_specific_suite` allowlist.

Every entry needs: `name`, `category`, `technique`, `languages` (non-empty, lowercase-with-hyphens), `description`, and a `url` when the tool has a canonical home.

---

## References

- **Authoritative catalog source:** [`crates/ix-code/src/catalog.rs`](../../crates/ix-code/src/catalog.rs)
- **MCP tool schema:** `ix_code_catalog` in [`crates/ix-agent/src/tools.rs`](../../crates/ix-agent/src/tools.rs)
- **MCP handler:** `code_catalog` in [`crates/ix-agent/src/handlers.rs`](../../crates/ix-agent/src/handlers.rs)
- **Smoke tests:** [`crates/ix-agent/tests/code_catalog_smoke.rs`](../../crates/ix-agent/tests/code_catalog_smoke.rs)
- **Unit tests:** `#[cfg(test)] mod tests` at the bottom of `crates/ix-code/src/catalog.rs`
- **User manual entry point:** [`docs/MANUAL.md`](../MANUAL.md) — see §4 for where this catalog slots into the broader tool inventory
