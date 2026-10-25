# Local Transformer Model Explorer

## 1. Objective

Build a local application that helps users understand a transformer model by viewing its actual Python implementation beside an interactive architecture diagram.

The main interaction is bidirectional navigation:

- Selecting a supported code region highlights the corresponding architecture nodes.
- Clicking an architecture node opens its implementation and highlights the relevant source lines.

This is a new product concept. The existing decision-model script is not the product to visualize or the basis of the user workflow.

## 2. Agreed scope

- Run on the user's local machine, with a browser interface backed by a local Python process.
- Start with Hugging Face model IDs as the loading entry point.
- Start with transformer models implemented in PyTorch and supported by Hugging Face Transformers.
- Show source code and architecture next to each other.
- Highlight the location of selected code within the architecture.
- Leave hosting and deployment outside the current scope.

The following are proposed implementation defaults, to validate during development:

- Support one transformer family first, then add further families through adapters.
- Use React Flow for the interactive diagram and ELK.js for automatic layout.
- Use a read-only code viewer with line highlighting and selection support.
- Separate architecture inspection from loading parameter tensors into memory.

## 3. Important technical boundaries

### Weights do not contain the original source code

A weights file contains parameter tensors, not the original Python implementation. The application needs a model configuration and an available implementation to display meaningful source and architecture.

For the first version, resolve those through a Hugging Face model repository and the installed Transformers package. Display where the implementation came from and which package version was used. A repository revision identifies repository artifacts; it does not automatically identify the locally installed Transformers implementation.

### `inspect` retrieves available source

Use Python's `inspect` module to obtain source files and line ranges for available Python classes and methods. It does not reconstruct source from weights, recover compiled kernels, or automatically collect every dependency of a model.

The product should present the available model implementation through navigable files and symbols. Avoid promising a single self-contained file containing every dependency or the original training code.

### Structure and computation are different

The module hierarchy reveals which components exist. It does not establish execution order, tensor flow, residual connections, or every functional operation.

Build the initial semantic architecture using a validated adapter for each supported model family. Do not infer execution edges from the order returned by `named_modules()`.

### Source links can map to multiple instances

Repeated transformer layers often share one class and one `forward` implementation. Selecting that method can highlight several layer instances. The interface must make this relationship clear and let users focus on a specific instance.

## 4. User workflow

1. Start the application locally and open its browser interface.
2. Enter a Hugging Face model ID; optionally specify a revision.
3. Resolve configuration and check whether the model family is supported.
4. Resolve the installed implementation and build an inspection representation.
5. Open the source viewer and architecture diagram.
6. Expand a transformer block to inspect its attention, normalization, MLP, and residual paths.
7. Select a mapped code region to highlight the matching diagram components.
8. Click a diagram component to jump to its class or method implementation.
9. Optionally load actual weights to inspect parameter metadata from the loaded checkpoint.

Show progress and errors for individual stages. Configuration download, inspection, and weight loading should be distinct operations with distinct statuses.

## 5. Loading behavior

### Initial input

Accept a Hugging Face model ID, such as a public repository containing a supported transformer. Use one small supported model as the development fixture; the final family choice remains an implementation decision.

Provide:

- Model ID field.
- Optional revision field.
- Inspect action.
- Model identity and configuration summary after successful inspection.
- Optional load-weights action with download and memory information when available.

### Inspection without full weights

Prefer constructing an inspectable model from configuration using a supported empty-weight or meta-device approach. Verify compatibility with the chosen family before relying on it. Avoid accidentally allocating full parameter tensors just to obtain source or structure.

The interface must distinguish:

- Architecture derived from configuration.
- Parameter counts derived from instantiated parameter shapes.
- Actual checkpoint weights loaded into memory.

Never label configuration-only inspection as successful weight loading.

### Local cache

Reuse Hugging Face's cache. A previously cached model can be inspected offline if its required configuration and compatible implementation are available. Give a clear message when required artifacts are missing.

### Later input support

Add local Hugging Face folders containing configuration and saved model artifacts after the model-ID workflow is stable. Arbitrary weight files and custom architecture uploads remain outside the first version.

## 6. Interface and visual direction

Use https://yuguerten.github.io/library/ as the requested visual reference. Its appearance has not yet been verified, so inspect it before making typography, color, spacing, and component decisions. Do not substitute an assumed style and describe it as matching the reference.

The primary screen should contain:

- A compact model-loading and status area.
- A source panel on the left.
- An architecture panel on the right.
- A resizable divider.
- A compact component details area when a node is selected.

Prioritize desktop use because both panels must remain readable at once. On smaller screens, allow switching between panels while preserving selection.

### Source panel

- Read-only syntax-highlighted Python.
- Line numbers and source file names.
- File and symbol navigation.
- Class and method selection.
- Highlighting of mapped source ranges.
- Automatic scrolling when a diagram node is selected.
- A clear message for unavailable Python source.

Expose source progressively: begin with the top-level implementation and let users navigate to component implementations. Preserve access to full available files without dumping unrelated package code into one panel.

### Architecture panel

- Pan, zoom, fit-to-view, and reset controls.
- Expandable groups for repeated transformer layers.
- Readable node labels and component types.
- A visible selection highlight that remains distinguishable without relying only on color.
- Edges describing verified semantic flow for the supported implementation.
- A compact overview by default, with one block expandable into its internal components.

Do not render hundreds of detailed nodes initially. Represent repeated layers as a group with a repetition count, then allow inspection of individual layers.

### Component details

Where available, show module path, class, source file, line range, dimensions, parameter count, and configuration values relevant to the component. Distinguish direct parameters from totals including child modules to avoid double counting.

## 7. Code-to-architecture mapping

Create an explicit mapping layer shared by both panels. Do not rely on frontend text matching or matching node labels to code strings.

Each architecture node should have:

- Stable ID within an inspection result.
- Module instance path, when applicable.
- Component type and display label.
- Parent group ID, when applicable.
- Configuration and parameter metadata.
- One or more source references.

Each source reference should have:

- Source file ID.
- Class or symbol identity.
- Method name, when applicable.
- Start and end line, and optional columns.
- Mapping granularity: class, method, or supported expression.
- Related architecture node IDs.
- Mapping origin, such as inspected symbol or family-adapter mapping.

Use `inspect.getsourcefile()` and `inspect.getsourcelines()` for source locations. Use Python AST analysis where needed to identify method bodies and explicitly supported call sites. Resolve imported and inherited implementations deliberately.

### Selection rules

- Selecting a mapped class or method highlights its related components.
- A mapped expression can highlight a narrower component when the adapter supports it.
- An unmapped selection reports that no architecture mapping exists; it must not invent one.
- If multiple nodes share the selected source, highlight the group or matching instances and show the match count.
- Selecting a node inside a collapsed group reveals it or highlights the containing group with a clear path to expand it.
- Clicking a node opens its most relevant mapped method, usually `forward`, with a class-level fallback.
- Preserve selection while panning, zooming, and resizing panels.
- Prevent feedback loops between code-selection events and diagram-selection events.

Exact mapping of every arbitrary Python expression is not an initial requirement.

## 8. Architecture extraction strategy

### First version: a family adapter

Create an adapter for one supported transformer family. It should:

1. Recognize the configuration and implementation classes.
2. Enumerate real component instances and their paths.
3. Identify embeddings, repeated blocks, attention, normalization, MLP, and output components as applicable.
4. Build verified semantic edges, including residual branches.
5. Link nodes and selected call sites to actual implementation source.
6. Return an explicit unsupported result when assumptions do not hold.

Adapters must reflect the resolved implementation. Account for configurable branches, such as attention backends, by showing their status or withholding unsupported detail.

### Later: execution-derived graphs

Evaluate `torch.fx` or other tracing approaches after the initial module and source navigation works. Tracing can expose operations but has limitations with dynamic control flow and transformer implementations. It also requires representative inputs and can describe only the captured path.

If execution-derived graphs are introduced, label them separately from the architecture overview and include the input assumptions. Runtime tensor shapes must come from execution or tracing; configuration dimensions alone are not observed runtime shapes.

## 9. Proposed implementation stack

| Responsibility | Proposed tool | Purpose |
| --- | --- | --- |
| Model resolution | Hugging Face Transformers | Resolve configurations and supported PyTorch implementations |
| Source extraction | Python `inspect` | Retrieve available classes, methods, files, and line ranges |
| Source analysis | Python `ast` | Locate supported call sites and refine source mappings |
| Model structure | PyTorch module APIs | Enumerate instances, module paths, and parameter metadata |
| Local API | FastAPI | Connect the browser interface to the Python inspection process |
| Frontend | React and TypeScript | Coordinate source, graph, and selection state |
| Code viewer | Monaco Editor, read-only | Display source, selections, and line decorations |
| Diagram | React Flow | Render custom nodes, groups, edges, and navigation |
| Automatic layout | ELK.js | Position nodes and route graph structure |

React Flow renders the graph; it does not extract model architecture. ELK.js lays out the graph; it does not determine model semantics. The Python inspection layer and family adapters provide those semantics.

Treat these as proposed choices until a small integration prototype confirms source selection, expandable groups, and graph layout work together.

## 10. Local system design

The local Python service resolves model information and produces a serializable inspection result. The browser renders that result and manages selection without additional model inference.

Suggested logical components:

- Model resolver: ID, revision, configuration, cache state, and optional weights.
- Inspector: instance hierarchy, symbols, source files, and parameter metadata.
- Adapter registry: supported families and semantic graph generation.
- Mapping builder: relationships between nodes and source ranges.
- Local API: inspection jobs, status, results, and optional weight loading.
- Frontend workspace: loading controls, code viewer, architecture canvas, and details.

The inspection result should include model identity, repository revision when resolved, implementation package versions, loading mode, graph nodes and edges, source files, mappings, and any unsupported details.

Bind the service to localhost by default. Use a development proxy or explicit local origins for browser API access. Keep repository credentials in the local backend environment and out of inspection responses.

Do not enable arbitrary repository code execution implicitly. The initial supported scope should use built-in Transformers implementations; repositories requiring custom code should receive an explicit unsupported message.

## 11. Implementation phases

### Phase 1: resolve the reference and prove inspection

- Inspect the visual reference and record its relevant design choices.
- Choose one small supported transformer family and a fixed development revision.
- Resolve its configuration and implementation without full weight allocation.
- Extract the top-level class and component methods with correct source locations.
- Produce a minimal inspection result containing actual modules and source references.

Completion criterion: the backend can identify the model and return its available implementation source and component hierarchy.

### Phase 2: build the local workspace

- Add the local API and browser interface.
- Implement model-ID loading, status, and readable errors.
- Add the read-only source panel and resizable architecture panel.
- Apply the verified reference style.
- Render the initial overview with a family adapter and automatic layout.

Completion criterion: one supported model can be inspected through the browser with readable code and architecture side by side.

### Phase 3: link code and architecture

- Implement stable node IDs and source mappings.
- Add diagram-to-source navigation.
- Add source-to-diagram highlighting.
- Handle repeated instances, collapsed groups, inherited methods, and unmapped selections.
- Add component details and parameter-count definitions.

Completion criterion: bidirectional selection works reliably for the supported classes, methods, and explicitly mapped call sites.

### Phase 4: complete loading and usability

- Add optional actual weight loading with clear state and failure handling.
- Reuse cached artifacts and verify offline inspection.
- Improve large repeated-layer navigation and graph readability.
- Document installation, startup, supported models, and limitations.
- Verify a clean local setup using the documented instructions.

Completion criterion: the agreed local model-ID workflow is usable from setup through inspection, with optional checkpoint loading accurately reported.

### Later phases

- Support local Hugging Face folders.
- Add additional family adapters.
- Add richer expression mappings where reliable.
- Explore execution-derived tensor shapes and computation graphs.
- Consider arbitrary custom models only after defining their implementation requirements.

## 12. Validation and acceptance criteria

Use a small randomly initialized configuration of the supported family for routine automated validation. Avoid requiring large downloads or GPU hardware for basic checks.

The first version is complete when:

1. A supported public Hugging Face model ID resolves through the local interface.
2. Inspection does not require loading the full weights into memory.
3. Displayed source is extracted from the resolved implementation, with valid file and line references.
4. The diagram agrees with the supported model's configuration and verified forward structure.
5. Clicking attention, MLP, normalization, and other supported components navigates to their source.
6. Selecting mapped code highlights the correct nodes, including shared implementations across repeated layers.
7. Expanding and collapsing blocks preserves meaningful selection.
8. Unmapped source and unavailable source are reported accurately.
9. Unsupported model families and custom-code repositories fail clearly without fabricated diagrams.
10. Configuration-only inspection and actual weight loading have distinct visible states.
11. Cached inspection works offline when required artifacts are present.
12. The documented startup procedure works on a clean local environment.

Test mapping correctness, repeated-layer relationships, graph semantics, and loading failures. Perform a browser check of scrolling, highlighting, grouping, and layout; backend tests alone cannot validate these interactions.

## 13. Out of scope for the first version

- Hosted services, deployment, multi-user accounts, and cloud storage.
- Training, fine-tuning, or editing model implementations.
- Live tensor-flow animation and inference playback.
- Universal support for arbitrary weights or all transformer families.
- Recovering training code or Python source from tensors.
- Displaying every compiled backend kernel or library dependency.
- Guaranteed exact mappings for arbitrary selected expressions.

## 14. Decisions still to make during implementation

- Which transformer family provides the clearest initial proof of concept?
- Which visual details from the reference should carry into the two-panel workspace?
- How much internal attention detail should appear at each expansion level?
- Does the proposed editor support the required selection behavior without unnecessary complexity?
- Which empty-weight construction approach works reliably for the selected implementation?

These choices should be resolved with a small prototype before extending model coverage.

## 15. Technical references

- Visual reference: https://yuguerten.github.io/library/
- Python source inspection: https://docs.python.org/3/library/inspect.html
- PyTorch model saving and loading: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- Hugging Face model loading: https://huggingface.co/docs/transformers/models
- React Flow custom nodes: https://reactflow.dev/learn/customization/custom-nodes
- React Flow layout options: https://reactflow.dev/learn/layouting/layouting
- PyTorch FX and tracing limitations: https://docs.pytorch.org/docs/stable/fx.html
