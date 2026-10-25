import { useMemo, useState, useRef, useCallback, useEffect } from "react";
import Editor from "@monaco-editor/react";
import type { InspectionResult, SourceReference } from "../types";

export function SourcePanel({
  result,
  selectedRefId,
  onJumpToRef,
}: {
  result: InspectionResult;
  selectedRefId: string | null;
  onJumpToRef: (id: string) => void;
}) {
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const editorRef = useRef<any>(null);
  const decorationIdsRef = useRef<string[]>([]);
  const cursorSubscriptionRef = useRef<{ dispose: () => void } | null>(null);
  const [editorReady, setEditorReady] = useState(false);

  const activeFile =
    result.source_files.find((f) => f.file_id === activeFileId) ??
    result.source_files[0] ??
    null;
  const activeFileIdRef = useRef<string | null>(null);
  const selectedRefIdRef = useRef<string | null>(selectedRefId);
  const sourceReferencesRef = useRef(result.source_references);
  const onJumpToRefRef = useRef(onJumpToRef);
  activeFileIdRef.current = activeFile?.file_id ?? null;
  selectedRefIdRef.current = selectedRefId;
  sourceReferencesRef.current = result.source_references;
  onJumpToRefRef.current = onJumpToRef;

  const fileRefs = useMemo(() => {
    if (!activeFile) return [];
    return result.source_references.filter(
      (r) => r.source_file_id === activeFile.file_id
    );
  }, [result, activeFile]);

  const selectedRef = selectedRefId
    ? result.source_references.find((r) => r.id === selectedRefId)
    : null;

  const handleEditorMount = (editor: any) => {
    editorRef.current = editor;
    setEditorReady(true);
    cursorSubscriptionRef.current?.dispose();
    cursorSubscriptionRef.current = editor.onDidChangeCursorPosition(
      ({ position }: { position: { lineNumber: number } }) => {
        const matching = sourceReferencesRef.current
          .filter(
            (ref) =>
              ref.source_file_id === activeFileIdRef.current &&
              ref.start_line <= position.lineNumber &&
              ref.end_line >= position.lineNumber
          )
          .sort((a, b) => (a.end_line - a.start_line) - (b.end_line - b.start_line))[0];
        if (matching && matching.id !== selectedRefIdRef.current) onJumpToRefRef.current(matching.id);
      }
    );
  };

  const highlightRange = useCallback(
    (ref: SourceReference | null) => {
      const editor = editorRef.current;
      if (!editor || !ref) return;
      const file = result.source_files.find((f) => f.file_id === ref.source_file_id);
      if (!file) return;
      // Ensure we're showing the right file before navigating
      setActiveFileId(ref.source_file_id);
      requestAnimationFrame(() => {
        try {
          editor.revealLineInCenter(ref.start_line);
        } catch {
          /* noop */
        }
      });
    },
    [result.source_files]
  );

  useEffect(() => {
    if (selectedRef && activeFile && selectedRef.source_file_id === activeFile.file_id) {
      highlightRange(selectedRef);
    }
  }, [selectedRef, activeFile, highlightRange]);

  // Apply decorations for mapped ranges whenever file/selection changes.
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    const deco = result.source_references
      .filter((r) => r.source_file_id === activeFile?.file_id)
      .map((r) => ({
        range: {
          startLineNumber: r.start_line,
          startColumn: 1,
          endLineNumber: r.end_line,
          endColumn: 1,
        },
        options: {
          isWholeLine: true,
          className: r.id === selectedRefId ? "line-decoration-selected" : "line-decoration",
          overviewRuler: r.id === selectedRefId
            ? { color: "rgba(166, 61, 47, 0.9)", position: 4 }
            : undefined,
        },
      }));
    decorationIdsRef.current = editor.deltaDecorations(decorationIdsRef.current, deco);
  }, [result, activeFile, selectedRefId, editorReady]);

  useEffect(() => () => {
    cursorSubscriptionRef.current?.dispose();
    const editor = editorRef.current;
    if (editor && decorationIdsRef.current.length) {
      editor.deltaDecorations(decorationIdsRef.current, []);
    }
  }, []);

  if (!activeFile) {
    return <div className="placeholder">No Python source available.</div>;
  }

  return (
    <>
      <div className="source-nav">
        <select
          value={activeFile.file_id}
          onChange={(e) => setActiveFileId(e.target.value)}
          aria-label="Source file"
        >
          {result.source_files.map((f) => (
            <option key={f.file_id} value={f.file_id}>
              {f.display_name}
            </option>
          ))}
        </select>
        <span className="file-label">{activeFile.display_name}</span>
        {activeFile.package_version && (
          <span className="mono-badge">v{activeFile.package_version}</span>
        )}
        <select
          value={selectedRefId ?? ""}
          onChange={(e) => e.target.value && onJumpToRef(e.target.value)}
          aria-label="Symbol"
          className="jump"
        >
          <option value="">— symbol —</option>
          {fileRefs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.symbol}
              {r.method ? `::${r.method}` : ""} ({r.start_line}–{r.end_line})
            </option>
          ))}
        </select>
      </div>
      <div className="source-editor">
        <Editor
          height="100%"
          language="python"
          theme="vs-light"
          value={activeFile.content}
          path={activeFile.file_id}
          onMount={handleEditorMount}
          options={{
            readOnly: true,
            minimap: { enabled: false },
            fontSize: 12.5,
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            folding: true,
            wordWrap: "off",
          }}
        />
      </div>
      <style>{`
        .line-decoration { background: rgba(166, 61, 47, 0.025); }
        .line-decoration-selected {
          background: rgba(166, 61, 47, 0.18);
          box-shadow: inset 3px 0 0 #a63d2f;
        }
      `}</style>
    </>
  );
}
