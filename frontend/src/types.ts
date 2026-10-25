export interface SourceFile {
  file_id: string;
  path: string;
  display_name: string;
  content: string;
  package_version?: string | null;
}

export interface SourceReference {
  id: string;
  source_file_id: string;
  symbol: string;
  method?: string | null;
  start_line: number;
  end_line: number;
  start_col?: number | null;
  end_col?: number | null;
  granularity: string;
  architecture_node_ids: string[];
  origin: string;
}

export interface ArchitectureNode {
  id: string;
  label: string;
  component_type: string;
  module_path?: string | null;
  parent_id?: string | null;
  repetition?: number | null;
  repetition_index?: number | null;
  source_reference_ids: string[];
  config: Record<string, unknown>;
  parameter_count?: number | null;
  direct_parameter_count?: number | null;
}

export interface ArchitectureEdge {
  source: string;
  target: string;
  kind: string;
  label?: string | null;
}

export interface InspectionResult {
  model_id: string;
  revision?: string | null;
  config_source?: string | null;
  implementation_package: string;
  implementation_version: string;
  implementation_file?: string | null;
  family?: string | null;
  loading_mode: string;
  model_identity: Record<string, unknown>;
  graph: ArchitectureNode[];
  edges: ArchitectureEdge[];
  source_files: SourceFile[];
  source_references: SourceReference[];
  unsupported: string[];
  parameter_count?: number | null;
  warnings: string[];
}
