// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  WorkflowOperationCategory,
  WorkflowPortKind,
  WorkflowStepStatus
} from '@luma-contributors/kepler';

/**
 * Shared vocabulary of the analysis workbench UI: test ids, colours, labels and shortcuts. Every
 * workbench component and every acceptance script reads these; none hard-codes its own.
 */

/** `data-testid` values. Per-item ids are functions. */
export const WORKBENCH_TEST_IDS = {
  dock: 'workflow-dock',
  dockToggle: 'workflow-dock-toggle',
  dockResize: 'workflow-dock-resize',
  dockSummary: 'workflow-dock-summary',
  toolbar: 'workflow-toolbar',
  workflowSelect: 'workflow-select',
  workflowTitle: 'workflow-title',
  workflowNew: 'workflow-new',
  workflowDelete: 'workflow-delete',
  workflowDuplicate: 'workflow-duplicate',
  runAll: 'workflow-run-all',
  runToHere: 'workflow-run-to-here',
  cancel: 'workflow-cancel',
  undo: 'workflow-undo',
  redo: 'workflow-redo',
  addStep: 'workflow-add-step',
  layout: 'workflow-layout',
  autoRun: 'workflow-auto-run',
  engine: 'workflow-engine',
  templates: 'workflow-templates',
  template: (templateId: string) => `workflow-template-${templateId}`,
  importExport: 'workflow-import-export',
  exportJSON: 'workflow-export-json',
  exportTypeScript: 'workflow-export-typescript',
  exportText: 'workflow-export-text',
  importText: 'workflow-import-text',
  importApply: 'workflow-import-apply',
  promoteHistory: 'workflow-promote-history',
  canvas: 'workflow-canvas',
  canvasEmpty: 'workflow-canvas-empty',
  node: (stepId: string) => `workflow-node-${stepId}`,
  nodeStatus: (stepId: string) => `workflow-node-status-${stepId}`,
  nodeDisplay: (stepId: string) => `workflow-node-display-${stepId}`,
  nodeDisable: (stepId: string) => `workflow-node-disable-${stepId}`,
  inputPort: (stepId: string, port: string) => `workflow-port-in-${stepId}-${port}`,
  outputPort: (stepId: string, port: string) => `workflow-port-out-${stepId}-${port}`,
  palette: 'workflow-palette',
  paletteSearch: 'workflow-palette-search',
  paletteCategory: (category: string) => `workflow-palette-category-${category}`,
  paletteItem: (operationId: string) => `workflow-palette-item-${operationId}`,
  inspector: 'workflow-inspector',
  inspectorTab: (tab: string) => `workflow-inspector-tab-${tab}`,
  stepLabel: 'workflow-step-label',
  parameter: (name: string) => `workflow-parameter-${name}`,
  parameterPromote: (name: string) => `workflow-parameter-promote-${name}`,
  parameterError: (name: string) => `workflow-parameter-error-${name}`,
  inputBinding: (port: string) => `workflow-input-${port}`,
  stepIssues: 'workflow-step-issues',
  result: 'workflow-result',
  resultOutput: (output: string) => `workflow-result-output-${output}`,
  resultTable: (output: string) => `workflow-result-table-${output}`,
  resultDisplay: (output: string) => `workflow-result-display-${output}`,
  resultStage: (output: string) => `workflow-result-stage-${output}`,
  resultDownload: (output: string, format: string) =>
    `workflow-result-download-${output}-${format}`,
  resultEngine: 'workflow-result-engine',
  resultDiagnostics: 'workflow-result-diagnostics',
  runLog: 'workflow-run-log',
  runLogToggle: 'workflow-run-log-toggle',
  notice: 'workflow-notice',
  draft: 'workflow-draft',
  draftAccept: 'workflow-draft-accept',
  draftReject: 'workflow-draft-reject',
  displayColorColumn: (output: string) => `workflow-display-color-column-${output}`,
  displayColorScale: (output: string) => `workflow-display-color-scale-${output}`,
  displayPalette: (output: string) => `workflow-display-palette-${output}`,
  suggestion: (suggestionId: string) => `workflow-suggestion-${suggestionId}`,
  saveTemplate: 'workflow-save-template',
  userTemplate: (templateId: string) => `workflow-user-template-${templateId}`,
  userTemplateDelete: (templateId: string) => `workflow-user-template-delete-${templateId}`,
  insertWorkflow: (sourceId: string) => `workflow-insert-${sourceId}`,
  runReport: 'workflow-run-report',
  runReportText: 'workflow-run-report-text',
  graphDot: 'workflow-graph-dot',
  copySteps: 'workflow-copy-steps',
  pasteSteps: 'workflow-paste-steps',
  datasetQuickStart: (datasetId: string) => `workflow-dataset-quick-start-${datasetId}`,
  recommendation: (recommendationId: string) => `workflow-recommendation-${recommendationId}`,
  sourceType: 'workflow-source-type',
  resultFocus: (output: string) => `workflow-result-focus-${output}`,
  drawingQuickStart: 'workflow-drawing-quick-start',
  sidePanel: 'workflow-side-panel',
  sidePanelOpenDock: 'workflow-side-panel-open-dock',
  stepRow: (stepId: string) => `workflow-step-row-${stepId}`,
  workflowInput: (inputId: string) => `workflow-input-value-${inputId}`
} as const;

/** Colours follow the Kepler dark theme; each pair meets 4.5:1 on `surface`. */
export const WORKBENCH_COLORS = {
  surface: '#1b2028',
  surfaceRaised: '#242a34',
  surfaceSunken: '#151a21',
  border: '#3a4352',
  borderStrong: '#586274',
  text: '#e6e9ee',
  textMuted: '#a6afbd',
  accent: '#1fbad6',
  accentText: '#0b1a20',
  danger: '#ff8a80',
  warning: '#f0c35a',
  success: '#5fd4a0',
  focus: '#62d3ff'
} as const;

export const WORKBENCH_PORT_COLORS: Record<WorkflowPortKind, string> = {
  point: '#6bbcf5',
  line: '#f5a968',
  polygon: '#84dc98',
  features: '#c9a8ff',
  raster: '#e0c07a',
  table: '#b6bdc9'
};

export const WORKBENCH_PORT_LABELS: Record<WorkflowPortKind, string> = {
  point: 'Points',
  line: 'Lines',
  polygon: 'Polygons',
  features: 'Features',
  raster: 'Raster band/grid',
  table: 'Table'
};

export const WORKBENCH_STATUS: Record<WorkflowStepStatus, {label: string; color: string}> = {
  'not-configured': {label: 'Not configured', color: '#f0a070'},
  blocked: {label: 'Blocked', color: '#c9a58f'},
  ready: {label: 'Ready', color: '#b6bdc9'},
  queued: {label: 'Queued', color: '#8fa3c4'},
  running: {label: 'Running', color: '#1fbad6'},
  done: {label: 'Done', color: '#5fd4a0'},
  stale: {label: 'Stale', color: '#f0c35a'},
  error: {label: 'Error', color: '#ff8a80'},
  disabled: {label: 'Disabled', color: '#8a94a6'}
};

export const WORKBENCH_CATEGORY_LABELS: Record<WorkflowOperationCategory, string> = {
  source: 'Sources',
  table: 'Tables',
  geometry: 'Geometry',
  network: 'Network',
  aggregation: 'Aggregation',
  interpolation: 'Interpolation',
  density: 'Density',
  clustering: 'Clustering',
  raster: 'Raster',
  analysis: 'Statistics'
};

/**
 * Keyboard shortcuts of the canvas (Houdini and noodles.gl conventions). They apply only while
 * focus is inside the dock and not in a text field.
 */
export const WORKBENCH_SHORTCUTS = [
  // Tab adds a step only after the canvas was focused with the pointer; for keyboard focus Tab
  // keeps moving focus, and N is the key that always works.
  {keys: 'Tab', action: 'add-step', label: 'Add a step (after clicking the canvas)'},
  {keys: 'N', action: 'add-step', label: 'Add a step'},
  {keys: 'Enter', action: 'run-to-here', label: 'Run to the selected step'},
  {keys: 'Shift+Enter', action: 'run-all', label: 'Run the workflow'},
  {keys: 'Escape', action: 'cancel-or-close', label: 'Close the palette or cancel the run'},
  {keys: 'B', action: 'toggle-disabled', label: 'Disable or enable the selected steps'},
  {keys: 'D', action: 'toggle-display', label: 'Show the selected step on the map'},
  {keys: 'F', action: 'frame', label: 'Frame the selection or the whole graph'},
  {keys: 'L', action: 'layout', label: 'Lay out the graph'},
  {keys: 'Delete', action: 'delete', label: 'Delete the selected steps'},
  {keys: 'Mod+D', action: 'duplicate', label: 'Duplicate the selected steps'},
  {keys: 'Mod+Z', action: 'undo', label: 'Undo'},
  {keys: 'Shift+Mod+Z', action: 'redo', label: 'Redo'},
  {keys: 'Mod+A', action: 'select-all', label: 'Select every step'},
  {keys: 'Mod+C', action: 'copy', label: 'Copy the selected steps'},
  {keys: 'Mod+X', action: 'cut', label: 'Cut the selected steps'},
  {keys: 'Mod+V', action: 'paste', label: 'Paste steps'}
] as const;

export type WorkbenchShortcutAction = (typeof WORKBENCH_SHORTCUTS)[number]['action'];

export const WORKBENCH_DOCK = {
  collapsedHeight: 36,
  defaultHeight: 380,
  minimumHeight: 220,
  /** Fraction of the window height. */
  maximumHeightFraction: 0.7,
  inspectorWidth: 360,
  nodeWidth: 220
} as const;
