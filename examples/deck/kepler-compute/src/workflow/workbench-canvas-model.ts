// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getNodesBounds,
  type Edge,
  type FitViewOptions,
  type Node,
  type NodeSelectionChange,
  type Rect,
  type Viewport,
  type Dimensions
} from '@xyflow/react';
import type {
  WorkflowDocument,
  WorkflowOperation,
  WorkflowOperationRegistry,
  WorkflowPortKind,
  WorkflowRunnerState,
  WorkflowStep,
  WorkflowStepState,
  WorkflowStepStatus
} from '@luma-contributors/kepler';
import type {WorkbenchNodeData} from './workbench-node';
import {WORKBENCH_DOCK} from './workbench-contract';

export type WorkbenchFlowNode = Node<WorkbenchNodeData, 'step'>;
export type WorkbenchEdgeData = {
  kind: WorkflowPortKind;
  state: 'running' | 'stale' | 'invalid' | 'ready';
};
export type WorkbenchFlowEdge = Edge<WorkbenchEdgeData>;

// Immutable step objects own the cache lifetime; removed documents are not retained.
const nodeCache = new WeakMap<WorkflowStep, Map<boolean, WorkbenchFlowNode>>();

export function makeWorkbenchHandleId(direction: 'in' | 'out', port: string): string {
  return `${direction}:${port}`;
}

export function parseWorkbenchHandleId(id: string): {direction: 'in' | 'out'; port: string} | null {
  if (id.startsWith('in:')) return {direction: 'in', port: id.slice(3)};
  if (id.startsWith('out:')) return {direction: 'out', port: id.slice(4)};
  return null;
}

export function makeWorkbenchNodes(
  document: WorkflowDocument,
  selectedStepIds: readonly string[]
): WorkbenchFlowNode[] {
  const selection = new Set(selectedStepIds);
  return document.steps.map(step => {
    const selected = selection.has(step.id);
    let variants = nodeCache.get(step);
    if (!variants) {
      variants = new Map();
      nodeCache.set(step, variants);
    }
    let node = variants.get(selected);
    if (!node) {
      node = {
        id: step.id,
        type: 'step',
        position: step.position,
        data: {stepId: step.id},
        selected
      };
      variants.set(selected, node);
    }
    return node;
  });
}

/** Keep React Flow measurements and drag metadata while applying the document and selection. */
export function reconcileWorkbenchNodes(
  current: WorkbenchFlowNode[],
  desired: readonly WorkbenchFlowNode[]
): WorkbenchFlowNode[] {
  const existing = new Map(current.map(node => [node.id, node]));
  const reconciled = desired.map(node => {
    const previous = existing.get(node.id);
    if (!previous) return node;
    const position =
      previous.position.x === node.position.x && previous.position.y === node.position.y
        ? previous.position
        : node.position;
    const data = previous.data.stepId === node.data.stepId ? previous.data : node.data;
    if (
      position === previous.position &&
      data === previous.data &&
      previous.selected === node.selected &&
      previous.type === node.type
    )
      return previous;
    return {...previous, type: node.type, position, data, selected: node.selected};
  });
  return current.length === reconciled.length &&
    reconciled.every((node, index) => node === current[index])
    ? current
    : reconciled;
}

/** Only user select changes affect selection; repeated or cancelling changes preserve identity. */
export function applyWorkbenchSelectionChanges(
  current: readonly string[],
  changes: readonly NodeSelectionChange[]
): readonly string[] {
  const selected = new Set(current);
  for (const change of changes) {
    if (change.selected) selected.add(change.id);
    else selected.delete(change.id);
  }
  return selected.size === current.length && current.every(id => selected.has(id))
    ? current
    : [...selected];
}

export const WORKBENCH_CANVAS_LAYOUT = {
  minimapWidth: 120,
  minimapHeight: 80,
  minimapMargin: 8,
  minimumMinimapCanvasHeight: 320,
  minimumMinimapSteps: 6
} as const;

export function shouldShowWorkbenchMinimap(canvasHeight: number, stepCount: number): boolean {
  return (
    canvasHeight >= WORKBENCH_CANVAS_LAYOUT.minimumMinimapCanvasHeight &&
    stepCount >= WORKBENCH_CANVAS_LAYOUT.minimumMinimapSteps
  );
}

/** Prefer readable framing without cropping graphs that cannot fit at that scale. */
export function getWorkbenchFitViewOptions(
  _showMinimap: boolean,
  canvasSize?: Dimensions,
  bounds?: Pick<Rect, 'width' | 'height'>
): FitViewOptions {
  let padding = 0.12;
  let minZoom = 0.2;
  if (
    canvasSize &&
    bounds &&
    canvasSize.width > 0 &&
    canvasSize.height > 0 &&
    bounds.width > 0 &&
    bounds.height > 0
  ) {
    const unpaddedZoom = Math.min(
      canvasSize.width / bounds.width,
      canvasSize.height / bounds.height
    );
    if (unpaddedZoom >= 0.7) padding = Math.min(padding, Math.max(0, unpaddedZoom / 0.7 - 1));
    minZoom = Math.min(minZoom, unpaddedZoom / (1 + padding));
  }
  return {padding, minZoom, maxZoom: 1};
}

export function getWorkbenchNodeRectangle(node: WorkbenchFlowNode): Rect {
  return {
    ...node.position,
    width: node.measured?.width ?? WORKBENCH_DOCK.nodeWidth,
    height: node.measured?.height ?? 120
  };
}

export function getWorkbenchNodeBounds(nodes: readonly WorkbenchFlowNode[]): Rect | undefined {
  return nodes.length
    ? getNodesBounds(nodes.map(node => ({...node, ...getWorkbenchNodeRectangle(node)})))
    : undefined;
}

/** Minimal pan in screen pixels; oversized rectangles are centred without changing zoom. */
export function getWorkbenchRevealViewport(
  rectangle: Rect,
  viewport: Viewport,
  canvasSize: Dimensions,
  margin = 24
): Viewport | null {
  if (canvasSize.width <= 0 || canvasSize.height <= 0 || viewport.zoom <= 0) return null;
  function getPan(start: number, size: number, available: number): number {
    if (size > available - 2 * margin) return (available - size) / 2 - start;
    if (start < margin) return margin - start;
    if (start + size > available - margin) return available - margin - start - size;
    return 0;
  }
  const horizontal = getPan(
    rectangle.x * viewport.zoom + viewport.x,
    rectangle.width * viewport.zoom,
    canvasSize.width
  );
  const vertical = getPan(
    rectangle.y * viewport.zoom + viewport.y,
    rectangle.height * viewport.zoom,
    canvasSize.height
  );
  if (Math.abs(horizontal) < 0.01 && Math.abs(vertical) < 0.01) return null;
  return {...viewport, x: viewport.x + horizontal, y: viewport.y + vertical};
}

/** Reveal an inserted block without enlarging it or exceeding the normal reading scale. */
export function getWorkbenchGroupRevealViewport(
  rectangle: Rect,
  viewport: Viewport,
  canvasSize: Dimensions,
  margin = 24
): Viewport | null {
  if (
    canvasSize.width <= margin * 2 ||
    canvasSize.height <= margin * 2 ||
    rectangle.width <= 0 ||
    rectangle.height <= 0 ||
    viewport.zoom <= 0
  )
    return null;
  const zoom = Math.min(
    viewport.zoom,
    1,
    (canvasSize.width - margin * 2) / rectangle.width,
    (canvasSize.height - margin * 2) / rectangle.height
  );
  const scaled = zoom === viewport.zoom ? viewport : {...viewport, zoom};
  return (
    getWorkbenchRevealViewport(rectangle, scaled, canvasSize, margin) ??
    (scaled === viewport ? null : scaled)
  );
}

export function makeWorkbenchEdges(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  runner: WorkflowRunnerState | null
): WorkbenchFlowEdge[] {
  const steps = new Map(document.steps.map(step => [step.id, step]));
  const edges: WorkbenchFlowEdge[] = [];
  for (const target of document.steps) {
    for (const [inputName, connection] of Object.entries(target.inputs)) {
      const source = steps.get(connection.stepId);
      if (!source) continue;
      const output = registry
        .get(source.operationId)
        ?.outputs.find(port => port.name === connection.output);
      const input = registry.get(target.operationId)?.inputs.find(port => port.name === inputName);
      const producer = runner?.steps[source.id];
      const consumer = runner?.steps[target.id];
      const observed = producer?.result?.outputs[connection.output];
      const invalid =
        !output ||
        !input ||
        consumer?.issues.some(issue => issue.severity === 'error' && issue.port === inputName) ||
        (consumer?.error?.severity === 'error' && consumer.error.port === inputName);
      const state = invalid
        ? 'invalid'
        : consumer?.status === 'running'
          ? 'running'
          : !observed || producer?.status === 'stale'
            ? 'stale'
            : 'ready';
      edges.push({
        id: `${target.id}:${inputName}`,
        source: source.id,
        sourceHandle: makeWorkbenchHandleId('out', connection.output),
        target: target.id,
        targetHandle: makeWorkbenchHandleId('in', inputName),
        data: {kind: observed?.kind ?? output?.kind ?? input?.accepts[0] ?? 'table', state}
      });
    }
  }
  return edges;
}

export function getWorkbenchStepSummary(
  step: WorkflowStep,
  operation: WorkflowOperation | undefined,
  stepState: WorkflowStepState | undefined
): string {
  const error = stepState?.issues.find(issue => issue.severity === 'error') ?? stepState?.error;
  if (error) return error.message;
  if (stepState?.result) {
    const result = stepState.result;
    const firstOutput = operation?.outputs[0]?.name;
    const data =
      firstOutput !== undefined ? result.outputs[firstOutput] : Object.values(result.outputs)[0];
    const engine =
      result.engine === 'cpu-reference' ? 'CPU' : result.engine === 'gpu' ? 'GPU' : 'Host';
    const duration = result.cacheHit
      ? 'cached'
      : result.durationMilliseconds < 1000
        ? `${Math.round(result.durationMilliseconds).toLocaleString()} ms`
        : `${(result.durationMilliseconds / 1000).toLocaleString(undefined, {maximumFractionDigits: 1})} s`;
    return `${(data?.table.rowCount ?? 0).toLocaleString()} rows · ${engine} · ${duration}`;
  }
  if (!operation) return 'Unknown operation';
  const missing = operation.inputs.find(
    port => !port.optional && !Object.hasOwn(step.inputs, port.name)
  );
  return missing ? `Needs an input for "${missing.label}"` : 'Ready to run';
}

/** The runner is optional while a workflow is being opened. */
export function getWorkbenchStepStatus(
  step: WorkflowStep,
  operation: WorkflowOperation | undefined,
  stepState: WorkflowStepState | undefined
): WorkflowStepStatus {
  if (step.disabled) return 'disabled';
  if (stepState) return stepState.status;
  return !operation ||
    operation.inputs.some(port => !port.optional && !Object.hasOwn(step.inputs, port.name))
    ? 'not-configured'
    : 'ready';
}
