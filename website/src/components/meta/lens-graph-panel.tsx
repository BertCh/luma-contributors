// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The command graph of a recorded lens run: an SVG DAG (compute and copy nodes, resources as edges
// with byte sizes) with a live luma.gl layer behind it that animates data flowing along the edges.

import {Suspense, useEffect, useMemo, useState} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {formatBytes, formatCount} from '../../lens/format';
import {getEdgePath, layoutLensGraph, type GraphLayout} from '../../lens/layout-graph';
import type {LensViewEdge, LensViewModel, LensViewNode} from '../../lens/types';
import {useLensViewModel} from '../../lens/use-lens-view-model';
import {getLiveBytes, getPeakLiveBytes, LifetimeChart} from './lifetime-chart';
import {createFlowSceneLoader, type FlowGraph} from './scenes/flow-scene';
import './meta.css';

export type LensGraphPanelProps = {
  /** A scenario id such as `gpu-spatial-analysis.kernel-density`, or an already loaded model. */
  scenario: string | LensViewModel;
  /** Hide the stats and legend bar and the step controls. */
  compact?: boolean;
  /** Color dependencies by hazard kind (read-after-write and so on). */
  showHazards?: boolean;
  /** Start on the buffer lifetime view when the model has lifetimes. */
  defaultView?: 'graph' | 'buffers';
  /** Step shown first (0 to node count); the last step shows the whole graph. */
  defaultStep?: number;
  /**
   * Nodes to keep at full strength. Other nodes, and edges not between two highlighted nodes, are
   * dimmed. Undefined or empty dims nothing. Applied as SVG attributes, so it prerenders.
   */
  highlightNodeIds?: readonly string[];
};

/** Opacity of nodes and edges outside `highlightNodeIds`. */
const DIMMED_NODE_OPACITY = 0.28;
const DIMMED_EDGE_OPACITY = 0.12;

const NODE_LAYOUT = {
  nodeWidth: 188,
  nodeHeight: 42,
  layerGap: 64,
  rowGap: 14,
  padding: 22,
  targetWidth: 1040
};
const MINIMUM_STAGE_WIDTH = 620;

export function LensGraphPanel({scenario, ...options}: LensGraphPanelProps) {
  return typeof scenario === 'string' ? (
    <Suspense fallback={<div className="meta-panel meta-skeleton">loading {scenario}</div>}>
      <LoadedLensGraphPanel scenarioId={scenario} options={options} />
    </Suspense>
  ) : (
    <LensGraphView model={scenario} {...options} />
  );
}

function LoadedLensGraphPanel({
  scenarioId,
  options
}: {
  scenarioId: string;
  options: Omit<LensGraphPanelProps, 'scenario'>;
}) {
  const model = useLensViewModel(scenarioId);
  return model ? (
    <LensGraphView model={model} {...options} />
  ) : (
    <div className="meta-panel meta-skeleton">No recorded run for {scenarioId}</div>
  );
}

type Hover = {kind: 'node'; id: string} | {kind: 'edge'; index: number} | null;

const HAZARD_NAMES = {
  raw: 'read after write',
  war: 'write after read',
  waw: 'write after write'
} as const;

function getEdgeHazard(edge: LensViewEdge): 'raw' | 'war' | 'waw' {
  const hazards = edge.hazards ?? [];
  return hazards.includes('waw') ? 'waw' : hazards.includes('war') ? 'war' : 'raw';
}

function LensGraphView({
  model,
  compact = false,
  showHazards = false,
  defaultView = 'graph',
  defaultStep,
  highlightNodeIds
}: {model: LensViewModel} & Omit<LensGraphPanelProps, 'scenario'>) {
  const stepCount = model.nodes.length;
  const hasLifetimes = Boolean(model.lifetimes?.length);
  const [view, setView] = useState<'graph' | 'buffers'>(hasLifetimes ? defaultView : 'graph');
  const [step, setStep] = useState(Math.min(defaultStep ?? stepCount, stepCount));
  const [isPlaying, setIsPlaying] = useState(false);
  const stepOfNode = useMemo(
    () => new Map(model.nodes.map((node, index) => [node.id, index])),
    [model]
  );

  useEffect(() => {
    if (!isPlaying) return undefined;
    const handle = window.setInterval(() => {
      setStep(value => {
        if (value >= stepCount) {
          setIsPlaying(false);
          return value;
        }
        return value + 1;
      });
    }, 650);
    return () => window.clearInterval(handle);
  }, [isPlaying, stepCount]);

  const togglePlay = () => {
    if (isPlaying) return setIsPlaying(false);
    if (step >= stepCount) setStep(0);
    setIsPlaying(true);
  };
  const isPartial = step < stepCount;
  const runningNode = step > 0 ? model.nodes[step - 1] : undefined;
  const layout = useMemo(() => layoutLensGraph(model, NODE_LAYOUT), [model]);
  const [hover, setHover] = useState<Hover>(null);
  const flowGraph = useMemo(() => makeFlowGraph(model, layout), [model, layout]);
  const sceneLoader = useMemo(() => createFlowSceneLoader(flowGraph), [flowGraph]);
  const nodeById = useMemo(() => new Map(model.nodes.map(node => [node.id, node])), [model]);
  const heaviest = useMemo(() => getHeaviestEdgeIndices(model.edges, 5), [model]);
  const maximumBytes = Math.max(1, ...model.edges.map(edge => edge.bytes));
  const highlightKey = highlightNodeIds?.join('\n') ?? '';
  const highlighted = useMemo(
    () => (highlightKey ? new Set(highlightKey.split('\n')) : undefined),
    [highlightKey]
  );
  const isEdgeDimmed = (edge: LensViewEdge) =>
    highlighted !== undefined && !(highlighted.has(edge.from) && highlighted.has(edge.to));

  return (
    <figure
      className="meta-panel"
      data-hazards={showHazards}
      aria-label={`Command graph of ${model.title}`}
    >
      {compact ? null : (
        <div className="meta-panel__bar">
          <span className="meta-panel__title">{model.title}</span>
          <ul className="meta-stats">
            <li>
              <strong>{formatCount(model.recordedNodeCount)}</strong> kernels
              {model.collapsed ? (
                <>
                  {' '}
                  in <strong>{model.nodes.length}</strong> groups
                </>
              ) : null}
            </li>
            <li>
              <strong>{formatCount(model.recordedEdgeCount)}</strong> dependencies
            </li>
            <li>
              <strong>{formatBytes(model.resources.totalBytes)}</strong> in{' '}
              <strong>{formatCount(model.resources.count)}</strong> buffers
            </li>
            <li>
              <strong>{formatCount(model.totals.workgroups)}</strong> workgroups
            </li>
          </ul>
          <span className="meta-legend" aria-hidden="true">
            <span>
              <i data-kind="compute" />
              compute
            </span>
            <span>
              <i data-kind="copy" />
              copy
            </span>
          </span>
        </div>
      )}
      {compact ? null : (
        <div className="meta-controls">
          {hasLifetimes ? (
            <div className="meta-toggle" role="group" aria-label="View">
              <button
                type="button"
                aria-pressed={view === 'graph'}
                onClick={() => setView('graph')}
              >
                Graph
              </button>
              <button
                type="button"
                aria-pressed={view === 'buffers'}
                onClick={() => setView('buffers')}
              >
                Buffers
              </button>
            </div>
          ) : null}
          <button type="button" className="meta-play" onClick={togglePlay}>
            {isPlaying ? 'Pause' : step >= stepCount ? 'Replay' : 'Play'}
          </button>
          <label className="meta-slider meta-slider--wide">
            <span>
              Step <b>{step}</b> / {stepCount}
            </span>
            <input
              type="range"
              min={0}
              max={stepCount}
              step={1}
              value={step}
              onChange={event => {
                setIsPlaying(false);
                setStep(Number(event.target.value));
              }}
              aria-valuetext={
                runningNode ? `step ${step}: ${runningNode.label}` : 'before the first step'
              }
            />
          </label>
          <span className="meta-controls__now" aria-live="polite">
            {runningNode ? (
              <>
                ran <b>{runningNode.label}</b>
              </>
            ) : (
              'nothing has run'
            )}
          </span>
        </div>
      )}
      {showHazards && !compact && view === 'graph' ? (
        <ul className="meta-hazard-key" aria-label="Dependency kinds">
          {(['raw', 'war', 'waw'] as const).map(kind => (
            <li key={kind} data-hazard={kind}>
              <i />
              {HAZARD_NAMES[kind]}
              <span>{formatCount(model.hazards[kind])}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {view === 'buffers' && model.lifetimes ? (
        <LifetimeChart model={model} step={step} />
      ) : (
        <>
          <div className="meta-scroll">
            <div
              className="meta-stage"
              style={{
                minWidth: Math.min(layout.width, MINIMUM_STAGE_WIDTH),
                aspectRatio: `${layout.width} / ${layout.height}`
              }}
            >
              {isPartial || highlighted ? null : (
                <div className="meta-stage__canvas" aria-hidden="true">
                  <GpuCanvas
                    scene={sceneLoader}
                    fallback={null}
                    seed={1}
                    label={`Data flowing through ${model.title}`}
                    aspectRatio={layout.width / layout.height}
                  />
                </div>
              )}
              <svg
                className="meta-stage__svg"
                viewBox={`0 0 ${layout.width} ${layout.height}`}
                role="img"
                aria-label={`${model.nodes.length} kernel groups and ${model.edges.length} dependencies`}
              >
                <g>
                  {layout.edges.map((edge, index) => {
                    const data = model.edges[index];
                    const edgeState = isPartial
                      ? (stepOfNode.get(data.to) ?? 0) < step
                        ? 'done'
                        : (stepOfNode.get(data.from) ?? 0) < step
                          ? 'live'
                          : 'idle'
                      : undefined;
                    const active =
                      hover?.kind === 'edge'
                        ? hover.index === index
                        : hover?.kind === 'node' &&
                          (hover.id === data.from || hover.id === data.to);
                    return (
                      <g
                        key={`${edge.from}>${edge.to}`}
                        opacity={isEdgeDimmed(data) ? DIMMED_EDGE_OPACITY : undefined}
                      >
                        <path
                          className="lens-edge"
                          data-active={active}
                          data-skip={edge.span > 1}
                          data-state={edgeState}
                          data-hazard={getEdgeHazard(data)}
                          d={getEdgePath(edge)}
                          strokeWidth={getEdgeWidth(data.bytes, maximumBytes)}
                          // The hazard colours in meta.css also set stroke-opacity, which would
                          // make dashed layer-skipping edges as strong as direct ones.
                          style={
                            showHazards && edge.span > 1 && !active && edgeState !== 'idle'
                              ? {strokeOpacity: 0.32}
                              : undefined
                          }
                        />
                        <path
                          className="lens-edge-hit"
                          d={getEdgePath(edge)}
                          onPointerEnter={() => setHover({kind: 'edge', index})}
                          onPointerLeave={() => setHover(null)}
                        />
                      </g>
                    );
                  })}
                </g>
                <g aria-hidden="true">
                  {[...heaviest].map(index => {
                    const edge = layout.edges[index];
                    const data = model.edges[index];
                    if (edge.x1 <= edge.x0 || edge.span > 1 || isEdgeDimmed(data)) return null;
                    return (
                      <text
                        key={index}
                        className="lens-edge-label"
                        x={(edge.x0 + edge.x1) / 2}
                        y={(edge.y0 + edge.y1) / 2 - 4}
                        textAnchor="middle"
                      >
                        {formatBytes(data.bytes)}
                      </text>
                    );
                  })}
                </g>
                <g>
                  {layout.nodes.map(placed => {
                    const node = nodeById.get(placed.id);
                    if (!node) return null;
                    const nodeStep = stepOfNode.get(node.id) ?? 0;
                    const nodeState = isPartial
                      ? nodeStep < step - 1
                        ? 'done'
                        : nodeStep === step - 1
                          ? 'running'
                          : 'pending'
                      : undefined;
                    return (
                      <g
                        key={node.id}
                        className="lens-node"
                        data-kind={node.kind}
                        data-state={nodeState}
                        data-active={hover?.kind === 'node' && hover.id === node.id}
                        data-dimmed={highlighted && !highlighted.has(node.id) ? 'true' : undefined}
                        opacity={
                          highlighted && !highlighted.has(node.id) ? DIMMED_NODE_OPACITY : undefined
                        }
                        transform={`translate(${placed.x} ${placed.y})`}
                        tabIndex={0}
                        onPointerEnter={() => setHover({kind: 'node', id: node.id})}
                        onPointerLeave={() => setHover(null)}
                        onClick={() =>
                          setHover(current =>
                            current?.kind === 'node' && current.id === node.id
                              ? null
                              : {kind: 'node', id: node.id}
                          )
                        }
                        onFocus={() => setHover({kind: 'node', id: node.id})}
                        onBlur={() => setHover(null)}
                      >
                        <title>{node.label}</title>
                        <rect width={placed.width} height={placed.height} rx={7} />
                        <text className="lens-node__label" x={10} y={17}>
                          {fitLabel(node.label, placed.width)}
                        </text>
                        <text className="lens-node__meta" x={10} y={32}>
                          {describeNode(node)}
                        </text>
                        {node.count > 1 ? (
                          <text
                            className="lens-node__badge"
                            x={placed.width - 8}
                            y={32}
                            textAnchor="end"
                          >
                            x{node.count}
                          </text>
                        ) : null}
                      </g>
                    );
                  })}
                </g>
              </svg>
            </div>
          </div>
          <p className="meta-scroll-hint">Scroll sideways; tap a kernel for details.</p>
        </>
      )}
      <figcaption className="meta-detail" aria-live="polite">
        {view === 'buffers' && model.lifetimes
          ? renderLifetimeDetail(model, step)
          : renderDetail(model, hover, nodeById)}
      </figcaption>
    </figure>
  );
}

/** Keeps a label inside its node; the full name stays in the node's tooltip. */
function fitLabel(label: string, width: number): string {
  const maximumCharacters = Math.max(8, Math.floor((width - 20) / 6.4));
  return label.length > maximumCharacters ? `${label.slice(0, maximumCharacters - 1)}…` : label;
}

function describeNode(node: LensViewNode): string {
  const parts: string[] = [node.kind];
  if (node.workgroups > 0) parts.push(`${formatCount(node.workgroups)} wg`);
  if (node.milliseconds !== undefined) parts.push(`${node.milliseconds.toFixed(2)} ms`);
  return parts.join(' / ');
}

function renderDetail(model: LensViewModel, hover: Hover, nodeById: Map<string, LensViewNode>) {
  if (hover?.kind === 'node') {
    const node = nodeById.get(hover.id);
    if (node) {
      return (
        <>
          <strong>{node.label}</strong>
          <span>
            <b>{node.kind}</b>
            {node.count > 1 ? <> x{node.count} unrolled</> : null}
          </span>
          {node.operators.length ? <span>{node.operators.join(', ')}</span> : null}
          <span>
            <b>{formatCount(node.workgroups)}</b> workgroups
          </span>
          <span>
            reads <b>{formatBytes(node.readBytes)}</b>, writes <b>{formatBytes(node.writeBytes)}</b>
          </span>
        </>
      );
    }
  }
  if (hover?.kind === 'edge') {
    const edge: LensViewEdge | undefined = model.edges[hover.index];
    if (edge) {
      return (
        <>
          <strong>
            {edge.resources.join(', ')}
            {edge.resourceCount > edge.resources.length ? ', ...' : ''}
          </strong>
          <span>
            <b>{formatBytes(edge.bytes)}</b> in <b>{edge.resourceCount}</b>{' '}
            {edge.resourceCount === 1 ? 'buffer' : 'buffers'}
          </span>
          <span>
            {getEdgeHazard(edge) === 'raw'
              ? 'written by one kernel, read by the next'
              : HAZARD_NAMES[getEdgeHazard(edge)]}
            {edge.hazards && edge.hazards.length > 1
              ? ` (also ${edge.hazards
                  .filter(kind => kind !== getEdgeHazard(edge))
                  .map(kind => HAZARD_NAMES[kind])
                  .join(', ')})`
              : null}
          </span>
        </>
      );
    }
  }
  return (
    <>
      <span>
        Hover or tap a kernel or a dependency.
        {model.collapsed
          ? ' Unrolled loops and repeated passes are folded into one node (xN).'
          : ''}
      </span>
    </>
  );
}

function renderLifetimeDetail(model: LensViewModel, step: number) {
  const lifetimes = model.lifetimes ?? [];
  const live = lifetimes.filter(lifetime => getLiveBytes([lifetime], step) > 0).length;
  const hasTransient = lifetimes.some(lifetime => lifetime.ownership === 'transient');
  return (
    <>
      <span>
        <b>{formatBytes(getLiveBytes(lifetimes, step))}</b> alive in <b>{live}</b> of{' '}
        {lifetimes.length} buffers
      </span>
      <span>
        {hasTransient ? 'peak' : 'peak logical live'}{' '}
        <b>{formatBytes(getPeakLiveBytes(lifetimes, model.nodes.length))}</b>
      </span>
      <span>
        {hasTransient
          ? 'Buffers that never overlap in time can share one block of memory.'
          : 'All buffers are imported, so all stay allocated for the whole run; a bar is the logical span between first write and last use.'}
        {model.lifetimesOmitted ? ` ${model.lifetimesOmitted} smaller buffers not shown.` : ''}
      </span>
    </>
  );
}

function getEdgeWidth(bytes: number, maximumBytes: number): number {
  return 1 + 2.4 * (Math.log10(1 + bytes) / Math.log10(1 + maximumBytes));
}

function getHeaviestEdgeIndices(edges: readonly LensViewEdge[], count: number): Set<number> {
  return new Set(
    edges
      .map((edge, index) => ({bytes: edge.bytes, index}))
      .filter(entry => entry.bytes >= 1024)
      .sort((a, b) => b.bytes - a.bytes)
      .slice(0, count)
      .map(entry => entry.index)
  );
}

export function makeFlowGraph(model: LensViewModel, layout: GraphLayout): FlowGraph {
  const maximumBytes = Math.max(1, ...model.edges.map(edge => edge.bytes));
  const lastLayer = Math.max(1, ...layout.nodes.map(node => node.layer));
  const layerByNode = new Map(layout.nodes.map(node => [node.id, node.layer]));
  const kindByNode = new Map(model.nodes.map(node => [node.id, node.kind]));
  const indexByNode = new Map(layout.nodes.map((node, index) => [node.id, index]));
  return {
    width: layout.width,
    height: layout.height,
    nodes: layout.nodes.map(node => {
      const kind = kindByNode.get(node.id);
      return {
        x: node.x,
        y: node.y,
        width: node.width,
        height: node.height,
        kind: kind === 'copy' ? 1 : kind === 'compute' ? 0 : 3,
        progress: node.layer / lastLayer
      };
    }),
    edges: layout.edges.map((edge, index) => ({
      x0: edge.x0,
      y0: edge.y0,
      x1: edge.x1,
      y1: edge.y1,
      controlX0: edge.controlX0,
      controlY0: edge.controlY0,
      controlX1: edge.controlX1,
      controlY1: edge.controlY1,
      weight:
        edge.span > 1 ? 0 : Math.log10(1 + model.edges[index].bytes) / Math.log10(1 + maximumBytes),
      progressStart: (layerByNode.get(edge.from) ?? 0) / lastLayer,
      progressEnd: (layerByNode.get(edge.to) ?? 0) / lastLayer,
      fromNode: indexByNode.get(edge.from),
      toNode: indexByNode.get(edge.to)
    }))
  };
}
