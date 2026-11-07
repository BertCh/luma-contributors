// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The command graph of a recorded lens run: an SVG DAG (compute and copy nodes, resources as edges
// with byte sizes) with a live luma.gl layer behind it that animates data flowing along the edges.

import {Suspense, useMemo, useState} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {formatBytes, formatCount} from '../../lens/format';
import {getEdgePath, layoutLensGraph, type GraphLayout} from '../../lens/layout-graph';
import type {LensViewEdge, LensViewModel, LensViewNode} from '../../lens/types';
import {useLensViewModel} from '../../lens/use-lens-view-model';
import {createFlowSceneLoader, type FlowGraph} from './scenes/flow-scene';
import './meta.css';

export type LensGraphPanelProps = {
  /** A scenario id such as `gpu-spatial-analysis.kernel-density`, or an already loaded model. */
  scenario: string | LensViewModel;
  /** Hide the stats and legend bar. */
  compact?: boolean;
};

const NODE_LAYOUT = {
  nodeWidth: 188,
  nodeHeight: 42,
  layerGap: 64,
  rowGap: 14,
  padding: 22,
  targetWidth: 1040
};
const MINIMUM_STAGE_WIDTH = 620;

export function LensGraphPanel({scenario, compact = false}: LensGraphPanelProps) {
  return typeof scenario === 'string' ? (
    <Suspense fallback={<div className="meta-panel meta-skeleton">loading {scenario}</div>}>
      <LoadedLensGraphPanel scenarioId={scenario} compact={compact} />
    </Suspense>
  ) : (
    <LensGraphView model={scenario} compact={compact} />
  );
}

function LoadedLensGraphPanel({scenarioId, compact}: {scenarioId: string; compact: boolean}) {
  const model = useLensViewModel(scenarioId);
  return model ? (
    <LensGraphView model={model} compact={compact} />
  ) : (
    <div className="meta-panel meta-skeleton">No recorded run for {scenarioId}</div>
  );
}

type Hover = {kind: 'node'; id: string} | {kind: 'edge'; index: number} | null;

function LensGraphView({model, compact}: {model: LensViewModel; compact: boolean}) {
  const layout = useMemo(() => layoutLensGraph(model, NODE_LAYOUT), [model]);
  const [hover, setHover] = useState<Hover>(null);
  const flowGraph = useMemo(() => makeFlowGraph(model, layout), [model, layout]);
  const sceneLoader = useMemo(() => createFlowSceneLoader(flowGraph), [flowGraph]);
  const nodeById = useMemo(() => new Map(model.nodes.map(node => [node.id, node])), [model]);
  const heaviest = useMemo(() => getHeaviestEdgeIndices(model.edges, 5), [model]);
  const maximumBytes = Math.max(1, ...model.edges.map(edge => edge.bytes));

  return (
    <figure className="meta-panel" aria-label={`Command graph of ${model.title}`}>
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
      <div className="meta-scroll">
        <div
          className="meta-stage"
          style={{
            minWidth: Math.min(layout.width, MINIMUM_STAGE_WIDTH),
            aspectRatio: `${layout.width} / ${layout.height}`
          }}
        >
          <div className="meta-stage__canvas" aria-hidden="true">
            <GpuCanvas
              scene={sceneLoader}
              fallback={null}
              seed={1}
              label={`Data flowing through ${model.title}`}
              aspectRatio={layout.width / layout.height}
            />
          </div>
          <svg
            className="meta-stage__svg"
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            role="img"
            aria-label={`${model.nodes.length} kernel groups and ${model.edges.length} dependencies`}
          >
            <g>
              {layout.edges.map((edge, index) => {
                const data = model.edges[index];
                const active =
                  hover?.kind === 'edge'
                    ? hover.index === index
                    : hover?.kind === 'node' && (hover.id === data.from || hover.id === data.to);
                return (
                  <g key={`${edge.from}>${edge.to}`}>
                    <path
                      className="lens-edge"
                      data-active={active}
                      data-skip={edge.span > 1}
                      d={getEdgePath(edge)}
                      strokeWidth={getEdgeWidth(data.bytes, maximumBytes)}
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
                if (edge.x1 <= edge.x0 || edge.span > 1) return null;
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
                return (
                  <g
                    key={node.id}
                    className="lens-node"
                    data-kind={node.kind}
                    data-active={hover?.kind === 'node' && hover.id === node.id}
                    transform={`translate(${placed.x} ${placed.y})`}
                    tabIndex={0}
                    onPointerEnter={() => setHover({kind: 'node', id: node.id})}
                    onPointerLeave={() => setHover(null)}
                    onFocus={() => setHover({kind: 'node', id: node.id})}
                    onBlur={() => setHover(null)}
                  >
                    <title>{node.label}</title>
                    <rect width={placed.width} height={placed.height} rx={7} />
                    <text className="lens-node__label" x={10} y={17}>
                      {node.label}
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
      <figcaption className="meta-detail" aria-live="polite">
        {renderDetail(model, hover, nodeById)}
      </figcaption>
    </figure>
  );
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
          <span>written by one kernel, read by the next</span>
        </>
      );
    }
  }
  return (
    <>
      <span>
        Hover a kernel or a dependency.
        {model.collapsed
          ? ' Unrolled loops and repeated passes are folded into one node (xN).'
          : ''}
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

function makeFlowGraph(model: LensViewModel, layout: GraphLayout): FlowGraph {
  const maximumBytes = Math.max(1, ...model.edges.map(edge => edge.bytes));
  const lastLayer = Math.max(1, ...layout.nodes.map(node => node.layer));
  const layerByNode = new Map(layout.nodes.map(node => [node.id, node.layer]));
  const kindByNode = new Map(model.nodes.map(node => [node.id, node.kind]));
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
      progressEnd: (layerByNode.get(edge.to) ?? 0) / lastLayer
    }))
  };
}
