// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The whole system in one picture: seven stages from a CPU table to a readback, with luma.gl
// animating data along the chain. Stage names match the public API calls.

import {useEffect, useMemo, useState} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {createFlowSceneLoader, type FlowGraph} from './scenes/flow-scene';
import './meta.css';

export type PipelineStage = {
  id: string;
  title: string;
  subtitle: string;
  caption: string;
  side: 'cpu' | 'gpu';
};

export const PIPELINE_STAGES: readonly PipelineStage[] = [
  {
    id: 'table',
    title: 'Table',
    subtitle: 'Arrow batches',
    caption: 'Rows arrive as record batches. Batch boundaries are kept.',
    side: 'cpu'
  },
  {
    id: 'vectors',
    title: 'GPU vectors',
    subtitle: 'one buffer per chunk',
    caption: 'Each column becomes a GPUVector: a list of GPUData chunks.',
    side: 'cpu'
  },
  {
    id: 'author',
    title: 'Author',
    subtitle: 'graph.add(operation)',
    caption: 'An operation adds kernel nodes that name what they read and write.',
    side: 'cpu'
  },
  {
    id: 'compile',
    title: 'Compile',
    subtitle: 'order + hazards',
    caption: 'Read and write sets give the dependency edges and the schedule.',
    side: 'cpu'
  },
  {
    id: 'encode',
    title: 'Encode',
    subtitle: 'command encoder',
    caption: 'Nodes become compute and copy passes in one command buffer.',
    side: 'cpu'
  },
  {
    id: 'dispatch',
    title: 'Dispatch',
    subtitle: 'workgroups',
    caption: 'The GPU runs every kernel as a grid of workgroups.',
    side: 'gpu'
  },
  {
    id: 'readback',
    title: 'Readback',
    subtitle: 'snapshots',
    caption: 'Results stay on the GPU for drawing, or copy out to the CPU.',
    side: 'gpu'
  }
];

const NODE_WIDTH = 150;
const NODE_HEIGHT = 64;
const GAP = 52;
const PADDING = 20;
const WIDTH =
  PADDING * 2 + PIPELINE_STAGES.length * NODE_WIDTH + (PIPELINE_STAGES.length - 1) * GAP;
const HEIGHT = 128;
const ROW_TOP = 40;
const STAGE_SECONDS = 1.7;

function makeFlowGraph(): FlowGraph {
  const nodes = PIPELINE_STAGES.map((stage, index) => ({
    x: PADDING + index * (NODE_WIDTH + GAP),
    y: ROW_TOP,
    width: NODE_WIDTH,
    height: NODE_HEIGHT,
    kind: (stage.side === 'gpu' ? 1 : 0) as 0 | 1,
    progress: index / (PIPELINE_STAGES.length - 1)
  }));
  const edges = nodes.slice(0, -1).map((node, index) => ({
    x0: node.x + node.width,
    y0: node.y + node.height / 2,
    x1: nodes[index + 1].x,
    y1: node.y + node.height / 2,
    controlX0: node.x + node.width + GAP / 2,
    controlY0: node.y + node.height / 2,
    controlX1: nodes[index + 1].x - GAP / 2,
    controlY1: node.y + node.height / 2,
    weight: 0.8,
    progressStart: index / (PIPELINE_STAGES.length - 1),
    progressEnd: (index + 1) / (PIPELINE_STAGES.length - 1)
  }));
  return {width: WIDTH, height: HEIGHT, nodes, edges};
}

export function PipelineDiagram() {
  const loader = useMemo(() => createFlowSceneLoader(makeFlowGraph()), []);
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (paused || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const timer = window.setInterval(
      () => setActive(current => (current + 1) % PIPELINE_STAGES.length),
      STAGE_SECONDS * 1000
    );
    return () => window.clearInterval(timer);
  }, [paused]);

  const select = (index: number) => {
    setActive(index);
    setPaused(true);
  };

  return (
    <div className="meta-canvas" onPointerLeave={() => setPaused(false)}>
      <div className="meta-scroll">
        <div className="meta-stage" style={{minWidth: 760, aspectRatio: `${WIDTH} / ${HEIGHT}`}}>
          <div className="meta-stage__canvas" aria-hidden="true">
            <GpuCanvas
              scene={loader}
              fallback={null}
              seed={1}
              label="Data flowing from a table to a readback"
              aspectRatio={WIDTH / HEIGHT}
            />
          </div>
          <svg
            className="meta-stage__svg"
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            role="img"
            aria-label="Pipeline: table, GPU vectors, author, compile, encode, dispatch, readback"
          >
            <text className="pipeline-side" x={PADDING} y={24}>
              CPU: build
            </text>
            <text className="pipeline-side" x={PADDING + 5 * (NODE_WIDTH + GAP)} y={24}>
              GPU: run
            </text>
            {PIPELINE_STAGES.map((stage, index) => (
              <g
                key={stage.id}
                className="pipeline-node"
                data-active={index === active}
                data-side={stage.side}
                transform={`translate(${PADDING + index * (NODE_WIDTH + GAP)} ${ROW_TOP})`}
                tabIndex={0}
                onPointerEnter={() => select(index)}
                onFocus={() => select(index)}
              >
                <rect width={NODE_WIDTH} height={NODE_HEIGHT} rx={10} />
                <text className="pipeline-node__title" x={14} y={28}>
                  {stage.title}
                </text>
                <text className="pipeline-node__meta" x={14} y={47}>
                  {stage.subtitle}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
      <ol className="pipeline-steps" style={{padding: 'var(--space-3) var(--space-4)'}}>
        {PIPELINE_STAGES.map((stage, index) => (
          <li key={stage.id} data-active={index === active} onPointerEnter={() => select(index)}>
            <strong>{stage.title}</strong>
            {stage.caption}
          </li>
        ))}
      </ol>
    </div>
  );
}
