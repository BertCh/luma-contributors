// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The package stack: our three packages on top of published luma.gl. A call descends the chain.

import {useMemo} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {createFlowSceneLoader, type FlowGraph} from './scenes/flow-scene';
import './meta.css';

type LayerSpec = {name: string; role: string; owner: 'ours' | 'upstream'};

export const PACKAGE_LAYERS: readonly LayerSpec[] = [
  {name: '@luma-contributors/kepler', role: 'kepler.gl adapters', owner: 'ours'},
  {name: '@luma-contributors/lenses', role: 'records to pictures', owner: 'ours'},
  {name: '@luma-contributors/analysis', role: 'GPU operations', owner: 'ours'},
  {name: '@luma.gl/graph', role: 'kernels, command graphs', owner: 'upstream'},
  {name: '@luma.gl/core', role: 'device, buffers, passes', owner: 'upstream'},
  {name: '@luma.gl/shaders', role: 'shader modules', owner: 'upstream'}
];

const NODE_WIDTH = 188;
const NODE_HEIGHT = 58;
const GAP = 40;
const PADDING = 22;
const TOP = 46;
const WIDTH = PADDING * 2 + PACKAGE_LAYERS.length * NODE_WIDTH + (PACKAGE_LAYERS.length - 1) * GAP;
const HEIGHT = TOP + NODE_HEIGHT + 34;

function makeFlowGraph(): FlowGraph {
  const nodes = PACKAGE_LAYERS.map((layer, index) => ({
    x: PADDING + index * (NODE_WIDTH + GAP),
    y: TOP,
    width: NODE_WIDTH,
    height: NODE_HEIGHT,
    kind: (layer.owner === 'ours' ? 0 : 3) as 0 | 3,
    progress: index / (PACKAGE_LAYERS.length - 1)
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
    weight: 0.7,
    progressStart: index / (PACKAGE_LAYERS.length - 1),
    progressEnd: (index + 1) / (PACKAGE_LAYERS.length - 1)
  }));
  return {width: WIDTH, height: HEIGHT, nodes, edges};
}

export function PackageLayers() {
  const loader = useMemo(() => createFlowSceneLoader(makeFlowGraph()), []);
  const groupX = (index: number) => PADDING + index * (NODE_WIDTH + GAP) - 10;
  return (
    <div className="meta-canvas">
      <div className="meta-scroll">
        <div className="meta-stage" style={{minWidth: 900, aspectRatio: `${WIDTH} / ${HEIGHT}`}}>
          <div className="meta-stage__canvas" aria-hidden="true">
            <GpuCanvas
              scene={loader}
              fallback={null}
              seed={1}
              label="A call descending the package stack"
              aspectRatio={WIDTH / HEIGHT}
            />
          </div>
          <svg
            className="meta-stage__svg"
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            role="img"
            aria-label="kepler depends on lenses, which depend on analysis, which builds on luma.gl graph, core and shaders"
          >
            <rect
              className="layer-group"
              x={groupX(0)}
              y={14}
              width={3 * (NODE_WIDTH + GAP) - GAP + 20}
              height={HEIGHT - 20}
              rx={12}
            />
            <rect
              className="layer-group"
              x={groupX(3)}
              y={14}
              width={3 * (NODE_WIDTH + GAP) - GAP + 20}
              height={HEIGHT - 20}
              rx={12}
            />
            <text className="layer-group-label" x={groupX(0) + 12} y={34}>
              this repository
            </text>
            <text className="layer-group-label" x={groupX(3) + 12} y={34}>
              published luma.gl 10.0.0-alpha.10
            </text>
            {PACKAGE_LAYERS.map((layer, index) => (
              <g
                key={layer.name}
                className="layer-node"
                data-owner={layer.owner}
                transform={`translate(${PADDING + index * (NODE_WIDTH + GAP)} ${TOP})`}
              >
                <rect width={NODE_WIDTH} height={NODE_HEIGHT} rx={9} />
                <text className="layer-node__title" x={12} y={25}>
                  {layer.name.replace('@luma-contributors/', '')}
                </text>
                <text className="layer-node__meta" x={12} y={43}>
                  {layer.role}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
    </div>
  );
}
