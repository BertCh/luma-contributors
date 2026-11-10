// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The package stack: our three packages on top of published luma.gl. A call descends the chain.

import {useMemo} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {createFlowSceneLoader, type FlowGraph} from './scenes/flow-scene';
import './meta.css';
import './package-layers.css';

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

const CARD_WIDTH = 640;
const CARD_HEIGHT = 400;
const CARD_NODE_WIDTH = 168;
const CARD_NODE_HEIGHT = 92;
const CARD_COLUMNS = [34, 236, 438];
const CARD_ROW_TOPS = [64, 244];
const CARD_GAP = CARD_COLUMNS[1] - CARD_COLUMNS[0] - CARD_NODE_WIDTH;

/** The six layers as a graph. 'row' is the topic-page strip; 'card' snakes over two rows for 16:10. */
export function makePackageFlowGraph(arrangement: 'card' | 'row' = 'card'): FlowGraph {
  const lastIndex = PACKAGE_LAYERS.length - 1;
  const columnCount = CARD_COLUMNS.length;
  const nodes = PACKAGE_LAYERS.map((layer, index) => {
    const isCard = arrangement === 'card';
    const row = Math.floor(index / columnCount);
    // Row 2 runs right to left so consecutive layers stay adjacent.
    const column = row === 0 ? index : columnCount * 2 - 1 - index;
    return {
      x: isCard ? CARD_COLUMNS[column] : PADDING + index * (NODE_WIDTH + GAP),
      y: isCard ? CARD_ROW_TOPS[row] : TOP,
      width: isCard ? CARD_NODE_WIDTH : NODE_WIDTH,
      height: isCard ? CARD_NODE_HEIGHT : NODE_HEIGHT,
      kind: (layer.owner === 'ours' ? 0 : 3) as 0 | 3,
      progress: index / lastIndex
    };
  });
  const edges = nodes.slice(0, -1).map((node, index) => {
    const next = nodes[index + 1];
    const centerY = node.y + node.height / 2;
    const gap = arrangement === 'card' ? CARD_GAP : GAP;
    const edge = {
      x0: node.x + node.width,
      y0: centerY,
      x1: next.x,
      y1: centerY,
      controlX0: node.x + node.width + gap / 2,
      controlY0: centerY,
      controlX1: next.x - gap / 2,
      controlY1: centerY,
      weight: 0.7,
      progressStart: index / lastIndex,
      progressEnd: (index + 1) / lastIndex,
      fromNode: index,
      toNode: index + 1
    };
    if (arrangement === 'row' || next.y === node.y) {
      if (arrangement === 'card' && index >= columnCount) {
        // Row 2 flows right to left: leave by the left side, enter by the right side.
        return {
          ...edge,
          x0: node.x,
          x1: next.x + next.width,
          controlX0: node.x - gap / 2,
          controlX1: next.x + next.width + gap / 2
        };
      }
      return edge;
    }
    // The turn between rows is vertical, from the bottom center to the top center.
    const centerX = node.x + node.width / 2;
    const middleY = (node.y + node.height + next.y) / 2;
    return {
      ...edge,
      x0: centerX,
      y0: node.y + node.height,
      x1: next.x + next.width / 2,
      y1: next.y,
      controlX0: centerX,
      controlY0: middleY,
      controlX1: next.x + next.width / 2,
      controlY1: middleY
    };
  });
  return arrangement === 'card'
    ? {width: CARD_WIDTH, height: CARD_HEIGHT, nodes, edges}
    : {width: WIDTH, height: HEIGHT, nodes, edges};
}

/** `version` is the luma.gl release the packages are built on; pages read it from data. */
export function PackageLayers({version}: {version?: string}) {
  const loader = useMemo(() => createFlowSceneLoader(makePackageFlowGraph('row')), []);
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
              {version ? `published luma.gl ${version}` : 'published luma.gl'}
            </text>
            {PACKAGE_LAYERS.slice(0, -1).map((layer, index) => {
              const x = PADDING + index * (NODE_WIDTH + GAP) + NODE_WIDTH;
              const y = TOP + NODE_HEIGHT / 2;
              return (
                <path
                  key={`arrow-${layer.name}`}
                  className="layer-arrow"
                  d={`M${x + 4} ${y} H${x + GAP - 5} M${x + GAP - 11} ${y - 5} L${x + GAP - 4} ${y} L${x + GAP - 11} ${y + 5}`}
                />
              );
            })}
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
