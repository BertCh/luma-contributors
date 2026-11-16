// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonOverlayOutput} from './gpu-polygon-overlay';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Representative boundary provenance for each assembled polygon, including its holes. */
export type GPUPolygonOverlayFeatureSources = {
  /** Lowest (operand, source row) among retained boundary edges; 0xffffffff for unused slots. */
  sourceRows: GraphDataView<'uint32'>;
  /** Operand of the representative (0=left, 1=right); 0xffffffff for unused slots. */
  operandIds: GraphDataView<'uint32'>;
  /** One when retained boundary edges have more than one (operand, source row). */
  mixed: GraphDataView<'uint32'>;
};

/** Maps raw ring ownership to assembly's ascending-shell output order, then reduces edge owners. */
export function createOverlayFeatureSourceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    ringShells: GraphDataView<'uint32'>;
    ringAreas: GraphDataView<'float32'>;
    ringCount: GraphDataView<'uint32'>;
    segmentRings: GraphDataView<'uint32'>;
    boundaryCount: GraphDataView<'uint32'>;
    sourceRows: GraphDataView<'uint32'>;
    operandIds: GraphDataView<'uint32'>;
    output: GPUPolygonOverlayFeatureSources;
  }
): GPUCommandNode<Parameters>[] {
  const {id, output} = props;
  const capacity = props.ringShells.length;
  const flags = createTransientView(graph, `${id}-source-shell-flags`, 'uint32', capacity);
  const ranks = createTransientView(graph, `${id}-source-shell-ranks`, 'uint32', capacity);
  const minimum = createTransientView(graph, `${id}-source-minimum`, 'uint32', capacity);
  const maximum = createTransientView(graph, `${id}-source-maximum`, 'uint32', capacity);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-source-shells`,
      operation: 'GPUPolygonOverlay',
      variant: 'feature-source-shells',
      invocationCount: capacity,
      bindings: [
        {name: 'shells', view: props.ringShells, type: 'u32', access: 'read'},
        {name: 'areas', view: props.ringAreas, type: 'f32', access: 'read'},
        {name: 'ringCount', view: props.ringCount, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
        {name: 'minimum', view: minimum, type: 'u32', access: 'read_write'},
        {name: 'maximum', view: maximum, type: 'u32', access: 'read_write'}
      ],
      body: `flags[flagsOffset + index] = select(0u, 1u, index < ringCount[ringCountOffset] && shells[shellsOffset + index] == index && abs(areas[areasOffset + index]) > 0.0);
minimum[minimumOffset + index] = 0xffffffffu;
maximum[maximumOffset + index] = 0u;`
    }),
    ...new GPUScan({
      id: `${id}-source-shell-scan`,
      input: flags,
      output: ranks,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-reduce-feature-sources`,
      operation: 'GPUPolygonOverlay',
      variant: 'reduce-feature-sources',
      invocationCount: props.segmentRings.length,
      bindings: [
        {name: 'segmentRings', view: props.segmentRings, type: 'u32', access: 'read'},
        {name: 'shells', view: props.ringShells, type: 'u32', access: 'read'},
        {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
        {name: 'boundaryCount', view: props.boundaryCount, type: 'u32', access: 'read'},
        {name: 'sourceRows', view: props.sourceRows, type: 'u32', access: 'read'},
        {name: 'operandIds', view: props.operandIds, type: 'u32', access: 'read'},
        {name: 'minimum', view: minimum, type: 'atomic<u32>', access: 'read_write'},
        {name: 'maximum', view: maximum, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `if (index >= boundaryCount[boundaryCountOffset]) { return; }
let ring = segmentRings[segmentRingsOffset + index];
if (ring == 0xffffffffu) { return; }
let shell = shells[shellsOffset + ring];
if (shell == 0xffffffffu) { return; }
let feature = ranks[ranksOffset + shell];
let owner = (operandIds[operandIdsOffset + index] << 31u) | sourceRows[sourceRowsOffset + index];
atomicMin(&minimum[minimumOffset + feature], owner);
atomicMax(&maximum[maximumOffset + feature], owner);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-publish-feature-sources`,
      operation: 'GPUPolygonOverlay',
      variant: 'publish-feature-sources',
      invocationCount: capacity,
      bindings: [
        {name: 'minimum', view: minimum, type: 'u32', access: 'read'},
        {name: 'maximum', view: maximum, type: 'u32', access: 'read'},
        {name: 'sourceRows', view: output.sourceRows, type: 'u32', access: 'read_write'},
        {name: 'operandIds', view: output.operandIds, type: 'u32', access: 'read_write'},
        {name: 'mixed', view: output.mixed, type: 'u32', access: 'read_write'}
      ],
      body: `let owner = minimum[minimumOffset + index];
sourceRows[sourceRowsOffset + index] = select(owner & 0x7fffffffu, 0xffffffffu, owner == 0xffffffffu);
operandIds[operandIdsOffset + index] = select(owner >> 31u, 0xffffffffu, owner == 0xffffffffu);
mixed[mixedOffset + index] = select(0u, 1u, owner != 0xffffffffu && owner != maximum[maximumOffset + index]);`
    })
  ];
}

/** Validates optional provenance at the outermost contributor construction, including buffers. */
export function validateOverlayFeatureSourceOutput(
  id: string,
  output: GPUPolygonOverlayOutput
): void {
  const ringCapacity = output.geometry.ringOffsets.length - 1;
  if (output.featureSources) {
    for (const [name, view] of Object.entries(output.featureSources)) {
      validateSinglePackedView(view, ['uint32'], id);
      validateRequirement(
        id,
        view.length === ringCapacity,
        `output.featureSources.${name}.length must equal ring capacity`
      );
    }
    validateOutlineOutputPasses(
      id,
      {
        positions: output.geometry.positions,
        featureOffsets: output.geometry.featureOffsets,
        polygonOffsets: output.geometry.polygonOffsets,
        ringOffsets: output.geometry.ringOffsets,
        ...Object.fromEntries(
          Object.entries(output.status).map(([name, view]) => [`status.${name}`, view])
        ),
        ...Object.fromEntries(
          Object.entries(output.boundary).filter(([name]) => name !== 'status')
        ),
        ...Object.fromEntries(
          Object.entries(output.boundary.status).map(([name, view]) => [
            `boundary.status.${name}`,
            view
          ])
        ),
        ...Object.fromEntries(
          Object.entries(output.featureSources).map(([name, view]) => [
            `featureSources.${name}`,
            view
          ])
        )
      },
      [['featureSources.sourceRows', 'featureSources.operandIds', 'featureSources.mixed']]
    );
  }
}
