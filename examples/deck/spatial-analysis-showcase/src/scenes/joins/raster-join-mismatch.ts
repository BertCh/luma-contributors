// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {addKernelPass} from '../../engine/mode-kernels';

/** Classify mismatches only for complete producers and observations inside the compared extent. */
export function addRasterJoinMismatchNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    positions: GraphDataView<'float32x2'>;
    parameters: GraphDataView<'float32'>;
    rasterZones: GraphDataView<'uint32'>;
    exactZones: GraphDataView<'uint32'>;
    producerOverflow: GraphDataView<'uint32'>;
    exactOverflow: GraphDataView<'uint32'>;
    exactUncertainCount: GraphDataView<'uint32'>;
    width: number;
    height: number;
    output: GraphDataView<'uint32'>;
    /** Same GPU extent classification used by mismatch, for readout denominators. */
    extentMembership?: GraphDataView<'uint32'>;
  }
): void {
  validateRequirement(
    'mismatch',
    props.positions.byteOffset % Float32Array.BYTES_PER_ELEMENT === 0,
    `positions.byteOffset must be aligned to 4 bytes; received ${props.positions.byteOffset}`
  );
  validateRequirement(
    'mismatch',
    props.positions.byteStride % Float32Array.BYTES_PER_ELEMENT === 0,
    `positions.byteStride must be aligned to 4 bytes; received ${props.positions.byteStride}`
  );
  const positionComponentStride = props.positions.byteStride / Float32Array.BYTES_PER_ELEMENT;
  const extentClassification = `let pointPosition = vec2<f32>(pointPositions[pointPositionsOffset + index * ${positionComponentStride}u], pointPositions[pointPositionsOffset + index * ${positionComponentStride}u + 1u]);
      let coordinate = (pointPosition - vec2<f32>(placement[placementOffset], placement[placementOffset + 1u])) / vec2<f32>(placement[placementOffset + 2u], placement[placementOffset + 3u]);
      let inside = all(coordinate >= vec2<f32>(0.0)) && coordinate.x < ${props.width}.0 && coordinate.y < ${props.height}.0;
`;
  // A separate parallel pass stays within the eight-storage-binding device limit.
  if (props.extentMembership)
    addKernelPass(graph, {
      id: 'raster-join-extent-membership',
      invocationCount: props.positions.length,
      bindings: [
        {name: 'pointPositions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'placement', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'extentMembership', view: props.extentMembership, type: 'u32', access: 'read_write'}
      ],
      body: `${extentClassification}
      extentMembership[extentMembershipOffset + index] = select(0u, 1u, inside);`
    });
  addKernelPass(graph, {
    id: 'mismatch',
    invocationCount: props.positions.length,
    bindings: [
      {name: 'rasterZone', view: props.rasterZones, type: 'u32', access: 'read'},
      {name: 'exactZone', view: props.exactZones, type: 'u32', access: 'read'},
      {name: 'pointPositions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'placement', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'producerOverflow', view: props.producerOverflow, type: 'u32', access: 'read'},
      {name: 'exactOverflow', view: props.exactOverflow, type: 'u32', access: 'read'},
      {name: 'exactUncertainCount', view: props.exactUncertainCount, type: 'u32', access: 'read'},
      {name: 'mismatch', view: props.output, type: 'u32', access: 'read_write'}
    ],
    // The shared kernel helper expresses offsets in scalar 32-bit components.
    body: `${extentClassification}      let complete = producerOverflow[producerOverflowOffset] == 0u && exactOverflow[exactOverflowOffset] == 0u && exactUncertainCount[exactUncertainCountOffset] == 0u;
      mismatch[mismatchOffset + index] = select(0u, 1u, complete && inside && rasterZone[rasterZoneOffset + index] != exactZone[exactZoneOffset + index]);`
  });
}
