// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSpatialAnalysisPackedKernelNodes} from '../geographic-distribution/spatial-analysis-packed-kernel';
import {
  createTransientView,
  GPUGroupAggregation,
  GPUReduction,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createSaturatedCountScanNodes} from '../../analysis-core/kernels/scan-total';
import {createLoadBalancedForEachNode} from '../../analysis-core/kernels/load-balanced-for-each';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Segment sample counts are converted from float32, whose integer precision ends at 2^24. */
const BUILDING_TESSELLATION_LOCAL_MAXIMUM_SAMPLES_PER_SEGMENT = 16777216;

/** Stable boundary sampling and the building-to-sample index. All sizes are prevalidated. @internal */
export function addBuildingTessellationSamples<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    segments: GraphDataView<'float32x4'>;
    segmentBuildingIds: GraphDataView<'uint32'>;
    buildingCount: number;
    parameters: GraphDataView<'float32'>;
    sampleCapacity: number;
    samplingStatus: GraphDataView<'uint32x4'>;
  }
) {
  const {id, buildingCount, sampleCapacity} = props;
  const segmentCount = props.segments.length;
  const counts = createTransientView(graph, `${id}-segment-counts`, 'uint32', segmentCount);
  const offsets = createTransientView(graph, `${id}-segment-offsets`, 'uint32', segmentCount + 1);
  const invalidFlags = createTransientView(graph, `${id}-invalid-segments`, 'uint32', segmentCount);
  const invalidCount = createTransientView(graph, `${id}-invalid-count`, 'uint32', 1);
  const total = createTransientView(graph, `${id}-sample-total`, 'uint32', 1);
  // One binding carries coordinates followed by owners; the grid reads only the coordinate view.
  const sampleData = createTransientView(graph, `${id}-sample-data`, 'uint32', sampleCapacity * 3);
  const samples = graph.createDataView(sampleData.buffer, {
    format: 'float32x2',
    length: sampleCapacity
  });
  const owners = graph.createDataView(sampleData.buffer, {
    format: 'uint32',
    byteOffset: sampleCapacity * 8,
    length: sampleCapacity
  });
  const sampleRows = createTransientView(graph, `${id}-sample-rows`, 'uint32', sampleCapacity);
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-count-segments`,
      operation: id,
      invocationCount: segmentCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'owners', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
        {name: 'invalid', view: invalidFlags, type: 'u32', access: 'read_write'}
      ],
      body: `let base=segmentsOffset+4u*index;
let start=vec2<f32>(segments[base],segments[base+1u]);
let end=vec2<f32>(segments[base+2u],segments[base+3u]);
let span=distance(start,end);
let spacing=parameters[parametersOffset];
let required=ceil(span/spacing);
let valid=isFiniteValue(spacing)&&spacing>0.0&&isFiniteValue(start.x)&&isFiniteValue(start.y)&&isFiniteValue(end.x)&&isFiniteValue(end.y)&&span>0.0&&isFiniteValue(required)&&required<=${BUILDING_TESSELLATION_LOCAL_MAXIMUM_SAMPLES_PER_SEGMENT}.0&&owners[ownersOffset+index]<${buildingCount}u;
counts[countsOffset+index]=select(0u,u32(clamp(required,0.0,${BUILDING_TESSELLATION_LOCAL_MAXIMUM_SAMPLES_PER_SEGMENT}.0)),valid);
invalid[invalidOffset+index]=u32(!valid);`
    })
  );
  graph.add(
    createSaturatedCountScanNodes(graph, {
      id: `${id}-sample-scan`,
      operation: id,
      counts,
      offsets,
      rowCount: segmentCount,
      total
    })
  );
  graph.add(
    new GPUReduction({
      id: `${id}-invalid-total`,
      input: invalidFlags,
      output: invalidCount,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-sample-status`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'total', view: total, type: 'u32', access: 'read'},
        {name: 'invalid', view: invalidCount, type: 'u32', access: 'read'},
        {name: 'status', view: props.samplingStatus, type: 'u32', access: 'read_write'},
        {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'}
      ],
      body: `let required=total[totalOffset];
offsets[offsetsOffset+${segmentCount}u]=required;
status[statusOffset]=min(required,${sampleCapacity}u);
status[statusOffset+1u]=required;
status[statusOffset+2u]=u32(required>${sampleCapacity}u);
status[statusOffset+3u]=invalid[invalidOffset];`
    })
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-clear-samples`,
      operation: id,
      invocationCount: sampleCapacity,
      bindings: [
        {name: 'samples', view: sampleData, type: 'u32', access: 'read_write'},
        {name: 'rows', view: sampleRows, type: 'u32', access: 'read_write'}
      ],
      body: `samples[samplesOffset+2u*index]=0x7fc00000u;
samples[samplesOffset+2u*index+1u]=0x7fc00000u;
samples[samplesOffset+${sampleCapacity * 2}u+index]=0xffffffffu;
rows[rowsOffset+index]=index;`
    })
  );
  graph.add(
    createLoadBalancedForEachNode(graph, {
      id: `${id}-write-samples`,
      operation: id,
      offsets,
      itemCapacity: sampleCapacity,
      bindings: [
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'buildingIds', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'samples', view: sampleData, type: 'u32', access: 'read_write'}
      ],
      declarations: `fn visitLoadBalancedItem(row:u32,item:u32,lane:u32){
let base=segmentsOffset+4u*row;
let start=vec2<f32>(segments[base],segments[base+1u]);
let end=vec2<f32>(segments[base+2u],segments[base+3u]);
let part=item-loadBalancedOffsets[loadBalancedOffsetsOffset+row];
let position=mix(start,end,f32(part)/f32(counts[countsOffset+row]));
samples[samplesOffset+2u*item]=bitcast<u32>(position.x);samples[samplesOffset+2u*item+1u]=bitcast<u32>(position.y);
samples[samplesOffset+${sampleCapacity * 2}u+item]=buildingIds[buildingIdsOffset+row];
}`
    })
  );
  const buildingCounts = createTransientView(
    graph,
    `${id}-building-counts`,
    'uint32',
    buildingCount
  );
  const buildingOffsets = createTransientView(
    graph,
    `${id}-building-offsets`,
    'uint32',
    buildingCount + 1
  );
  const sortedRows = createTransientView(graph, `${id}-building-samples`, 'uint32', sampleCapacity);
  const sortedOwners = createTransientView(graph, `${id}-sorted-owners`, 'uint32', sampleCapacity);
  graph.add(
    new GPUGroupAggregation({
      id: `${id}-count-buildings`,
      keys: owners,
      output: buildingCounts
    }).getCommandNodes(graph)
  );
  graph.add(
    new GPUScan({
      id: `${id}-building-scan`,
      input: buildingCounts,
      output: buildingOffsets
    }).getCommandNodes(graph)
  );
  graph.add(
    new GPUSort({
      id: `${id}-building-sort`,
      keys: owners,
      values: sampleRows,
      outputKeys: sortedOwners,
      outputValues: sortedRows
    }).getCommandNodes(graph)
  );
  const buildingIndex = createTransientView(
    graph,
    `${id}-building-index`,
    'uint32',
    4 + buildingCount + 1 + sampleCapacity
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-pack-buildings`,
      operation: id,
      invocationCount: Math.max(4, buildingCount + 1, sampleCapacity),
      bindings: [
        {name: 'counts', view: buildingCounts, type: 'u32', access: 'read'},
        {name: 'offsets', view: buildingOffsets, type: 'u32', access: 'read'},
        {name: 'rows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'output', view: buildingIndex, type: 'u32', access: 'read_write'},
        {name: 'status', view: props.samplingStatus, type: 'u32', access: 'read'}
      ],
      body: `if(index<4u){output[outputOffset+index]=status[statusOffset+index];}
if(index<${buildingCount}u){output[outputOffset+4u+index]=offsets[offsetsOffset+index];}
if(index==${buildingCount}u){output[outputOffset+4u+index]=offsets[offsetsOffset+${buildingCount - 1}u]+counts[countsOffset+${buildingCount - 1}u];}
if(index<${sampleCapacity}u){output[outputOffset+${buildingCount + 5}u+index]=rows[rowsOffset+index];}`
    })
  );
  return {samples, sampleData, owners, buildingIndex};
}
