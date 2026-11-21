// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Shared installed-package contract doubles. Keep dependency/peer probes in the calling spec. */
export function createPackedConsumerContractFixture(mode = 'unchanged') {
  const publicEntryKeys = [
    '.',
    './computation-record',
    './gpu-spatial-analysis',
    './gpu-spatial-analysis/ontology',
    './gpu-network',
    './gpu-terrain',
    './dataframe-analysis',
    './raster-analysis',
    './pointcloud-analysis',
    './geospatial-analysis',
    './analysis-core'
  ];
  const exportNames = [
    'GPUZonalStatistics',
    'GPUComputationRecorder',
    'getGraphViewChunks',
    'validateGraphViewsBelongToGraph',
    'createWGSLKernelNode',
    'getWGSLKernelNodeRecord',
    'GPUContributorOperation',
    'GPU_CONTRIBUTOR_OPERATION_LOWERING'
  ];
  // Upstream graphs, GPUVector/GPUData objects, buffers and dispatch validation are real.
  // The versioned fixture symbols emulate only analysis's cross-format protocols.
  const source = `
    const kernelRecordsKey = Symbol.for('packed-fixture/kernel-records');
    const operationBrand = Symbol.for('packed-fixture/operation');
    class GPUZonalStatistics {
      constructor(props) {this.props = props; this.sumOrder = 'atomic';}
      getCommandNodes(graph) {
        if (${mode === 'reverse-zonal' ? "moduleFormat === 'import'" : 'false'}) throw new Error('broken upstream aggregation');
        validateGraphViewsBelongToGraph('zonal', graph, [this.props.values, this.props.features.pointFeatureRows, this.props.output.sums]);
        const preparation = {id: 'fixture-zonal-prepare', type: 'compute', resources: [
          {buffer: this.props.output.sums, usage: 'storage-write'}]};
        const accumulation = getGraphViewChunks(this.props.values).map((chunk, index) => ({
          id: 'fixture-zonal-accumulate-' + index, type: 'compute', resources: [
            {buffer: chunk, usage: 'storage-read'},
            {buffer: this.props.output.sums, usage: 'storage-read-write'}]
        }));
        if (${mode === 'reverse-zonal-preparation-only' ? "moduleFormat === 'import'" : 'false'}) return [preparation];
        if (${mode === 'reverse-zonal-missing-second-chunk' ? "moduleFormat === 'import'" : 'false'}) return [preparation, accumulation[0]];
        return [preparation, ...accumulation];
      }
    }
    class GPUComputationRecorder {
      recordKernels(graph) {graph[kernelRecordsKey] = new Map(); return graph;}
    }
    function getGraphViewChunks(view) {
      if (${mode === 'foreign-vector' ? 'true' : mode === 'reverse-foreign-vector' ? "moduleFormat === 'import'" : 'false'}) return [view];
      return Array.isArray(view.data) ? view.data : [view];
    }
    function validateGraphViewsBelongToGraph(id, graph, views) {
      for (const view of views) for (const chunk of getGraphViewChunks(view)) {
        if (chunk.buffer.graph !== graph) throw new Error(id + ': wrong graph');
      }
    }
    function createWGSLKernelNode(graph, props) {
      const layout = getBoundedDispatchLayout(props.operation, props.invocationCount,
        props.workgroupSize, graph.device.limits.maxComputeWorkgroupsPerDimension);
      if (!(${mode === 'foreign-record' ? 'true' : mode === 'reverse-foreign-record' ? "moduleFormat === 'require'" : 'false'})) {
        graph[kernelRecordsKey]?.set(props.id, {...props,
        source: '@compute @workgroup_size(' + props.workgroupSize + ') fn main() {' + props.body + '}',
        workgroupSize: [props.workgroupSize,1,1], workgroups: [layout.x,layout.y,layout.z]});
      }
      return {id: props.id};
    }
    function getWGSLKernelNodeRecord(graph, id) {return graph[kernelRecordsKey]?.get(id);}
    class GPUContributorOperation {
      constructor(props) {Object.assign(this, props); this[operationBrand] = true; this.type = 'analysis-contributor';}
      destroy() {}
    }
    const GPU_CONTRIBUTOR_OPERATION_LOWERING = {lower(operation, context) {
      if (${mode === 'foreign-operation' ? 'true' : mode === 'reverse-foreign-operation' ? "moduleFormat === 'import' || !operation[operationBrand]" : mode === 'type-string-impostor' ? 'false' : '!operation[operationBrand]'}) throw new Error(operation.id + ' is not a GPUContributorOperation');
      const resolved = new Map();
      if (${mode === 'output-first' ? 'true' : 'false'} && operation[operationBrand]) {
        for (const output of Object.values(operation.outputs)) resolved.set(output.value, context.resolveVector(output.value));
      }
      context.lower(operation.createContributor((port, format) => {
        const value = operation.inputs[port] ?? operation.outputs[port]?.value;
        const vector = resolved.get(value) ?? context.resolveVector(value);
        if (${mode === 'vector-metadata' ? 'true' : 'false'} && (vector.format !== format || vector.length !== 1 || vector.data.length !== 1)) throw new Error('conforming lowering requires full graph vector metadata');
        const view = vector.data[0];
        if (${mode === 'reverse-operation-ports' ? "moduleFormat === 'import'" : 'false'} && port === 'selectedInput') return {...view};
        if (view.format !== format) throw new Error('wrong port format');
        return view;
      }));
    }};`;
  const declarationSource = `${[
    'CPUTerrainCatchmentSummaryOperationProps',
    'GPUNetworkIsochronesOutput',
    'GPUSpatialScanStatisticOutput',
    'GPURasterPatchLabelsStatus',
    'CPUPolygonRows',
    'GPUMapColoringPriority',
    'CPUCompositeDimensions',
    'VariogramModelFitDiagnostics',
    'PublishNodeProps',
    'CPURasterOperationBand',
    'RasterGridAffine'
  ]
    .filter(name => mode !== 'missing-public-type' || name !== 'GPUNetworkIsochronesOutput')
    .map(name => `export interface ${name} {fixture: string}`)
    .join('\n')}
    export interface WGSLDispatchGate<Parameters> {parameters: Parameters}`;
  return {
    publicEntryKeys,
    exportNames,
    declarationSource,
    esmSource: `import {getBoundedDispatchLayout} from '@luma.gl/graph/gpu-core';
      const moduleFormat = 'import';\n${source}`,
    commonjsSource: `const {getBoundedDispatchLayout} = require('@luma.gl/graph/gpu-core');
      const moduleFormat = 'require';\n${source}`
  };
}
