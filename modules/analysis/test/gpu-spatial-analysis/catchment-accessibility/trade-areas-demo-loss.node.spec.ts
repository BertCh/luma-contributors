// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {
  createSourceFile,
  forEachChild,
  isArrowFunction,
  ScriptTarget,
  transpileModule
} from 'typescript';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

const DEMOS = [
  'examples/deck/spatial-analysis-showcase/src/scenes/regression/trade-areas.compute.ts',
  'examples/deck/spatial-analysis-explorer/modes/trade-areas-mode.ts'
];

for (const path of DEMOS) {
  it.each(['demand', 'facility'] as const)(`${path} keeps %s loss on the Huff consumer`, role => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const producers: GPUCommandNodeProducer[] = [];
    const floats = (name: string, values: number[]) => {
      const view = createTransientView(graph, name, 'float32', values.length);
      memory.write(view, Float32Array.from(values));
      return view;
    };
    const integers = (name: string, values: number[]) => {
      const view = createTransientView(graph, name, 'uint32', values.length);
      memory.write(view, Uint32Array.from(values));
      return view;
    };
    const weights = (name: string, forward: boolean) => ({
      rows: forward ? 1 : 2,
      slots: 2,
      offsets: integers(
        `${name}-offsets`,
        forward
          ? role === 'demand'
            ? [0, 1]
            : [0, 2]
          : role === 'facility'
            ? [0, 1, 1]
            : [0, 1, 2]
      ),
      neighbors: integers(`${name}-neighbors`, forward ? [0, 1] : [0, 0]),
      weights: floats(`${name}-weights`, [1, 1]),
      overflow: integers(`${name}-overflow`, [Number(role === (forward ? 'demand' : 'facility'))])
    });
    try {
      const source = readFileSync(path, 'utf8');
      const start = source.indexOf('const weightsOf =');
      const end = source.indexOf('// Spokes from', start);
      expect(start).toBeGreaterThan(0);
      expect(end).toBeGreaterThan(start);
      // Execute the production wiring, with real contributor constructors and small resident
      // buffers. Exclude data loading, UI setup and the drawing passes; no source expressions
      // inside the graph-construction block are changed or replaced.
      const context = {
        graph: {add: (producer: GPUCommandNodeProducer) => producers.push(producer)},
        view: (_name: string, view: GraphDataView) => view,
        projection: {spatialContext: undefined},
        trips: {frame: {spatialContext: undefined}},
        facilityCount: 2,
        demandCount: 1,
        slotCapacity: 2,
        facilityPositionsView: createTransientView(graph, 'facility-positions', 'float32x2', 2),
        demandPositionsView: createTransientView(graph, 'demand-positions', 'float32x2', 1),
        facilityMaskView: integers('facility-mask', [1, 1]),
        supplyView: floats('supply', [1, 1]),
        capacityView: floats('capacity', [1, 1]),
        demandView: floats('demand', [1]),
        demandRows: weights('demand', true),
        facilityRows: weights('facility', false),
        catchmentDemandRows: weights('catchment-demand', true),
        catchmentFacilityRows: weights('catchment-facility', false),
        huffDemandRows: weights('huff-demand', true),
        huffFacilityRows: weights('huff-facility', false),
        searchParameters: {importToGraph: () => floats('search-parameters', new Array(12).fill(0))},
        catchmentParameters: {
          importToGraph: () => floats('catchment-parameters', new Array(12).fill(0))
        },
        huffParameters: {
          importToGraph: () => floats('huff-search-parameters', new Array(12).fill(0))
        },
        huffExponent: {importToGraph: () => floats('huff-exponent', [1])},
        huffExponentParameter: {importToGraph: () => floats('huff-exponent-parameter', [1])},
        reachable: integers('reachable', [99]),
        access2: floats('access2', [99]),
        access3: floats('access3', [99]),
        accessibility2: floats('accessibility2', [99]),
        accessibility3: floats('accessibility3', [99]),
        ratios2: floats('ratios2', [99, 99]),
        ratios3: floats('ratios3', [99, 99]),
        tradeArea: integers('trade-area', [99]),
        tradeAreaProbability: floats('probability', [99]),
        expectedDemand: floats('expected-demand', [99, 99]),
        GPUNeighborSearch,
        GPUCatchmentAccessibility,
        GPUHuffTradeAreas
      };
      const code = transpileModule(source.slice(start, end), {
        compilerOptions: {target: ScriptTarget.ESNext}
      }).outputText;
      new Function(...Object.keys(context), code)(...Object.values(context));
      const huff = producers.find(
        producer => producer instanceof GPUHuffTradeAreas
      ) as GPUHuffTradeAreas;
      expect(huff).toBeDefined();
      executeCommandNodesOnCPU(huff.getCommandNodes(graph), memory);
      if (role === 'demand') {
        expect(Array.from(memory.readUint32(huff.props.output.tradeArea))).toEqual([0xffffffff]);
        expect(memory.readFloat32(huff.props.output.tradeAreaProbability!)[0]).toBeNaN();
      }
      expect(
        Array.from(memory.readFloat32(huff.props.output.expectedDemand!)).every(Number.isNaN)
      ).toBe(true);
      const searches = producers.filter(
        (producer): producer is GPUNeighborSearch => producer instanceof GPUNeighborSearch
      );
      for (const consumer of producers.filter(
        (producer): producer is GPUHuffTradeAreas | GPUCatchmentAccessibility =>
          producer instanceof GPUHuffTradeAreas || producer instanceof GPUCatchmentAccessibility
      )) {
        for (const port of [consumer.props.demandWeights, consumer.props.facilityWeights]) {
          expect(searches.some(search => search.outputWeights === port)).toBe(true);
        }
      }
    } finally {
      device.destroy();
    }
  });
}

it.each(DEMOS)('%s reports unavailable summaries for incomplete weights', path => {
  const source = createSourceFile(path, readFileSync(path, 'utf8'), ScriptTarget.ESNext, true);
  let callback = '';
  const visit = (node: import('typescript').Node): void => {
    if (
      isArrowFunction(node) &&
      ((node.parameters[0]?.name.getText(source) === 'get' &&
        node.getText(source).includes("get('access2')")) ||
        (node.parameters[0]?.name.getText(source) === 'bytes' &&
          node.getText(source).includes('const overflows =')))
    )
      callback = node.getText(source);
    forEachChild(node, visit);
  };
  visit(source);
  expect(callback).not.toBe('');
  const readouts = new Map<string, string>();
  const readout = (name: string) => ({setValue: (value: string) => readouts.set(name, value)});
  const context = {
    ctx: {
      options: {demand: 'population', facility: 'grocery', map: 'access2sfca'},
      setReadout: (name: string, value: string) => readouts.set(name, value),
      setLegendExtent: () => {},
      requestLayers: () => {}
    },
    context: {updateLayers: () => {}},
    measures: {population: Float32Array.of(1)},
    demandValues: Float32Array.of(1),
    poverty: Float32Array.of(0),
    demandCount: 1,
    facilityCount: 2,
    totalDemand: 1,
    selectedSupplyTotal: 2,
    facilitySupply: 2,
    kindTotals: [2],
    FACILITY_KINDS: ['grocery'],
    adjacency: [],
    getQuantile: () => 1,
    getPositiveQuantile: () => 1,
    formatInteger: String,
    formatNumber: String,
    formatCount: String,
    formatDecimal: String,
    tradeAreaColors: {write: () => {}},
    assignTradeColors: () => Uint32Array.of(0xffffffff),
    demandColor: '2sfca',
    accessibilityRanges: new Map(),
    overflowReadout: readout('overflow'),
    unreachedReadout: readout('unreached'),
    conservationReadout: readout('conservation'),
    capturedReadout: readout('captured')
  };
  const code = transpileModule(
    `let latest, expectedRange, ratioRange, accessRange, accessibilityRange;
return (${callback});`,
    {compilerOptions: {target: ScriptTarget.ESNext}}
  ).outputText;
  const run = new Function(...Object.keys(context), code)(...Object.values(context));
  if (path.includes('showcase')) {
    run((name: string) => ({
      f32:
        name === 'expected' || name === 'ratios' ? Float32Array.of(NaN, NaN) : Float32Array.of(NaN),
      u32: Uint32Array.of(name === 'overflows' ? 1 : name === 'facilityOverflow' ? 0 : 0xffffffff)
    }));
  } else {
    const bytes = new ArrayBuffer(36);
    new Uint32Array(bytes, 0, 4).set([1, 0, 0, 0]);
    new Float32Array(bytes, 16, 4).fill(NaN);
    new Uint32Array(bytes, 32, 1)[0] = 0xffffffff;
    run(bytes);
  }
  expect(readouts.get('captured')).toBe('n/a: incomplete weights');
  expect(readouts.get('conservation')).toBe('n/a: incomplete weights');
  expect(readouts.get('unreached')).toBe('n/a: incomplete weights');
});
