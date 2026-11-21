// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {SCAN_MEMBERSHIP_BODY} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/time/space-time-clusters-membership';
import {getNeighborhoodSelectionBody} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/neighborhood-display';

const read = (path: string) => readFileSync(`examples/deck/${path}`, 'utf8');
const compile = (source: string) =>
  transpileModule(source, {compilerOptions: {target: ScriptTarget.ES2022}}).outputText;
type Kernel = {
  body: string;
  declarations?: string;
  invocationCount: number;
  bindings: {name: string; type: string; access: string; view: Float32Array | Uint32Array}[];
};
function execute(kernel: Kernel) {
  const declarations = kernel.bindings
    .map(
      (binding, index) =>
        `const ${binding.name}Offset: u32 = 0u; @group(0) @binding(${index}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const shader = `${declarations}\n${kernel.declarations ?? ''}
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
      let index = invocation.x; ${kernel.body}
    }`;
  new WgslExec(new WgslParser().parse(shader)).dispatchWorkgroups(
    'main',
    [kernel.invocationCount, 1, 1],
    {0: Object.fromEntries(kernel.bindings.map((binding, index) => [index, binding.view]))}
  );
}

it('B2 explorer selection preserves unavailable displayed values and recovers', () => {
  const source = read('spatial-analysis-explorer/modes/spatial-weights-mode.ts');
  const start = source.lastIndexOf(
    '      addKernelPass(neighborhoodGraph,',
    source.indexOf("id: 'neighborhood-select'")
  );
  const end = source.indexOf('      neighborhoodGraph.add(', start);
  const table = Float32Array.of(NaN, NaN),
    selected = new Float32Array(2);
  const capture = new Function(
    'table',
    'selected',
    'getNeighborhoodSelectionBody',
    `
    const cellCount = 2, neighborhoodGraph = {}, modes = new Uint32Array(2), entropy = new Float32Array(2);
    const NEIGHBORHOOD_COLUMNS = ['mean'], GPU_NEIGHBORHOOD_SUMMARY_NO_MODE = 0xffffffff;
    const neighborhoodSourceBuffer = Uint32Array.of(0);
    const importGraphBuffer = (graph, name, buffer) => buffer;
    let kernel; const addKernelPass = (graph, props) => {kernel = props;};
    ${compile(source.slice(start, end))} return kernel;
  `
  );
  execute(capture(table, selected, getNeighborhoodSelectionBody));
  expect(Array.from(selected)).toEqual([NaN, NaN]);
  table.set([2, 7]);
  execute(capture(table, selected, getNeighborhoodSelectionBody));
  expect(Array.from(selected)).toEqual([2, 7]);
});

it('B4 scan membership distinguishes invalid, valid empty, and recovered frames', () => {
  const source = read('spatial-analysis-showcase/src/scenes/time/space-time-clusters.compute.ts');
  const start = source.indexOf(
    '    addKernelPass(graph,',
    source.indexOf('// Zones inside a significant')
  );
  const end = source.indexOf('    const compiled =', start);
  const invalid = Uint32Array.of(1),
    indices = new Uint32Array(4),
    statistics = new Float32Array(8);
  const membership = new Uint32Array(3);
  const capture = new Function(
    'invalidCount',
    'clusterIndicesView',
    'clusterStatisticsView',
    'membership',
    'SCAN_MEMBERSHIP_BODY',
    `
    const graph = {}, key = 'fixture', zoneCount = 2, MAXIMUM_CLUSTERS = 1;
    const zonePositions = Float32Array.of(0,0,1,0);
    const significanceParameter = {importToGraph: () => Float32Array.of(.05)};
    const v = (name, buffer) => buffer;
    let kernel; const addKernelPass = (graph, props) => {kernel = props;};
    ${compile(source.slice(start, end))} return kernel;
  `
  );
  const run = () =>
    execute(capture(invalid, indices, statistics, membership, SCAN_MEMBERSHIP_BODY));
  run();
  expect(Array.from(membership)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  invalid[0] = 0;
  run();
  expect(Array.from(membership)).toEqual([0, 0, 0xffffffff]);
  indices.set([0, 2, 0, 0]);
  statistics[3] = Math.fround(0.05);
  statistics[4] = 2;
  run();
  expect(Array.from(membership)).toEqual([1, 1, 0xffffffff]);
});

it('B1 scene analysis binds the CSR writer status rather than the original search status', async () => {
  const source = read('spatial-analysis-showcase/src/scenes/weights/spatial-weights.compute.ts');
  const start = source.indexOf('    const analysisSlots =');
  const end = source.indexOf('    const populationSorted =', start);
  const core = {
    positions: {},
    overflow: Uint32Array.of(0),
    total: Uint32Array.of(2),
    candidateOverflow: Uint32Array.of(0)
  };
  const build = new Function(
    'core',
    `return (async () => {
    const ANALYSIS_SLOTS = {fixture: 2}, geography = {id: 'fixture', analysisFrame: {spatialContext: {}}, groups: new Uint32Array(2), getVariable: () => [1,2]};
    const ctx = {options: {}}, device = {}, resources = {}, rowCount = 2;
    const create = (name, data) => typeof data === 'number' ? new Uint32Array(data/4) : data;
    const createAnalysisSpace = async props => props;
    ${compile(source.slice(start, end))}
    return space.producer;
  })();`
  );
  const producer = await build(core);
  // The independent algebra route must be able to clear search loss, and final algebra loss
  // must be visible even while the original search is complete.
  expect(producer.overflow).not.toBe(core.overflow);
  expect(producer.candidateOverflow).not.toBe(core.candidateOverflow);
  const encodeStart = source.indexOf('          target.core.encode(commandEncoder,');
  const encodeEnd = source.indexOf('          const display =', encodeStart);
  const target = {
    core: {...core, csr: {}, slots: 2, encode() {}},
    geography: {count: 2},
    space: {csr: {}},
    partner: {encode() {}},
    algebraOverflow: producer.overflow,
    algebraTotal: producer.total,
    algebraCandidateOverflow: producer.candidateOverflow,
    algebra: new Map<string, {encode: () => void}>()
  };
  target.algebra.set('higherOrder', {
    encode() {
      producer.overflow[0] = 1;
      producer.candidateOverflow[0] = 0;
    }
  });
  target.algebra.set('block', {
    encode() {
      producer.overflow[0] = 0;
      producer.candidateOverflow[0] = 0;
    }
  });
  const encode = new Function(
    'target',
    'route',
    `
    const commandEncoder = {copyBufferToBuffer: ({sourceBuffer, destinationBuffer}) => destinationBuffer.set(sourceBuffer)};
    const getConfig = () => ({}), getAlgebraKey = () => route, isBinaryCombine = () => false;
    const copyCsr = () => {}, o = {};
    ${compile(source.slice(encodeStart, encodeEnd))}
  `
  );
  encode(target, 'higherOrder');
  expect(producer.overflow[0]).toBe(1); // Complete search, truncated algebra result.
  core.overflow[0] = 1;
  encode(target, 'block');
  expect(producer.overflow[0]).toBe(0); // Independent fitting block does not inherit search loss.
  core.overflow[0] = 0;
  core.candidateOverflow[0] = 1;
  encode(target, null);
  expect(producer.candidateOverflow[0]).toBe(1); // Direct route copies the original producer flags.
  core.candidateOverflow[0] = 0;
  encode(target, null);
  expect(producer.candidateOverflow[0]).toBe(0);
});

it('B1 algebra receives operand loss and exposes intermediate loss to analysis', async () => {
  const source = read('spatial-analysis-showcase/src/scenes/weights/spatial-weights.compute.ts');
  const start = source.indexOf('  const prepareAlgebra =');
  const end = source.indexOf('  const isBinaryCombine =', start);
  const capture = new Function(
    'combine',
    `return (async () => {
    const ctx = {options: {combine, order: 3, cumulative: true}}, device = {};
    const getAlgebraKey = () => combine;
    let captured;
    const GPUSpatialWeightsAlgebra = class {constructor(props) {captured = props;}};
    const GPUCommandGraph = class {add() {} async compileAsync() {return {};}};
    const importGraphBuffer = (graph, name, buffer) => buffer;
    const csr = {offsets: {}, neighbors: {}, weights: {}};
    const target = {geography: {count: 3, groupCount: 2, analysisFrame: {spatialContext: {}}},
      algebra: new Map(), resources: {track: value => value}, space: {csr, slots: 4},
      core: {csr, slots: 6, overflow: Uint32Array.of(0), candidateOverflow: Uint32Array.of(1)},
      partner: {csr, slots: 6, overflow: Uint32Array.of(1)},
      algebraOverflow: Uint32Array.of(0), algebraCandidateOverflow: Uint32Array.of(0), algebraTotal: Uint32Array.of(0), groups: {}};
    ${compile(source.slice(start, end))}
    await prepareAlgebra(target); return {captured, target};
  })();`
  );
  const higher = await capture('higherOrder');
  expect(higher.captured.weights.status.candidateOverflow).toBe(
    higher.target.core.candidateOverflow
  );
  expect(higher.captured.output.status.candidateOverflow).toBe(
    higher.target.algebraCandidateOverflow
  );
  const union = await capture('union');
  expect(union.captured.left.status.overflow).toBe(union.target.core.overflow);
  expect(union.captured.right.status.overflow).toBe(union.target.partner.overflow);
  const block = await capture('block');
  expect(block.captured.weights).toBeUndefined();
  expect(block.captured.left).toBeUndefined();
  expect(block.captured.output.status.candidateOverflow).toBe(
    block.target.algebraCandidateOverflow
  );
});
