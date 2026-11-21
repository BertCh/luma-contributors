// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, test} from 'vitest';
import {getRegressionResidualForDisplay} from '../../../examples/deck/spatial-analysis-shared/regression-residual-display';

const showcasePath =
  'examples/deck/spatial-analysis-showcase/src/scenes/regression/health-regression.compute.ts';
const explorerPath = 'examples/deck/spatial-analysis-explorer/modes/regression-mode.ts';

// Execute the actual construction expressions, retaining imported resource identities. A flag
// written directly to a consumer would miss the producer/consumer boundary this test checks.
test('showcase weight graphs export the same distinct certificates to regression consumers', async () => {
  const source = readFileSync(showcasePath, 'utf8');
  for (const graph of ['recipeGraph', 'errorGraph', 'lagGraph'])
    expect(source).toContain(`weights: importWeights(${graph})`);
  const start = source.indexOf('  const importWeights =');
  const end = source.indexOf('\n  searchParameters.write(', start);
  const code = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const construct = new Function(
    'environment',
    `with(environment) { ${code}; return {importWeights, getWeightsGraph}; }`
  );
  const identity = (name: string) => ({name, word: 0});
  const environment = {
    projection: {spatialContext: {}},
    device: {},
    rowCount: 3,
    capacity: 20,
    offsetsBuffer: identity('offsets'),
    neighborsBuffer: identity('neighbors'),
    weightValuesBuffer: identity('values'),
    overflowBuffer: identity('final'),
    candidateOverflowBuffer: identity('candidate'),
    totalNeighborsBuffer: identity('required'),
    geometry: {vertices: [], featureCount: 3, partRingOffsets: [], ringOffsets: []},
    verticesBuffer: {},
    featureOffsetsBuffer: {},
    polygonOffsetsBuffer: {},
    ringOffsetsBuffer: {},
    positionsBuffer: {},
    importGraphBuffer: (_graph: unknown, _name: string, buffer: unknown) => buffer,
    resources: {track: (value: unknown) => value},
    searchParameters: {importToGraph: () => ({})},
    NEIGHBOR_COUNTS: {knn4: 4},
    GRID_SIZE: [2, 2],
    GPUCommandGraph: class {
      nodes: unknown[] = [];
      add(node: unknown) {
        this.nodes.push(node);
      }
      async compileAsync() {
        return this;
      }
    },
    GPUContiguityWeights: class {
      constructor(public props: unknown) {}
    },
    GPUSpatialWeightsTransform: class {
      constructor(public props: unknown) {}
    },
    GPUNeighborSearch: class {
      constructor(public props: unknown) {}
    }
  };
  const construction = construct(environment);
  const consumer = construction.importWeights({});
  const queen = await construction.getWeightsGraph('queen');
  const producer = queen.nodes[0].props.output;
  expect(consumer.status).toBeDefined();
  for (const field of ['overflow', 'candidateOverflow', 'requiredCount']) {
    expect(consumer.status[field]).toBe(producer.status[field]);
  }
  expect(new Set(Object.values(consumer.status)).size).toBe(3);
  // Candidate-only loss comes from the producer destination, then a source switch resets it.
  producer.status.candidateOverflow.word = 1;
  expect(consumer.status.candidateOverflow.word).toBe(1);
  expect(consumer.status.overflow.word).toBe(0);
  const knn = await construction.getWeightsGraph('knn4');
  expect(knn.nodes[0].props.output.status.candidateOverflow).toBe(
    producer.status.candidateOverflow
  );
  knn.nodes[0].props.output.status.candidateOverflow.word = 0;
  expect(consumer.status.candidateOverflow.word).toBe(0);
});

test('explorer search exports its actual loss and count destinations to regression consumers', () => {
  const source = readFileSync(explorerPath, 'utf8');
  expect(source.match(/weights: stage\.weights/g)).toHaveLength(3);
  const start = source.indexOf(
    "  const positions = resources.createBuffer('positions', study.positions);",
    source.indexOf('function createSpatialStage(')
  );
  const end = source.indexOf('\n  const writeSearch =', start);
  const code = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const construct = new Function('environment', `with(environment) { ${code}; return weights; }`);
  const nodes: {props: {output: {status: Record<string, {word: number}>}}}[] = [];
  const consumer = construct({
    resources: {
      createBuffer: (name: string) => ({name, word: 0}),
      createParameterBuffer: () => ({importToGraph: () => ({})})
    },
    study: {
      positions: [],
      predictors: [],
      response: [],
      rowOfCell: [],
      frame: {spatialContext: {}}
    },
    rowCount: 3,
    capacity: 20,
    kind: 'knn',
    neighborCount: 2,
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH: 8,
    graph: {add: (node: (typeof nodes)[number]) => nodes.push(node)},
    importGraphBuffer: (_graph: unknown, _name: string, buffer: unknown) => buffer,
    GPUNeighborSearch: class {
      constructor(public props: unknown) {}
    }
  });
  expect(consumer.status).toBeDefined();
  for (const field of ['overflow', 'candidateOverflow', 'requiredCount'])
    expect(consumer.status[field]).toBe(nodes[0].props.output.status[field]);
  nodes[0].props.output.status['candidateOverflow'].word = 1;
  expect(consumer.status.candidateOverflow.word).toBe(1);
  nodes[0].props.output.status['candidateOverflow'].word = 0;
  expect(consumer.status.candidateOverflow.word).toBe(0);
});

test('health regression discards old outcome callbacks and failed current fits cannot revive residuals', () => {
  const source = readFileSync(showcasePath, 'utf8');
  const start = source.indexOf('  const onResult =');
  const end = source.indexOf("    ctx.setReadout('rows',", start);
  const code = transpileModule(`${source.slice(start, end)}\n};`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const invalidationStart = source.indexOf('  const invalidateModelResult =');
  const invalidationEnd = source.indexOf('\n  };', invalidationStart) + 5;
  expect(invalidationStart).toBeGreaterThan(0);
  const invalidation = transpileModule(source.slice(invalidationStart, invalidationEnd), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const optionStart = source.indexOf('    setOption(id)');
  const optionEnd = source.indexOf('\n    encode(', optionStart);
  const tooltipStart = source.indexOf('    getTooltip(event)');
  const tooltipEnd = source.indexOf('\n    onClick(', tooltipStart);
  const tooltip = source.slice(tooltipStart, tooltipEnd);
  const createState = (options: string) =>
    new Function(
      'getRegressionResidualForDisplay',
      `
    let modelGeneration=0, latest=null, olsSigma=1, validationGeneration=0, latestValidation=null, modelDirty=false;
    const modelGenerationBuffer={write() {}}, validationGenerationBuffer={write() {}}, responseBuffer={write() {}},
      ctx={options:{model:'base',weights:'queen',instruments:'1',outcome:'diabetes',map:'residual'}, setLegendExtent() {},requestLayers() {}},
      activeVariantKey='base', activeWeights='queen', activeOrder='1', getVariantKey=()=> 'base',
      variants=new Map([['base', {}]]), GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED=0,
      getResponse=()=>Float32Array.of(10),updateResponseRange=()=>{},writeRidge=()=>{},writeSignificance=()=>{},
      graphSwap={request(){}},compileSelection=()=>{},commitSelection=()=>{},locate=()=>0,features=[0],geoids=['001'],income=[100],rawPredictors={poverty:[12]},
      OUTCOME_LABELS={diabetes:'Diabetes',obesity:'Obesity'},formatInteger=String;
    ${code}\n${invalidation}
    ${transpileModule(`const scene={${options}\n${tooltip}};`, {compilerOptions: {target: ScriptTarget.ES2022}}).outputText}
    return {onResult, scene, changeOutcome(){ctx.options.outcome='obesity';scene.setOption('outcome');}};
  `
    )(getRegressionResidualForDisplay);
  const check = (options: string) => {
    const state = createState(options);
    const callback = (generation: number, status: number, residual: number) =>
      state.onResult('base', 1, 'base', (name: string) => ({
        u32: Uint32Array.of(name === 'generation' ? generation : status),
        f32: Float32Array.of(name === 'olsResiduals' ? residual : 2)
      }));
    callback(0, 0, 7);
    const tooltipEvent = {coordinate: [0, 0]};
    expect(state.scene.getTooltip(tooltipEvent)).toContain('Residual: +7.0 points');
    state.changeOutcome();
    expect(state.scene.getTooltip(tooltipEvent)).toBeNull();
    callback(0, 0, 7); // Old successful callback arrives after the outcome changed.
    expect(state.scene.getTooltip(tooltipEvent)).toBeNull();
    callback(1, 1, 7); // A failed current fit can retain old GPU bytes; status must suppress them.
    expect(state.scene.getTooltip(tooltipEvent)).not.toMatch(/Residual:|Fitted \(OLS\)/);
    callback(0, 0, 7);
    expect(state.scene.getTooltip(tooltipEvent)).not.toMatch(/Residual:|Fitted \(OLS\)/);
  };
  const options = source.slice(optionStart, optionEnd);
  check(options);
  expect(() =>
    check(
      options.replace(
        '      invalidateModelResult();',
        "      if (id === 'ridge') invalidateModelResult();"
      )
    )
  ).toThrow();
  expect(source).toContain("{name: 'generation', buffer: buffers.generation, bytes: 4}");
});
