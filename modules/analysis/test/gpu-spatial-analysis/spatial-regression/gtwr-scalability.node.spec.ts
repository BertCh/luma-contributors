// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  getViewBindingRange,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {
  GPUGeographicallyTemporallyWeightedRegression,
  GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT,
  GPU_GTWR_MAXIMUM_ROW_COUNT,
  getGPUGTWRParameterValues,
  type GPUGeographicallyTemporallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {
  GPUMultiscaleGeographicallyTemporallyWeightedRegression,
  type GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-temporally-weighted-regression';
import {
  GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT as PUBLIC_GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT,
  GPU_GTWR_INDEXED_COMMAND_NODE_ALLOWANCE as PUBLIC_GPU_GTWR_INDEXED_COMMAND_NODE_ALLOWANCE
} from '../../../src/gpu-spatial-analysis/spatial-regression';
import {createGTWRIndexedSummaryNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gtwr-indexed-summary';
import {
  createNullWebGPUDevice,
  createOutputBuffer,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  auditWGSLModule,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';

// W3C WGSL §16.2 (2026-11-06), not wgsl_reflect's identifier table. The shared canonical
// list was checked against https://www.w3.org/TR/WGSL/#reserved-words.
const BANNED_WGSL_IDENTIFIERS = WGSL_RESERVED_WORDS;
function expectAllowedWGSLIdentifiers(source: string, id: string): void {
  const identifiers =
    source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
  expect(
    identifiers.filter(identifier => BANNED_WGSL_IDENTIFIERS.has(identifier)),
    id
  ).toEqual([]);
}

it('uses the complete W3C reserved-word set for generated identifier checks', () => {
  expect(WGSL_RESERVED_WORDS.size).toBeGreaterThan(140);
  for (const name of ['target', 'active', 'filter', 'module', 'meta'])
    expect(BANNED_WGSL_IDENTIFIERS.has(name)).toBe(true);
  expect(BANNED_WGSL_IDENTIFIERS.has('sample')).toBe(false);
});

function createScene(rows = 36, predictorCount = 1) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `gtwr-scale-${serial++}`, format, length);
  const props: GPUGeographicallyTemporallyWeightedRegressionProps = {
    id: 'gtwr-scale',
    positions: view('float32x2', rows),
    times: view('float32', rows),
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    },
    predictors: view('float32', rows * predictorCount),
    predictorCount,
    response: view('float32', rows),
    parameters: view('float32', 4),
    mask: view('uint32', rows),
    kernel: 'bisquare',
    output: {
      coefficients: view('float32', rows * (predictorCount + 1)),
      fitted: view('float32', rows),
      hatDiagonal: view('float32', rows),
      localStatus: view('uint32', rows),
      residuals: view('float32', rows),
      summary: view('float32', 12)
    }
  };
  return {device, graph, props, view};
}

it('defaults compact kernels to the grid while retaining all bounded reference modes', () => {
  const scene = createScene(1_000_000);
  try {
    expect(new GPUGeographicallyTemporallyWeightedRegression(scene.props).searchStructure).toBe(
      'grid'
    );
    expect(
      new GPUGeographicallyTemporallyWeightedRegression({...scene.props, adaptive: true})
        .searchStructure
    ).toBe('grid');
    expect(
      new GPUGeographicallyTemporallyWeightedRegression({
        ...scene.props,
        kernel: 'gaussian',
        gaussianCutoff: 3
      }).searchStructure
    ).toBe('grid');
    for (const overrides of [
      {searchStructure: 'brute-force' as const},
      {kernel: 'gaussian' as const},
      {bandwidthSearch: {parameters: scene.view('float32', 6), maximumIterations: 1}}
    ]) {
      expect(
        () => new GPUGeographicallyTemporallyWeightedRegression({...scene.props, ...overrides})
      ).toThrow(`rows must be an integer in [2, ${GPU_GTWR_MAXIMUM_ROW_COUNT}]`);
    }
    for (const overrides of [
      {kernel: 'gaussian' as const},
      {bandwidthSearch: {parameters: scene.view('float32', 6), maximumIterations: 1}}
    ]) {
      expect(
        () =>
          new GPUGeographicallyTemporallyWeightedRegression({
            ...scene.props,
            ...overrides,
            searchStructure: 'grid'
          })
      ).toThrow('searchStructure grid requires');
    }
    for (const gaussianCutoff of [0, -1, Infinity, NaN, 1e100, 1e-100]) {
      expect(
        () => new GPUGeographicallyTemporallyWeightedRegression({...scene.props, gaussianCutoff})
      ).toThrow('gtwr-scale gaussianCutoff');
    }
  } finally {
    scene.device.destroy();
  }
});

for (const adaptive of [false, true])
  it(`declares million-row maximal-predictor fits within budgets, adaptive ${adaptive}`, () => {
    const scene = createScene(1_000_000, 11);
    try {
      const contributor = new GPUGeographicallyTemporallyWeightedRegression({
        ...scene.props,
        adaptive
      });
      const nodes = contributor.getCommandNodes(scene.graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(nodes.filter(node => node.id.endsWith('-fit'))).toHaveLength(2); // fit and publish-fit
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        try {
          expect(
            new WgslReflect(kernel.source.replace(/\bblock\b/g, 'block_')).storage.length,
            kernel.id
          ).toBeLessThanOrEqual(8);
        } catch (error) {
          throw new Error(`${kernel.id}: ${String(error)}\n${kernel.source}`);
        }
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        expectAllowedWGSLIdentifiers(kernel.source, kernel.id);
      }
    } finally {
      scene.device.destroy();
    }
  });

for (const adaptive of [false, true])
  it(`keeps GTWR shader and pipeline counts constant as focal fits double, adaptive ${adaptive}`, () => {
    const counts: {shaders: number; pipelines: number}[] = [];
    for (const rows of [1_000_000, 2_000_000]) {
      const scene = createScene(rows);
      try {
        const nodes = new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          adaptive
        }).getCommandNodes(scene.graph);
        // Only GTWR-owned passes: published GPUScan recursively specializes per level. That
        // upstream limitation is reported separately; a focal fit, adaptive shell or summary
        // tile must never generate a GTWR shader or pipeline specialization.
        const ownNodes = nodes.filter(node => !node.id.includes('-grid-'));
        counts.push({
          shaders: new Set(getGPUCommandNodeKernelSources(ownNodes).map(kernel => kernel.source))
            .size,
          pipelines: getGPUCommandNodePipelineKeys(ownNodes).size
        });
        expect(nodes.filter(node => node.id === 'gtwr-scale-fit')).toHaveLength(1);
      } finally {
        scene.device.destroy();
      }
    }
    expect(counts[1]).toEqual(counts[0]);
  });

it('preflights aligned input/output ranges and every derived scratch before mutating the graph', () => {
  for (const mode of ['input', 'output', 'scratch', 'dense'] as const) {
    const scene = createScene(257);
    try {
      const props = {
        ...scene.props,
        ...(mode === 'dense' ? {searchStructure: 'brute-force' as const} : {})
      };
      const padded = (view: GraphDataView<'float32'>) => {
        const storage = scene.view('float32', view.length + 65);
        return scene.graph.createDataView(storage.buffer, {
          format: 'float32',
          length: view.length,
          byteOffset: 260
        });
      };
      if (mode === 'input' || mode === 'dense')
        props.parameters = padded(scene.view('float32', 4096));
      if (mode === 'output') {
        const storage = padded(scene.view('float32', 4096));
        props.output = {
          ...props.output,
          coefficients: scene.graph.createDataView(storage.buffer, {
            format: 'float32',
            length: 514,
            byteOffset: 4
          })
        };
      }
      const contributor = new GPUGeographicallyTemporallyWeightedRegression(props);
      Object.assign(scene.device.limits, {
        maxStorageBufferBindingSize: mode === 'scratch' ? 4096 : mode === 'output' ? 2056 : 16384
      });
      const before = Object.fromEntries(
        Object.entries(scene.graph)
          .filter(([, value]) => value instanceof Map || Array.isArray(value))
          .map(([key, value]) => [key, value instanceof Map ? value.size : value.length])
      );
      expect(() => contributor.getCommandNodes(scene.graph)).toThrow(
        mode === 'scratch'
          ? /packed input and fit records.*5140/
          : mode === 'output'
            ? /needs 2060 bytes/
            : /needs 16388 bytes/
      );
      const after = Object.fromEntries(
        Object.entries(scene.graph)
          .filter(([, value]) => value instanceof Map || Array.isArray(value))
          .map(([key, value]) => [key, value instanceof Map ? value.size : value.length])
      );
      expect(after).toEqual(before);
    } finally {
      scene.device.destroy();
    }
  }
});

it('rejects overlapping writable fields including optional outputs at construction', () => {
  const scene = createScene();
  try {
    for (const output of [
      {...scene.props.output, hatDiagonal: scene.props.output.fitted},
      {...scene.props.output, residuals: scene.props.output.fitted},
      {...scene.props.output, fitted: scene.props.response}
    ])
      expect(
        () => new GPUGeographicallyTemporallyWeightedRegression({...scene.props, output})
      ).toThrow('gtwr-scale');
    const tooMany = createScene(GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT + 1);
    try {
      expect(() => new GPUGeographicallyTemporallyWeightedRegression(tooMany.props)).toThrow(
        `${GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT}`
      );
    } finally {
      tooMany.device.destroy();
    }
  } finally {
    scene.device.destroy();
  }
});

it('executes summary publication against hand-computed residuals, variance and statuses', () => {
  const scene = createScene(4);
  try {
    recordWGSLKernelNodes(scene.graph);
    const memory = createCPUGraphMemory();
    const nodes = createGTWRIndexedSummaryNodes(scene.graph, 'hand', scene.props);
    const record = getWGSLKernelNodeRecord(scene.graph, nodes.at(-1)!.id)!;
    const bindings = record.bindings.filter(
      binding => !['partialSums', 'output0'].includes(binding.name)
    );
    for (const binding of bindings) {
      const values: Record<string, number> = {
        residualSquares: 2,
        hatTrace: 0.5,
        observationCount: 2,
        includedCount: 3
      };
      if (binding.name in values)
        memory.write(
          binding.view,
          binding.type === 'u32' ? new Uint32Array([values[binding.name]]) : [values[binding.name]]
        );
    }
    // The serial interpreter cannot synchronize barriers. Execute the unmodified publication
    // statements using independently calculated reductions; headless specs exercise both trees.
    const publication = record.body
      .slice(record.body.indexOf('let observations ='))
      .replace(/}\s*$/, '');
    const node = createWGSLKernelNode(scene.graph, {
      id: 'publish-probe',
      operation: 'GTWRSummaryProbe',
      invocationCount: 1,
      bindings,
      declarations: GET_NAN_WGSL,
      body: `let sum0 = 2.0;\n${publication}`
    });
    executeCommandNodesOnCPU([node], memory);
    const summary = memory.readFloat32(scene.props.output.summary);
    expect(summary[0]).toBe(2); // (-1)^2 + 1^2
    expect(summary[1]).toBe(0.5);
    expect(summary[2]).toBeNaN(); // n - trace - 2 < 0
    expect(summary[3]).toBe(0); // RSS = TSS = 2
    expect(summary[4]).toBe(2);
    expect(summary[6]).toBe(1);
    expect(summary[7]).toBeCloseTo(4 / 3, 6);
    expect(summary[10]).toBeCloseTo(-Math.log(2 * Math.PI) - 1, 6);
    expect(summary[11]).toBe(3);
  } finally {
    scene.device.destroy();
  }
});

for (const adaptive of [false, true])
  for (const timeScale of [0, 0.75]) {
    it(`executes production 3D cell selection and weighted moments against an exhaustive CPU oracle, timeScale ${timeScale}, adaptive ${adaptive}`, () => {
      const scene = createScene(125);
      try {
        recordWGSLKernelNodes(scene.graph);
        const nodes = new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          adaptive
        }).getCommandNodes(scene.graph);
        const record = (suffix: string) =>
          getWGSLKernelNodeRecord(
            scene.graph,
            nodes.find(node => node.id === `gtwr-scale-${suffix}`)!.id
          )!;
        const memory = createCPUGraphMemory();
        const positions = new Float32Array(250);
        const times = new Float32Array(125);
        const predictors = new Float32Array(125);
        const response = new Float32Array(125);
        const mask = new Uint32Array(125).fill(1);
        for (let row = 0; row < 125; row++) {
          positions[row * 2] = row % 5;
          positions[row * 2 + 1] = Math.floor(row / 5) % 5;
          times[row] = Math.floor(row / 25);
          predictors[row] = row % 7;
          response[row] = 2 + 0.25 * (row % 13);
        }
        mask[7] = 0;
        response[13] = NaN;
        memory.write(scene.props.positions, positions);
        memory.write(scene.props.times, times);
        memory.write(scene.props.predictors, predictors);
        memory.write(scene.props.response, response);
        memory.write(scene.props.mask!, mask);
        memory.write(scene.props.parameters, [adaptive ? 8 : 2.25, timeScale, 0, 0]);
        executeCommandNodesOnCPU([nodes.find(node => node.id.endsWith('-pack-records'))!], memory);
        const keyRecord = record('cell-keys');
        const boundView = (name: string) =>
          keyRecord.bindings.find(binding => binding.name === name)!.view;
        memory.write(boundView('spatialBounds'), [0, 0, 4, 4]);
        memory.write(boundView('temporalBounds'), [0, 0, 4, 4]);
        executeCommandNodesOnCPU([nodes.find(node => node.id === 'gtwr-scale-cell-keys')!], memory);
        const keys = memory.readUint32(boundView('cellKeys'));
        const fitRecord = record('fit');
        const fitView = (name: string) =>
          fitRecord.bindings.find(binding => binding.name === name)!.view;
        const cellCount = fitView('cellOffsets').length - 1;
        const ordered = Array.from({length: 125}, (_, row) => row)
          .filter(row => keys[row] < cellCount)
          .sort((first, second) => keys[first] - keys[second] || first - second);
        const offsets = new Uint32Array(cellCount + 1);
        for (const row of ordered) offsets[keys[row] + 1]++;
        for (let cell = 0; cell < cellCount; cell++) offsets[cell + 1] += offsets[cell];
        memory.write(fitView('cellOffsets'), offsets);
        memory.write(fitView('sortedRows'), new Uint32Array(ordered));
        // Probe the actual candidate loop and moment accumulation, stopping before the pointer
        // Cholesky solve that the serial WGSL interpreter cannot execute. No candidate code changes.
        const body = `${fitRecord.body.slice(0, fitRecord.body.indexOf('    var scale:'))}
        fits[fitsOffset + index * RECORD_WORDS] = normalMatrix[0];
        fits[fitsOffset + index * RECORD_WORDS + 1u] = normalResponse[0];
        return;
      }
      fits[fitsOffset + index * RECORD_WORDS] = getNaN();
      fits[fitsOffset + index * RECORD_WORDS + 1u] = getNaN();`;
        const probe = createWGSLKernelNode(scene.graph, {
          id: 'moment-probe',
          operation: 'GTWRMomentProbe',
          invocationCount: 125,
          bindings: fitRecord.bindings,
          declarations: fitRecord.declarations,
          body
        });
        executeCommandNodesOnCPU([probe], memory);
        const moments = memory.readFloat32(fitView('fits'));
        for (let focal = 0; focal < 125; focal++) {
          if (!mask[focal] || !Number.isFinite(response[focal])) {
            expect(moments[focal * 5]).toBeNaN();
            continue;
          }
          const distances = Array.from({length: 125}, (_, row) => row)
            .filter(row => mask[row] && Number.isFinite(response[row]))
            .map(row =>
              Math.sqrt(
                (positions[row * 2] - positions[focal * 2]) ** 2 +
                  (positions[row * 2 + 1] - positions[focal * 2 + 1]) ** 2 +
                  timeScale * (times[row] - times[focal]) ** 2
              )
            )
            .sort((first, second) => first - second);
          const bandwidth = adaptive
            ? Math.fround(Math.fround(distances[7]) * Math.fround(1.0000001))
            : 2.25;
          let expectedWeight = 0;
          let expectedResponse = 0;
          for (let row = 0; row < 125; row++) {
            if (!mask[row] || !Number.isFinite(response[row])) continue;
            const squared =
              (positions[row * 2] - positions[focal * 2]) ** 2 +
              (positions[row * 2 + 1] - positions[focal * 2 + 1]) ** 2 +
              timeScale * (times[row] - times[focal]) ** 2;
            const weight = squared >= bandwidth ** 2 ? 0 : (1 - squared / bandwidth ** 2) ** 2;
            expectedWeight += weight;
            expectedResponse += weight * response[row];
          }
          expect(moments[focal * 5]).toBeCloseTo(expectedWeight, 4);
          expect(moments[focal * 5 + 1]).toBeCloseTo(expectedResponse, 4);
        }
      } finally {
        scene.device.destroy();
      }
    });
  }

it('selects exact adaptive bandwidths across many empty shells, degenerate axes and invalid neighbor counts', () => {
  const scene = createScene(129);
  try {
    recordWGSLKernelNodes(scene.graph);
    const nodes = new GPUGeographicallyTemporallyWeightedRegression({
      ...scene.props,
      adaptive: true
    }).getCommandNodes(scene.graph);
    const fitRecord = getWGSLKernelNodeRecord(scene.graph, 'gtwr-scale-fit')!;
    const memory = createCPUGraphMemory();
    const positions = new Float32Array(258);
    for (let row = 0; row < 129; row++)
      positions[row * 2] = row < 120 ? row * 0.129 : 10_000 + (row - 120);
    memory.write(scene.props.positions, positions);
    memory.write(scene.props.times, new Float32Array(129));
    memory.write(scene.props.predictors, new Float32Array(129));
    memory.write(scene.props.response, new Float32Array(129));
    memory.write(scene.props.mask!, new Uint32Array(129).fill(1));
    executeCommandNodesOnCPU([nodes.find(node => node.id.endsWith('-pack-records'))!], memory);
    const bound = (name: string) => fitRecord.bindings.find(binding => binding.name === name)!.view;
    memory.write(bound('spatialBounds'), [0, 0, 10008, 0]);
    memory.write(bound('temporalBounds'), [0, 0, 0, 0]);
    const cellOffsets = scene.view('uint32', 32 ** 3 + 1);
    const keys = Array.from({length: 129}, (_, row) =>
      Math.min(31, Math.floor((positions[row * 2] / 10008) * 32))
    );
    const ordered = Array.from({length: 129}, (_, row) => row).sort(
      (first, second) => keys[first] - keys[second] || first - second
    );
    const offsets = new Uint32Array(32 ** 3 + 1);
    for (const key of keys) offsets[key + 1]++;
    for (let cell = 0; cell < 32 ** 3; cell++) offsets[cell + 1] += offsets[cell];
    memory.write(cellOffsets, offsets);
    memory.write(bound('sortedRows'), new Uint32Array(ordered));
    const probe = createWGSLKernelNode(scene.graph, {
      id: 'adaptive-shell-probe',
      operation: 'GTWRAdaptiveProbe',
      invocationCount: 129,
      bindings: fitRecord.bindings.map(binding =>
        binding.name === 'cellOffsets' ? {...binding, view: cellOffsets} : binding
      ),
      declarations: fitRecord.declarations!.replace(
        /const GRID_DIMENSION: u32 = \d+u;/,
        'const GRID_DIMENSION: u32 = 32u;'
      ),
      body: 'fits[fitsOffset + index * RECORD_WORDS] = getAdaptiveBandwidth(index, parameters[parametersOffset], parameters[parametersOffset + 1u]);'
    });
    for (const timeScale of [0, 1])
      for (const requested of [1, 8, 128, 129, 130, 2.5]) {
        memory.write(scene.props.parameters, [requested, timeScale, 0, 0]);
        executeCommandNodesOnCPU([probe], memory);
        const bandwidths = memory.readFloat32(bound('fits'));
        for (let focal = 0; focal < 129; focal++) {
          if (requested > 128 || !Number.isInteger(requested)) {
            expect(bandwidths[focal * 5]).toBeNaN();
            continue;
          }
          const distances = Array.from({length: 129}, (_, row) =>
            Math.abs(positions[row * 2] - positions[focal * 2])
          ).sort((first, second) => first - second);
          const expected = Math.fround(distances[requested - 1] * Math.fround(1.0000001));
          expect(Math.abs(bandwidths[focal * 5] - expected)).toBeLessThanOrEqual(
            1e-6 * Math.max(1, expected)
          );
        }
      }
  } finally {
    scene.device.destroy();
  }
});

it('bounds the node count at the nominal indexed row maximum on a sufficiently large device', () => {
  const scene = createScene(GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT, 11);
  try {
    Object.assign(scene.device.limits, {
      maxStorageBufferBindingSize: 2 ** 30,
      maxBufferSize: 2 ** 30
    });
    const contributor = new GPUGeographicallyTemporallyWeightedRegression({
      ...scene.props,
      adaptive: true
    });
    const nodes = contributor.getCommandNodes(scene.graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.length).toBeGreaterThan(0);
  } finally {
    scene.device.destroy();
  }
});

function createMultiscaleProps(scene: ReturnType<typeof createScene>, iterationCount = 1) {
  return {
    ...scene.props,
    iterationCount,
    parameters: scene.view('float32', 6),
    output: {
      coefficients: scene.props.output.coefficients,
      fitted: scene.props.output.fitted,
      residuals: scene.props.output.residuals!,
      localStatus: scene.props.output.localStatus,
      scores: scene.view('float32', iterationCount)
    }
  };
}

it('rejects unsupported MGTWR options instead of silently fitting a different model', () => {
  const scene = createScene();
  try {
    const props = createMultiscaleProps(scene);
    for (const unsupported of [
      {gaussianCutoff: 1},
      {searchStructure: 'grid' as const},
      {searchStructure: 'brute-force' as const}
    ]) {
      expect(
        () =>
          new GPUMultiscaleGeographicallyTemporallyWeightedRegression({...props, ...unsupported})
      ).toThrow(/gtwr-scale (gaussianCutoff|searchStructure) is not supported/);
    }
    // These assertions are checked by scoped tsc as well as documenting the public exclusion.
    const excludesCutoff: 'gaussianCutoff' extends keyof GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps
      ? false
      : true = true;
    const excludesRoute: 'searchStructure' extends keyof GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps
      ? false
      : true = true;
    expect(excludesCutoff && excludesRoute).toBe(true);
    expect(PUBLIC_GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT).toBe(GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT);
    expect(PUBLIC_GPU_GTWR_INDEXED_COMMAND_NODE_ALLOWANCE).toBe(256);
  } finally {
    scene.device.destroy();
  }
});

it('keeps MGTWR initialization reference-only and its composed node estimate exact', () => {
  for (const rows of [36, 32768]) {
    const scene = createScene(rows);
    try {
      const contributor = new GPUMultiscaleGeographicallyTemporallyWeightedRegression(
        createMultiscaleProps(scene)
      );
      const nodes = contributor.getCommandNodes(scene.graph);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      expect(nodes.length).toBe(10);
      expect(nodes.some(node => node.id.includes('-grid-'))).toBe(false);
    } finally {
      scene.device.destroy();
    }
  }
});

it('keeps MGTWR shaders and pipelines constant as iteration count grows and writes each score slot', () => {
  const pipelineCounts = [];
  const shaderCounts = [];
  for (const iterationCount of [1, 2, 4, 32]) {
    const scene = createScene();
    try {
      recordWGSLKernelNodes(scene.graph);
      const props = createMultiscaleProps(scene, iterationCount);
      const contributor = new GPUMultiscaleGeographicallyTemporallyWeightedRegression(props);
      const nodes = contributor.getCommandNodes(scene.graph);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      pipelineCounts.push(getGPUCommandNodePipelineKeys(nodes).size);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      shaderCounts.push(new Set(kernels.map(kernel => kernel.source)).size);
      for (const kernel of kernels) {
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        expectAllowedWGSLIdentifiers(kernel.source, kernel.id);
      }
      const memory = createCPUGraphMemory();
      const startNode = nodes.find(node => node.id === 'gtwr-scale-start')!;
      const startRecord = getWGSLKernelNodeRecord(scene.graph, startNode.id)!;
      const iterationState = startRecord.bindings.find(
        binding => binding.name === 'iterationState'
      )!.view;
      // A new encoding must restart the GPU-written score index, including after an old run.
      memory.write(iterationState, new Uint32Array([31]));
      executeCommandNodesOnCPU([startNode], memory);
      expect(memory.readUint32(iterationState)[0]).toBe(0);
      const scoreNodes = nodes.filter(node => node.id.endsWith('-score'));
      for (const node of scoreNodes) {
        const record = getWGSLKernelNodeRecord(scene.graph, node.id)!;
        for (const binding of record.bindings) {
          if (binding.name === 'points')
            memory.write(binding.view, new Float32Array(scene.props.response.length * 3));
          if (binding.name === 'previous' || binding.name === 'next')
            memory.write(binding.view, new Float32Array(scene.props.response.length * 2).fill(1));
        }
      }
      executeCommandNodesOnCPU(scoreNodes, memory);
      expect(memory.readUint32(iterationState)[0]).toBe(iterationCount);
      expect(Array.from(memory.readFloat32(props.output.scores))).toEqual(
        new Array(iterationCount).fill(0)
      );
    } finally {
      scene.device.destroy();
    }
  }
  expect(new Set(pipelineCounts).size).toBe(1);
  expect(new Set(shaderCounts).size).toBe(1);
});

for (const rows of [10, 36])
  for (const responseValue of [1, 0.1, 1e30]) {
    it(`executes centered summary row reductions for ${rows} constant responses ${responseValue}`, () => {
      const scene = createScene(rows);
      try {
        recordWGSLKernelNodes(scene.graph);
        const nodes = createGTWRIndexedSummaryNodes(scene.graph, 'constant', scene.props);
        const records = nodes.map(node => getWGSLKernelNodeRecord(scene.graph, node.id)!);
        const statisticsPartial = records.find(
          record => record.id === 'constant-statistics-partial'
        )!;
        const statisticsFold = records.find(record => record.id === 'constant-statistics-fold')!;
        const variancePartial = records.find(record => record.id === 'constant-variance-partial')!;
        const publication = records.find(record => record.id === 'constant-variance-summary')!;
        const memory = createCPUGraphMemory();
        const observed = Math.fround(responseValue);
        memory.write(scene.props.response, new Float32Array(rows).fill(observed));
        memory.write(
          scene.props.output.fitted,
          new Float32Array(rows).fill(Math.fround(observed + 0.125))
        );
        memory.write(scene.props.output.hatDiagonal, new Float32Array(rows).fill(0.25));
        memory.write(scene.props.output.localStatus, new Uint32Array(rows));
        const anchor = statisticsPartial.bindings.find(
          binding => binding.name === 'responseAnchor'
        )!.view;
        // The published min/max primitive is represented by its independent closed-form result.
        // Execute both production row maps, including the divisions and mean reconstruction that
        // caused the review defect; only cooperative addition is emulated with float32 rounding.
        memory.write(anchor, [observed, 0, observed, 0]);
        const mapRows = (record: typeof statisticsPartial, start: string, quantities: number) => {
          const mappedValues = scene.view('float32', rows * quantities);
          const end = record.body.indexOf('\n    if (group >= GROUP_COUNT)');
          const snippet = record.body.slice(record.body.indexOf(start), end).replace(/}\s*$/, '');
          const probe = createWGSLKernelNode(scene.graph, {
            id: `${record.id}-row-probe`,
            operation: 'GTWRVarianceRowProbe',
            invocationCount: rows,
            bindings: [
              ...record.bindings,
              {name: 'mappedValues', view: mappedValues, type: 'f32', access: 'read_write'}
            ],
            declarations: `const ROWS: u32 = ${rows}u;`,
            body: `let row = index; var group = 0u; ${Array.from({length: quantities}, (_, quantity) => `var value${quantity} = 0.0;`).join(' ')}
${snippet}
${Array.from({length: quantities}, (_, quantity) => `mappedValues[mappedValuesOffset + index * ${quantities}u + ${quantity}u] = value${quantity};`).join('\n')}`
          });
          executeCommandNodesOnCPU([probe], memory);
          const values = memory.readFloat32(mappedValues);
          return Array.from({length: quantities}, (_, quantity) => {
            let sum = 0;
            for (let row = 0; row < rows; row++)
              sum = Math.fround(sum + values[row * quantities + quantity]);
            return sum;
          });
        };
        const statistics = mapRows(statisticsPartial, 'let code =', 4);
        expect(statistics[2]).toBe(0);
        for (let quantity = 0; quantity < 4; quantity++)
          memory.write(
            statisticsFold.bindings.find(binding => binding.name === `output${quantity}`)!.view,
            [statistics[quantity]]
          );
        memory.write(
          statisticsFold.bindings.find(binding => binding.name === 'countOutput')!.view,
          new Uint32Array([rows])
        );
        const [totalSquares] = mapRows(variancePartial, 'if (localStatus[', 1);
        expect(totalSquares).toBe(0);
        const body = publication.body
          .slice(publication.body.indexOf('let observations ='))
          .replace(/}\s*$/, '');
        const probe = createWGSLKernelNode(scene.graph, {
          id: 'constant-publish-probe',
          operation: 'GTWRVariancePublishProbe',
          invocationCount: 1,
          bindings: publication.bindings.filter(
            binding => !['partialSums', 'output0'].includes(binding.name)
          ),
          declarations: GET_NAN_WGSL,
          body: `let sum0 = ${totalSquares}.0;\n${body}`
        });
        executeCommandNodesOnCPU([probe], memory);
        expect(memory.readFloat32(scene.props.output.summary)[3]).toBeNaN();
      } finally {
        scene.device.destroy();
      }
    });
  }

it('rejects aligned writable binding aliases even when logical subviews do not overlap', () => {
  const scene = createScene(2);
  try {
    const storage = scene.view('float32', 128);
    const fitted = scene.graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 0
    });
    const hatDiagonal = scene.graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 128
    });
    const first = getViewBindingRange(fitted);
    const second = getViewBindingRange(hatDiagonal);
    expect(first.offset + first.size).toBeGreaterThan(second.offset);
    expect(
      () =>
        new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          output: {...scene.props.output, fitted, hatDiagonal}
        })
    ).toThrow(/gtwr-scale.*aligned binding byte ranges/);
    const props = createMultiscaleProps(scene);
    expect(
      () =>
        new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
          ...props,
          output: {...props.output, fitted, residuals: hatDiagonal}
        })
    ).toThrow(/gtwr-scale.*aligned binding byte ranges/);
  } finally {
    scene.device.destroy();
  }
});

it('rejects physical writable aliases hidden behind distinct imported handles', () => {
  const scene = createScene(2);
  const buffer = createOutputBuffer(scene.device, 128);
  try {
    const fitted = importGraphBuffer(scene.graph, 'physical-fitted', buffer, 'float32', 2);
    const other = importGraphBuffer(scene.graph, 'physical-other', buffer, 'float32', 2);
    expect(fitted.buffer).not.toBe(other.buffer);
    expect(
      () =>
        new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          output: {...scene.props.output, fitted, hatDiagonal: other}
        })
    ).toThrow(/gtwr-scale.*logical writable byte ranges/);
    const multiscale = createMultiscaleProps(scene);
    expect(
      () =>
        new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
          ...multiscale,
          output: {...multiscale.output, fitted, residuals: other}
        })
    ).toThrow(/gtwr-scale.*logical writable byte ranges/);
    expect(
      () =>
        new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          response: other,
          output: {...scene.props.output, fitted}
        })
    ).toThrow('gtwr-scale outputs must not share buffers with inputs');
    expect(
      () =>
        new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
          ...multiscale,
          response: other,
          output: {...multiscale.output, fitted}
        })
    ).toThrow('gtwr-scale outputs must not share buffers with inputs');
  } finally {
    buffer.destroy();
    scene.device.destroy();
  }
});

it('keeps extreme finite settings in runtime parameters and emits valid numeric literals', () => {
  const maximumFloat32 = Math.fround(3.4028234663852886e38);
  const minimumFloat32 = 2 ** -149;
  expect(
    Array.from(
      getGPUGTWRParameterValues({
        bandwidth: maximumFloat32,
        timeScale: maximumFloat32
      })
    )
  ).toEqual([maximumFloat32, maximumFloat32, 0, 0]);
  for (const value of [NaN, Infinity, -Infinity, 1e100]) {
    expect(() => getGPUGTWRParameterValues({bandwidth: value, timeScale: 1})).toThrow();
    expect(() => getGPUGTWRParameterValues({bandwidth: 1, timeScale: value})).toThrow();
  }
  for (const gaussianCutoff of [minimumFloat32, maximumFloat32]) {
    const scene = createScene(257, 11);
    try {
      const nodes = new GPUGeographicallyTemporallyWeightedRegression({
        ...scene.props,
        kernel: 'gaussian',
        gaussianCutoff,
        adaptive: true
      }).getCommandNodes(scene.graph);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        expectAllowedWGSLIdentifiers(kernel.source, kernel.id);
        const source = kernel.source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '');
        expect(source).not.toMatch(/\b(?:NaN|Infinity|nan|inf)\b/);
        // All emitted decimal floating literals fit f32, including underflow guards and cutoff.
        for (const match of source.matchAll(
          /(?<![A-Za-z_0-9.])(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+)(?:[eE][+-]?\d+)?[fh]?\b/g
        )) {
          expect(
            Number.isFinite(Math.fround(Number(match[0].replace(/[fh]$/, '')))),
            match[0]
          ).toBe(true);
        }
        for (const match of source.matchAll(/(?<![A-Za-z_0-9.])(\d+)([ui])\b/g)) {
          expect(Number(match[1]), match[0]).toBeLessThanOrEqual(
            match[2] === 'u' ? 0xffffffff : 0x7fffffff
          );
        }
      }
    } finally {
      scene.device.destroy();
    }
  }
});

it('preflights aligned MGTWR caller bindings before any graph declaration', () => {
  const scene = createScene(36);
  try {
    const props = createMultiscaleProps(scene);
    const storage = scene.view('float32', 4097);
    props.parameters = scene.graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 4096,
      byteOffset: 4
    });
    const contributor = new GPUMultiscaleGeographicallyTemporallyWeightedRegression(props);
    Object.assign(scene.device.limits, {maxStorageBufferBindingSize: 16384});
    const graphState = () =>
      Object.fromEntries(
        Object.entries(scene.graph)
          .filter(([, value]) => value instanceof Map || Array.isArray(value))
          .map(([name, value]) => [name, value instanceof Map ? value.size : value.length])
      );
    const before = graphState();
    expect(() => contributor.getCommandNodes(scene.graph)).toThrow(/needs 16388 bytes/);
    expect(graphState()).toEqual(before);
  } finally {
    scene.device.destroy();
  }
});

it('rejects chunked GTWR inputs at construction before any chunk-specific graph declaration', () => {
  for (const chunkCount of [1, 2, 8]) {
    const scene = createScene(64);
    try {
      const response = createVectorView(
        'gtwr-chunks',
        'float32',
        Array.from({length: chunkCount}, () => scene.view('float32', 64 / chunkCount))
      );
      const graphState = () =>
        Object.fromEntries(
          Object.entries(scene.graph)
            .filter(([, value]) => value instanceof Map || Array.isArray(value))
            .map(([name, value]) => [name, value instanceof Map ? value.size : value.length])
        );
      const before = graphState();
      expect(
        () =>
          new GPUGeographicallyTemporallyWeightedRegression({
            ...scene.props,
            response: response as unknown as GraphDataView<'float32'>
          })
      ).toThrow('gtwr-scale response must be one packed view');
      expect(graphState()).toEqual(before);
    } finally {
      scene.device.destroy();
    }
  }
});

for (const route of ['grid', 'brute-force', 'multiscale'] as const) {
  it(`accepts disjoint co-bound writable ranges imported once, ${route}`, () => {
    const scene = createScene(2);
    const buffer = createOutputBuffer(scene.device, 128);
    try {
      const fitted = importGraphBuffer(scene.graph, 'packed-fit-output', buffer, 'float32', 2);
      const other = scene.graph.createDataView(fitted.buffer, {
        format: 'float32',
        length: 2,
        byteOffset: 256
      });
      expect(other.buffer).toBe(fitted.buffer);
      const multiscale = route === 'multiscale' ? createMultiscaleProps(scene) : undefined;
      const contributor = multiscale
        ? new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
            ...multiscale,
            output: {...multiscale.output, fitted, residuals: other}
          })
        : new GPUGeographicallyTemporallyWeightedRegression({
            ...scene.props,
            searchStructure: route === 'grid' ? 'grid' : 'brute-force',
            output: {...scene.props.output, fitted, hatDiagonal: other}
          });
      const nodes = contributor.getCommandNodes(scene.graph);
      // Declaration and recording compile every generated module without importing the core buffer twice.
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expectAllowedWGSLIdentifiers(kernel.source, kernel.id);
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      }
    } finally {
      buffer.destroy();
      scene.device.destroy();
    }
  });
  it(`rejects disjoint physical input/output ranges before declaring resources, ${route}`, () => {
    const scene = createScene(2);
    const buffer = createOutputBuffer(scene.device, 128);
    try {
      const response = importGraphBuffer(scene.graph, 'shared-input-output', buffer, 'float32', 2);
      const fitted = scene.graph.createDataView(response.buffer, {
        format: 'float32',
        length: 2,
        byteOffset: 256
      });
      const allocate = vi.spyOn(scene.graph, 'createTransientBuffer');
      const createView = vi.spyOn(scene.graph, 'createDataView');
      if (route === 'multiscale') {
        const props = createMultiscaleProps(scene);
        allocate.mockClear();
        createView.mockClear();
        expect(
          () =>
            new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
              ...props,
              response,
              output: {...props.output, fitted}
            })
        ).toThrow('gtwr-scale outputs must not share buffers with inputs');
      } else {
        expect(
          () =>
            new GPUGeographicallyTemporallyWeightedRegression({
              ...scene.props,
              searchStructure: route === 'grid' ? 'grid' : 'brute-force',
              response,
              output: {...scene.props.output, fitted}
            })
        ).toThrow('gtwr-scale outputs must not share buffers with inputs');
      }
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
    } finally {
      buffer.destroy();
      scene.device.destroy();
    }
  });
  it(`preflights every caller STORAGE flag without graph mutation, ${route}`, () => {
    const scene = createScene(2);
    try {
      const props = route === 'multiscale' ? createMultiscaleProps(scene) : scene.props;
      const contributor =
        route === 'multiscale'
          ? new GPUMultiscaleGeographicallyTemporallyWeightedRegression(
              props as ReturnType<typeof createMultiscaleProps>
            )
          : new GPUGeographicallyTemporallyWeightedRegression({
              ...scene.props,
              searchStructure: route
            });
      const outputs = Object.values(props.output).filter(
        (view): view is GraphDataView<'float32'> | GraphDataView<'uint32'> =>
          view !== undefined && 'buffer' in view
      );
      const views = [
        props.positions,
        props.times,
        props.predictors,
        props.response,
        props.mask!,
        props.parameters,
        ...outputs
      ];
      const allocate = vi.spyOn(scene.graph, 'createTransientBuffer');
      const createView = vi.spyOn(scene.graph, 'createDataView');
      const addNode = vi.spyOn(scene.graph, 'add');
      for (const view of views) {
        const usage = view.buffer.usage;
        Object.defineProperty(view.buffer, 'usage', {value: Buffer.COPY_DST});
        expect(() => contributor.getCommandNodes(scene.graph), view.buffer.id).toThrow(
          /usage must include Buffer.STORAGE/
        );
        expect(allocate).not.toHaveBeenCalled();
        expect(createView).not.toHaveBeenCalled();
        expect(addNode).not.toHaveBeenCalled();
        Object.defineProperty(view.buffer, 'usage', {value: usage});
      }
    } finally {
      scene.device.destroy();
    }
  });
}

for (const multiscale of [false, true]) {
  it(`accepts consecutive output ranges used in different dispatches, multiscale=${multiscale}`, () => {
    const scene = createScene(2);
    const buffer = createOutputBuffer(scene.device, 128);
    try {
      const first = importGraphBuffer(
        scene.graph,
        'consecutive-outputs',
        buffer,
        'float32',
        multiscale ? 1 : 4
      );
      const residuals = scene.graph.createDataView(first.buffer, {
        format: 'float32',
        length: 2,
        byteOffset: multiscale ? 4 : 16
      });
      if (multiscale) {
        const props = createMultiscaleProps(scene);
        const contributor = new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
          ...props,
          output: {...props.output, scores: first, residuals}
        });
        expect(() => contributor.getCommandNodes(scene.graph)).not.toThrow();
      } else {
        const contributor = new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          searchStructure: 'brute-force',
          output: {...scene.props.output, coefficients: first, residuals}
        });
        expect(() => contributor.getCommandNodes(scene.graph)).not.toThrow();
      }
    } finally {
      buffer.destroy();
      scene.device.destroy();
    }
  });
  it(`ignores unused default parameters during bandwidth search, multiscale=${multiscale}`, () => {
    const scene = createScene(2);
    try {
      const foreignGraph = new GPUCommandGraph(scene.device);
      const parameters = createTransientView(foreignGraph, 'unused-defaults', 'float32', 6);
      Object.defineProperty(parameters.buffer, 'usage', {value: Buffer.COPY_DST});
      const bandwidthSearch = {parameters: scene.view('float32', 6), maximumIterations: 1};
      const contributor = multiscale
        ? new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
            ...createMultiscaleProps(scene),
            parameters,
            bandwidthSearch
          })
        : new GPUGeographicallyTemporallyWeightedRegression({
            ...scene.props,
            parameters,
            bandwidthSearch
          });
      expect(() => contributor.getCommandNodes(scene.graph)).not.toThrow();
    } finally {
      scene.device.destroy();
    }
  });
}
for (const route of ['grid', 'brute-force', 'multiscale'] as const) {
  it(`rejects read-only/writable output sharing across disjoint aligned ranges, ${route}`, () => {
    const scene = createScene(2);
    const buffer = createOutputBuffer(scene.device, 128);
    try {
      const first = importGraphBuffer(
        scene.graph,
        'mixed-output-usage',
        buffer,
        'float32',
        route === 'multiscale' ? 4 : 2
      );
      const second = scene.graph.createDataView(first.buffer, {
        format: 'float32',
        length: 2,
        byteOffset: 256
      });
      if (route === 'multiscale') {
        const props = createMultiscaleProps(scene);
        expect(
          () =>
            new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
              ...props,
              output: {...props.output, coefficients: first, fitted: second}
            })
        ).toThrow(/one physical buffer read-only and writable/);
      } else {
        expect(
          () =>
            new GPUGeographicallyTemporallyWeightedRegression({
              ...scene.props,
              searchStructure: route,
              output: {...scene.props.output, fitted: first, residuals: second}
            })
        ).toThrow(/one physical buffer read-only and writable/);
      }
    } finally {
      buffer.destroy();
      scene.device.destroy();
    }
  });
}
