// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import typescript from 'typescript';
import {readFileSync, existsSync, statSync} from 'node:fs';
import {resolve, dirname, extname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPURaster,
  GPURasterNDVI,
  GPURasterTextureToBuffer,
  type GPURasterNDVIProps,
  type GPURasterTextureToBufferProps,
  type GPURasterTextureBand
} from '@luma.gl/graph/gpu-raster';
import {Texture} from '@luma.gl/core';
import {getGPUTerrainDerivativesParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {GPUTransitionMatrix} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  getGPURasterSnapshotEOFDecomposition,
  makeGPURasterBand,
  makeGPURasterMetadata,
  computeCostDistanceOnCPU,
  getGPUCostDistanceParameterValues,
  computeRasterRegionGraphOnCPU,
  getGPULagrangianFlowMapParameterValues,
  GPURasterPatchMetrics,
  GPURasterSieve,
  type GPURasterPatchMetricsProps,
  type GPURasterSieveProps
} from '@luma-contributors/analysis/raster-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  extractSnippets,
  getSnippetDiagnostics
} from '../../analysis-core/doc-snippet-diagnostics.test-utils';

it('DOC-110 typechecks every editable raster and terrain-processing fragment with strict inputs', () => {
  const pages = [
    'gpu-raster/README.md',
    'gpu-raster/operations.md',
    'gpu-raster/lagrangian-diagnostics-mob06.md',
    'gpu-terrain/README.md',
    'gpu-terrain/concepts.md',
    'gpu-raster/spectral-indices.md',
    'gpu-raster/remote-sensing.md',
    'gpu-raster/flow-map-and-particle-diagnostics.md',
    'gpu-raster/contrast-equalization.md',
    'gpu-raster/fractional-categorical-statistics.md',
    'gpu-raster/fractional-zonal-distribution.md',
    'gpu-raster/raster-embedding-codecs.md',
    'gpu-raster/raster-eof-analysis.md',
    'gpu-raster/raster-patches.md',
    'gpu-raster/spectral-filter.md',
    'gpu-terrain/terrain-processing.md'
  ];
  const failures = getSnippetDiagnostics(pages.flatMap(page => extractSnippets(page)));
  expect(failures.map(failure => `${failure.page}:${failure.line} ${failure.message}`)).toEqual([]);
}, 120000);

it('keeps the patch example mask and completion guards on both label consumers', () => {
  const code = extractSnippets('gpu-raster/raster-patches.md')[0].code;
  const source = typescript.createSourceFile(
    'patch-example.ts',
    code,
    typescript.ScriptTarget.Latest,
    true
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const word = (name: string, length: number) =>
      createTransientView(graph, name, 'uint32', length);
    const inputs: Record<string, unknown> = {
      graph,
      rasterMetadata: {
        width: 16,
        height: 16,
        affine: [1, 0, 0, 0, -1, 0],
        pixelInterpretation: 'area'
      },
      denseLabels: word('labels', 256),
      mask: word('mask', 256),
      converged: word('converged', 1),
      componentCount: word('component-count', 1),
      overflow: word('overflow', 1),
      sieveParameters: {importToGraph: () => word('parameters', 4)},
      sievedLabels: word('sieved-labels', 256),
      sievedCount: word('sieved-count', 1)
    };
    for (const name of ['pixelCounts', 'minColumns', 'minRows', 'maxColumns', 'maxRows'])
      inputs[name] = word(name, 256);
    for (const name of ['areas', 'perimeters'])
      inputs[name] = createTransientView(graph, name, 'float32', 256);
    for (const [name, Constructor] of [
      ['GPURasterPatchMetrics', GPURasterPatchMetrics],
      ['GPURasterSieve', GPURasterSieve]
    ] as const) {
      let propsText: string | undefined;
      const visit = (node: typescript.Node): void => {
        if (typescript.isNewExpression(node) && node.expression.getText(source) === name)
          propsText = node.arguments?.[0].getText(source);
        node.forEachChild(visit);
      };
      visit(source);
      expect(propsText).toBeDefined();
      // Evaluate the example's actual object literal with typed graph-view stand-ins for its
      // declared application inputs. No documentation text or constructor is reimplemented.
      const bind = new Function(...Object.keys(inputs), `return (${propsText});`) as (
        ...values: unknown[]
      ) => GPURasterSieveProps;
      const props = bind(...Object.values(inputs));
      expect(props.mask === inputs['mask'], name).toBe(true);
      expect(props.status, name).toEqual({
        converged: inputs['converged'],
        componentCount: inputs['componentCount'],
        overflow: inputs['overflow']
      });
      const nodes = new Constructor(
        props as GPURasterSieveProps & GPURasterPatchMetricsProps
      ).getCommandNodes(graph);
      expect(
        nodes.some(node => /-unresolved-arm$/.test(String(node.id))),
        name
      ).toBe(true);
    }
  } finally {
    device.destroy();
  }
});

it('documents and resolves rank-one EOFs at binary64 norm extremes', () => {
  // Rank-one [a,a;a,a] has eigenvalues [2a,0], hence exact explained fractions [1,0].
  for (const magnitude of [1e-200, 1, 1e200]) {
    const gramMatrix = Float64Array.of(magnitude, magnitude, magnitude, magnitude);
    const {explainedVarianceRatio} = getGPURasterSnapshotEOFDecomposition(gramMatrix, 2, 2);
    expect(Array.from(explainedVarianceRatio)).toEqual([1, 0]);
  }
  const page = readFileSync(
    new URL('../../../../../docs/api/gpu-raster/raster-eof-analysis.md', import.meta.url),
    'utf8'
  );
  expect(page).toContain('scaled Frobenius norm');
  expect(page).toContain('the explained fractions are `[1,0]`');
});

it('AMG-04 distinguishes the region graph iteration units and GPU limits', () => {
  const page = readFileSync(
    new URL('../../../../../docs/api/gpu-raster/remote-sensing.md', import.meta.url),
    'utf8'
  );
  expect(page).toContain('one invocation per region');
  expect(page).toContain('one synchronous label-propagation round');
  expect(page).toContain('one component-merging phase');
  expect(page).toContain('one closest-pair merge');
  expect(page).toContain('[0,0,2,2,4,4,6,6]');
  expect(page).not.toContain('dense and adaptive routes run the one-invocation merge');
});

it.each([
  ['gpu-raster/remote-sensing.md', "from: 'period-major'", 'URT-09'],
  ['gpu-raster/README.md', 'assertCompatibleWith', 'URT-08'],
  ['gpu-terrain/README.md', 'first-run-terrain-derivatives', 'FST-14'],
  ['gpu-raster/operations.md', 'GPUFocalStatistics', 'URT-12'],
  ['gpu-raster/raster-patches.md', 'denseLabels: patchLabels', 'UCT-01'],
  ['gpu-raster/remote-sensing.md', 'actual starting coefficients', 'RT-140'],
  ['gpu-raster/operations.md', 'raster-embeddings.md', 'DST-12']
])('%s documents %s (%s)', (path, expected) => {
  const page = readFileSync(new URL(`../../../../../docs/api/${path}`, import.meta.url), 'utf8');
  expect(page.includes(expected)).toBe(true);
  if (path === 'gpu-raster/remote-sensing.md' && expected === "from: 'period-major'") {
    assertDisplayedTemporalBindings(
      extractSnippets(path).find(snippet => snippet.code.includes('new GPURasterTemporal'))!.code
    );
  }
  if (path === 'gpu-terrain/README.md') {
    assertDisplayedTerrainLighting(
      extractSnippets(path).find(snippet => snippet.code.includes('function addTerrainExample'))!
        .code
    );
  }
});

it.each(['gpu-raster', 'gpu-terrain'])('DST-07 publishes an ordinary %s catalog link', family => {
  const page = readFileSync(
    new URL(`../../../../../docs/api/${family}/README.md`, import.meta.url),
    'utf8'
  );
  expect(page.includes('::lens-catalog')).toBe(false);
  expect(page).toMatch(/\[[^\]]+catalog[^\]]*\]\(\/explore\/[^)]*\)/i);
});

it('DST-10 gives each recording-only class target a direct contract link', () => {
  for (const [path, names] of [
    ['focal-pairs', ['GPURasterFocalPairs']],
    ['spectral-indices', ['GPURasterNDWI', 'GPURasterNBR', 'GPURasterEVI', 'GPURasterSAVI']],
    [
      'remote-sensing',
      [
        'GPURasterConservativeRemapping',
        'GPURasterTexture',
        'GPURasterTemporal',
        'GPURasterRegistration',
        'GPURasterSpectralUnmixing',
        'GPURasterSeasonalBreak'
      ]
    ]
  ] as const) {
    const page = readFileSync(
      new URL(`../../../../../docs/api/gpu-raster/${path}.md`, import.meta.url),
      'utf8'
    );
    for (const name of names) {
      const section = page.split(`### ${name}\n`)[1]?.split(/\n#{1,3} /)[0] ?? '';
      expect(section, name).toMatch(/\[[^\]]+\]\(#[^)]+\)/);
    }
  }
});

it('URT-08 executes both guide alignment functions and rejects a one-column shift', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = createTransientView(graph, 'aligned-red', 'float32', 4);
    const metadata = makeGPURasterMetadata({
      width: 2,
      height: 2,
      origin: [100, 200],
      cellSize: [30, 30],
      rowDirection: 'south'
    });
    const second = new GPURaster({
      metadata,
      bands: [makeGPURasterBand(view, metadata, {id: 'red'})]
    });
    const shifted = new GPURaster({
      metadata: {...metadata, affine: [30, 0, 130, 0, -30, 200]},
      bands: [makeGPURasterBand(view, metadata, {id: 'red'})]
    });
    for (const page of ['gpu-raster/README.md', 'gpu-terrain/README.md']) {
      const snippet = extractSnippets(page)[0].code;
      const source = typescript.createSourceFile(
        'alignment.ts',
        snippet,
        typescript.ScriptTarget.Latest,
        true
      );
      const declaration = source.statements.find(
        statement =>
          typescript.isFunctionDeclaration(statement) &&
          statement.name?.text === 'createAlignedRaster'
      );
      expect(declaration).toBeDefined();
      const executable = typescript.transpileModule(
        declaration!.getText(source).replace('export ', ''),
        {
          compilerOptions: {target: typescript.ScriptTarget.ES2022}
        }
      ).outputText;
      const run = new Function(
        'GPURaster',
        'makeGPURasterBand',
        'makeGPURasterMetadata',
        `${executable}; return createAlignedRaster;`
      )(GPURaster, makeGPURasterBand, makeGPURasterMetadata) as (
        view: GraphDataView<'float32'>,
        second: GPURaster
      ) => {first: GPURaster; firstCenter: readonly [number, number]};
      // Area-pixel center: [100 + 30/2, 200 - 30/2]. One-column shift adds 30 to x.
      expect(run(view, second).firstCenter).toEqual([115, 185]);
      expect(shifted.getPixelWorldPosition(0, 0)).toEqual([145, 185]);
      expect(() => run(view, shifted)).toThrow('raster grids and coordinate systems must match');
    }
  } finally {
    device.destroy();
  }
});

it('URT-08 pins declared CRS versus selected cost-distance units', () => {
  const geographic = {
    ...makeGPURasterMetadata({width: 2, height: 1, cellSize: 1, origin: [0, 0.5]}),
    coordinateReferenceSystem: {authority: 'EPSG:4326'}
  };
  const input = {
    friction: {id: 'friction', format: 'float32' as const, values: Float32Array.of(1, 1)},
    parameters: getGPUCostDistanceParameterValues({}, geographic),
    sources: Uint32Array.of(0)
  };
  // Equatorial WGS84 eastward arc: 6378137*pi/180 = 111319.490793..., rounded to float32.
  expect(Array.from(computeCostDistanceOnCPU(input, {grid: geographic}).costs)).toEqual([0, 1]);
  expect(
    Array.from(
      computeCostDistanceOnCPU(input, {grid: geographic, cellSizeMode: 'geographic'}).costs
    )
  ).toEqual([0, 111319.4921875]);
  const projected = makeGPURasterMetadata({width: 3, height: 1, cellSize: 10});
  // Each 10 m edge has mean friction (1+3)/2=2, hence costs [0,20,40].
  const result = computeCostDistanceOnCPU(
    {
      ...input,
      friction: {...input.friction, values: Float32Array.of(1, 3, 1)},
      parameters: getGPUCostDistanceParameterValues({}, projected)
    },
    {grid: projected}
  );
  expect(Array.from(result.costs)).toEqual([0, 20, 40]);
});

it('AMG-04 pins route-specific partial mappings independently on an eight-region chain', () => {
  const labels = Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7);
  const values = Float32Array.of(0, 0, 10, 10, 20, 20, 30, 30);
  const options = {
    grid: makeGPURasterMetadata({width: 8, height: 1}),
    regionCount: 8,
    mergeThreshold: 10,
    maximumIterations: 1
  };
  // Dense reads previous neighbors, moving each minimum at most one edge. Boruvka selects
  // the zero-weight edge inside each pair first; it joins exactly those four pairs in phase 1.
  const dense = computeRasterRegionGraphOnCPU(labels, values, {
    ...options,
    searchStructure: 'dense'
  });
  const boruvka = computeRasterRegionGraphOnCPU(labels, values, {
    ...options,
    searchStructure: 'boruvka'
  });
  expect(Array.from(dense.mapping)).toEqual([0, 0, 1, 2, 3, 4, 5, 6]);
  expect(Array.from(boruvka.mapping)).toEqual([0, 0, 2, 2, 4, 4, 6, 6]);
  expect(Array.from(dense.iterationLimitReached)).toEqual([1]);
  expect(Array.from(boruvka.iterationLimitReached)).toEqual([1]);
});

it('DST-15 pins the canonical flow example local placement and rebased instants', () => {
  // Exact packer expectation from binary64 subtraction: origin-worldOrigin=0 and epoch-epoch=0.
  expect(
    Array.from(
      getGPULagrangianFlowMapParameterValues({
        fieldPlacement: {origin: [10_000_000, 100], cellSize: [100, -100]},
        localOrigin: [10_000_000, 0],
        timeOrigin: 1_700_000_000,
        startTime: 1_700_000_000,
        timeStep: 10,
        sliceStartTime: 1_700_000_000,
        sliceEndTime: 1_700_000_010
      })
    )
  ).toEqual([0, 100, 100, -100, 0, 10, 0, 10]);
});

const FAMILY_GUIDES = [
  'gpu-raster/README.md',
  'gpu-raster/operations.md',
  'gpu-raster/spectral-indices.md',
  'gpu-raster/remote-sensing.md',
  'gpu-raster/focal-pairs.md',
  'gpu-raster/raster-patches.md',
  'gpu-raster/tile-reprojection.md',
  'gpu-raster/lagrangian-diagnostics-mob06.md',
  'gpu-raster/flow-map-and-particle-diagnostics.md',
  'gpu-raster/expansion-lenses.md',
  'gpu-terrain/README.md',
  'gpu-terrain/concepts.md',
  'gpu-terrain/height-above-drainage.md',
  'gpu-terrain/expansion-lenses.md'
];

it('ENT-09 identifies the public entry on every directly landed operation guide', () => {
  for (const path of FAMILY_GUIDES.filter(path =>
    /(?:diagnostics|focal-pairs|raster-patches|tile-reprojection|expansion-lenses|height-above-drainage)/.test(
      path
    )
  )) {
    const entry = path.startsWith('gpu-terrain/') ? 'gpu-terrain' : 'raster-analysis';
    const page = readFileSync(new URL(`../../../../../docs/api/${path}`, import.meta.url), 'utf8');
    expect(page, path).toContain(`Import from \`@luma-contributors/analysis/${entry}\``);
  }
});

it('DST-12 links every sidebar family page directly from its own landing', () => {
  const tableOfContents: unknown = JSON.parse(
    readFileSync(new URL('../../../../../docs/table-of-contents.json', import.meta.url), 'utf8')
  );
  const destinations: string[] = [];
  const collect = (value: unknown): void => {
    if (typeof value === 'string' && value.startsWith('api/')) destinations.push(value);
    else if (Array.isArray(value)) value.forEach(collect);
    else if (value && typeof value === 'object' && 'items' in value) collect(value.items);
  };
  collect(tableOfContents);
  for (const family of ['gpu-raster', 'gpu-terrain']) {
    const page = readFileSync(
      new URL(`../../../../../docs/api/${family}/README.md`, import.meta.url),
      'utf8'
    );
    for (const destination of destinations.filter(
      destination => destination.startsWith(`api/${family}/`) && !destination.endsWith('/README')
    )) {
      const relative = destination.slice(`api/${family}/`.length);
      expect(page.includes(`](./${relative}.md)`), destination).toBe(true);
    }
  }
});

it('DST-10/DST-12/DST-15 resolve every local family-guide destination and fragment', () => {
  const apiRoot = fileURLToPath(new URL('../../../../../docs/api/', import.meta.url));
  const failures: string[] = [];
  const getFragments = (text: string): Set<string> => {
    const fragments = new Set<string>();
    const counts = new Map<string, number>();
    for (const match of text.matchAll(/^#{1,6}\s+(.+?)\s*#*$/gm)) {
      const base = match[1]
        .toLowerCase()
        .replace(/[^\p{L}\p{N}_\-\s]/gu, '')
        .replace(/ /g, '-');
      const count = counts.get(base) ?? 0;
      counts.set(base, count + 1);
      fragments.add(count ? `${base}-${count}` : base);
    }
    for (const match of text.matchAll(/<a\s+id=["']([^"']+)["'][^>]*>/g)) fragments.add(match[1]);
    return fragments;
  };
  for (const page of FAMILY_GUIDES) {
    const path = resolve(apiRoot, page);
    for (const match of readFileSync(path, 'utf8').matchAll(/\[[^\]]*\]\(([^\s)]+)\)/g)) {
      const target = match[1];
      if (
        target.includes('://') ||
        target.startsWith('mailto:') ||
        (target.startsWith('/') && !target.startsWith('/docs/api/'))
      )
        continue;
      const [relative, fragment] = target.split('#');
      let destination = relative.startsWith('/docs/api/')
        ? resolve(apiRoot, relative.slice('/docs/api/'.length))
        : relative
          ? resolve(dirname(path), relative)
          : path;
      if (existsSync(destination) && statSync(destination).isDirectory())
        destination = resolve(destination, 'README.md');
      else if (!existsSync(destination) && !extname(destination))
        destination = existsSync(`${destination}.md`)
          ? `${destination}.md`
          : resolve(destination, 'README.md');
      if (!existsSync(destination)) failures.push(`${page} -> missing ${target}`);
      else if (
        fragment &&
        !getFragments(readFileSync(destination, 'utf8')).has(decodeURIComponent(fragment))
      )
        failures.push(`${page} -> missing fragment ${target}`);
    }
  }
  expect(failures).toEqual([]);
});

function readFamilyGuide(page: string): string {
  return readFileSync(new URL(`../../../../../docs/api/${page}`, import.meta.url), 'utf8');
}

it('REVIEW B1 shows a checked texture gather with validity reaching NDVI', () => {
  const snippets = extractSnippets('gpu-raster/spectral-indices.md');
  expect(snippets.some(snippet => snippet.code.includes('function addTextureNDVI'))).toBe(true);
  expect(readFamilyGuide('gpu-raster/remote-sensing.md')).toContain(
    'texture-capable zero-adjustment SAVI'
  );
  expect(readFamilyGuide('gpu-raster/spectral-indices.md')).not.toContain(
    'that route accepts buffer\nor texture bands'
  );
});

it.each([
  'GPULagrangianFlowMap',
  'GPUFiniteTimeLyapunovExponent',
  'GPUScalarFieldRidges'
])('REVIEW B2 assigns %s only to the canonical guide during parameter generation', async className => {
  const {generateDocPages} = await import('../../analysis-core/doc-parameters');
  const pages = new Map([
    [
      'gpu-raster/flow-map-and-particle-diagnostics.md',
      readFamilyGuide('gpu-raster/flow-map-and-particle-diagnostics.md')
    ],
    [
      'gpu-raster/lagrangian-diagnostics-mob06.md',
      readFamilyGuide('gpu-raster/lagrangian-diagnostics-mob06.md')
    ]
  ]);
  const generated = generateDocPages([{className, packers: []}], pages);
  expect(generated.undocumented).toEqual([]);
  expect(generated.pages.get('gpu-raster/flow-map-and-particle-diagnostics.md')).toContain(
    `parameters: ${className}`
  );
  expect(generated.pages.get('gpu-raster/lagrangian-diagnostics-mob06.md')).not.toContain(
    'parameters:'
  );
});

it('REVIEW B3 distinguishes GDAL geotransforms from library affines', () => {
  const page = readFamilyGuide('gpu-terrain/concepts.md');
  expect(page.includes('GDAL affine order is `[GT1')).toBe(false);
  expect(page).toContain('[700,20,0,350,0,-10]');
  expect(page).toContain('[20,0,700,0,-10,350]');
  expect(page).toContain('[710,345]');
});

it.each([
  'flow-map-and-particle-diagnostics.md',
  'lagrangian-diagnostics-mob06.md'
])('REVIEW B4 gives spatial connectivity ownership and route-specific limits on %s', page => {
  const text = readFamilyGuide(`gpu-raster/${page}`);
  expect(text).toContain('65,535');
  expect(text).toContain('diagnostics');
  expect(text).toContain(
    "import {GPUTransitionMatrix} from '@luma-contributors/analysis/gpu-spatial-analysis'"
  );
});

it('REVIEW B5 separates available flow recordings from catalog examples', () => {
  const page = readFamilyGuide('gpu-raster/expansion-lenses.md').split(
    '## Flow diagnostics recordings'
  )[1];
  expect(page).toContain('./flow-map-and-particle-diagnostics.md#gpuscalarfieldridges');
  expect(page).toContain('./operations-analysis.md#gpuparticleadvection');
  expect(page).toContain('./operations-analysis.md#gpustreamlines');
  expect(page).toContain('/fn/lagrangian-flow-map');
  expect(page).toContain('/fn/lagrangian-finite-time-lyapunov-exponent');
  expect(page).not.toContain('contracts beside their recordings');
  expect(page).toContain('No dedicated connectivity or residence-time recording');
});

function getConstructorArguments(code: string, className: string): string[] {
  const source = typescript.createSourceFile(
    'displayed-recipe.ts',
    code,
    typescript.ScriptTarget.Latest,
    true
  );
  const argumentsText: string[] = [];
  const visit = (node: typescript.Node): void => {
    if (typescript.isNewExpression(node) && node.expression.getText(source) === className) {
      expect(node.arguments?.[0], className).toBeDefined();
      argumentsText.push(node.arguments![0].getText(source));
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return argumentsText;
}

function evaluateDisplayedExpression(
  expression: string,
  inputs: Record<string, unknown>
): Record<string, unknown> {
  return new Function(...Object.keys(inputs), `return (${expression});`)(
    ...Object.values(inputs)
  ) as Record<string, unknown>;
}

function assertDisplayedTemporalBindings(code: string): void {
  const names = [
    'dateMajorNDVI',
    'dateMajorQA',
    'cellMajorNDVI',
    'cellMajorQA',
    'values',
    'counts',
    'chosenSlice',
    'status'
  ];
  const inputs: Record<string, unknown> = Object.fromEntries(
    names.map(name => [name, Object.freeze({name})])
  );
  inputs['cellCount'] = 2;
  inputs['sliceCount'] = 2;
  const temporalArguments = getConstructorArguments(code, 'GPURasterTemporal');
  expect(temporalArguments).toHaveLength(1);
  const temporal = evaluateDisplayedExpression(temporalArguments[0], inputs);
  expect(temporal['values']).toBe(inputs['cellMajorNDVI']);
  expect(temporal['mask']).toBe(inputs['cellMajorQA']);
  expect(temporal['operation']).toBe('mean');
  expect(temporal['output']).toEqual(
    Object.fromEntries(
      ['values', 'counts', 'chosenSlice', 'status'].map(name => [name, inputs[name]])
    )
  );
  const transposes = getConstructorArguments(code, 'GPUCubeLayoutTranspose').map(expression =>
    evaluateDisplayedExpression(expression, inputs)
  );
  expect(transposes).toHaveLength(2);
  for (const [source, destination] of [
    ['dateMajorNDVI', 'cellMajorNDVI'],
    ['dateMajorQA', 'cellMajorQA']
  ]) {
    const transpose = transposes.find(props => props['output'] === inputs[destination]);
    expect(transpose).toBeDefined();
    expect(transpose!['values']).toBe(inputs[source]);
    expect(transpose!['from']).toBe('period-major');
    expect(transpose!['cellCount']).toBe(2);
    expect(transpose!['periodCount']).toBe(2);
  }
}

function assertDisplayedTerrainLighting(code: string): void {
  const source = typescript.createSourceFile(
    'displayed-lighting.ts',
    code,
    typescript.ScriptTarget.Latest,
    true
  );
  let gridExpression: string | undefined;
  let packerExpression: string | undefined;
  const visit = (node: typescript.Node): void => {
    if (typescript.isCallExpression(node)) {
      if (node.expression.getText(source) === 'makeGPURasterMetadata')
        gridExpression = node.getText(source);
      if (node.expression.getText(source) === 'getGPUTerrainDerivativesParameterValues')
        packerExpression = node.getText(source);
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  expect(gridExpression).toBeDefined();
  expect(packerExpression).toBeDefined();
  const grid = new Function('makeGPURasterMetadata', `return (${gridExpression});`)(
    makeGPURasterMetadata
  );
  const packed = new Function(
    'grid',
    'getGPUTerrainDerivativesParameterValues',
    `return (${packerExpression});`
  )(grid, getGPUTerrainDerivativesParameterValues) as Float32Array;
  // Horn's plane gradient is [1,0], so slope is 45 degrees; west light at 45 degrees aligns
  // with the [-1,0,1]/sqrt(2) normal. These are literal lighting/grid requirements, not text tokens.
  expect(Array.from(packed.slice(0, 5))).toEqual([30, 30, 1, 270, 45]);
}

it('REVIEW S1 detects a temporal mask in the wrong layout despite equal view types', () => {
  const code = extractSnippets('gpu-raster/remote-sensing.md').find(snippet =>
    snippet.code.includes('new GPURasterTemporal')
  )!.code;
  assertDisplayedTemporalBindings(code);
  expect(() =>
    assertDisplayedTemporalBindings(code.replace('mask: cellMajorQA', 'mask: dateMajorQA'))
  ).toThrow();
});

it('REVIEW S1 detects changed displayed lighting despite a valid first-run link', () => {
  const code = extractSnippets('gpu-terrain/README.md').find(snippet =>
    snippet.code.includes('function addTerrainExample')
  )!.code;
  assertDisplayedTerrainLighting(code);
  expect(() =>
    assertDisplayedTerrainLighting(code.replace('azimuthDegrees: 270', 'azimuthDegrees: 315'))
  ).toThrow();
});

it.each([
  'gpu-raster',
  'gpu-terrain'
])('REVIEW S2 preserves the historical expansion fragment for %s', family => {
  const page = readFamilyGuide(`${family}/expansion-lenses.md`);
  expect(page).toContain('<a id="expansion-computation-lenses"></a>');
});

it.each([
  'gpu-raster',
  'gpu-terrain'
])('REVIEW S3 describes the cross-family catalog data filter for %s', family => {
  const section = readFamilyGuide(`${family}/README.md`).split('## Catalog')[1].split('## ')[0];
  expect(section).not.toContain('One figure per contributor');
  expect(section).toContain('input or output');
  expect(section).toContain('across families');
});

function executeDisplayedModule(
  code: string,
  inputs: Record<string, unknown>
): Record<string, unknown> {
  const source = typescript.createSourceFile(
    'displayed.ts',
    code,
    typescript.ScriptTarget.Latest,
    true
  );
  const body = source.statements
    .filter(statement => !typescript.isImportDeclaration(statement))
    .map(statement => statement.getText(source))
    .join('\n');
  const javascript = typescript.transpileModule(body, {
    compilerOptions: {
      target: typescript.ScriptTarget.ES2022,
      module: typescript.ModuleKind.CommonJS
    }
  }).outputText;
  const documentExports: Record<string, unknown> = {};
  new Function('exports', ...Object.keys(inputs), javascript)(
    documentExports,
    ...Object.values(inputs)
  );
  return documentExports;
}

it('REVIEW B1 executes the displayed texture route and carries each gather validity into NDVI', () => {
  const code = extractSnippets('gpu-raster/spectral-indices.md').find(snippet =>
    snippet.code.includes('function addTextureNDVI')
  )!.code;
  const device = createNullWebGPUDevice();
  const gathers: GPURasterTextureToBufferProps[] = [];
  const indices: GPURasterNDVIProps[] = [];
  class CapturedGather extends GPURasterTextureToBuffer {
    constructor(props: GPURasterTextureToBufferProps) {
      super(props);
      gathers.push(props);
    }
  }
  class CapturedNDVI extends GPURasterNDVI {
    constructor(props: GPURasterNDVIProps) {
      super(props);
      indices.push(props);
    }
  }
  try {
    const graph = new GPUCommandGraph(device);
    const bands: GPURasterTextureBand[] = ['nir', 'red'].map(
      (id): GPURasterTextureBand => ({
        id,
        format: 'uint32',
        scale: 0.0000275,
        offset: -0.2,
        noDataValue: 0,
        validity: createTransientView(graph, `${id}-source-validity`, 'uint32', 2),
        storage: {
          kind: 'texture',
          view: graph.createTextureView(
            graph.createTransientTexture({
              id: `${id}-texture`,
              width: 2,
              height: 1,
              format: 'r32uint',
              usage: Texture.SAMPLE
            })
          )
        }
      })
    );
    const outputs = executeDisplayedModule(code, {
      createTransientView,
      GPURasterTextureToBuffer: CapturedGather,
      GPURasterNDVI: CapturedNDVI
    });
    const add = outputs['addTextureNDVI'] as (...values: unknown[]) => void;
    add(
      graph,
      2,
      1,
      bands[0],
      bands[1],
      createTransientView(graph, 'ndvi', 'float32', 2),
      createTransientView(graph, 'ndvi-validity', 'uint32', 2)
    );
    expect(gathers).toHaveLength(2);
    expect(indices).toHaveLength(1);
    for (const [index, name] of ['nearInfrared', 'red'].entries()) {
      expect(gathers[index].input).toBe(bands[index]);
      expect(gathers[index].applyCalibration).toBe(true);
      const band = name === 'nearInfrared' ? indices[0].nearInfrared : indices[0].red;
      expect(band.storage.values).toBe(gathers[index].output);
      expect(band.validity).toBe(gathers[index].outputValidity);
      expect(band.scale ?? 1).toBe(1);
      expect(band.offset ?? 0).toBe(0);
    }
  } finally {
    device.destroy();
  }
});

it('REVIEW B4 constructs 65-class connectivity while rejecting diagnostics above 64', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      classes: createTransientView(graph, 'classes', 'uint32', 2),
      periodCount: 2,
      classCount: 65,
      output: {
        counts: createTransientView(graph, 'counts', 'uint32', 65 * 65),
        probabilities: createTransientView(graph, 'probabilities', 'float32', 65 * 65)
      }
    };
    expect(new GPUTransitionMatrix(props).getCommandNodes(graph).length).toBeGreaterThan(0);
    expect(
      () =>
        new GPUTransitionMatrix({
          ...props,
          output: {
            ...props.output,
            stateSparsity: createTransientView(graph, 'state-sparsity', 'uint32', 2)
          }
        })
    ).toThrow(/64/);
  } finally {
    device.destroy();
  }
});

it('REVIEW B2 preserves incoming historical flow fragments as canonical links', () => {
  const page = readFamilyGuide('gpu-raster/lagrangian-diagnostics-mob06.md');
  for (const fragment of [
    'gpulagrangianflowmap',
    'gpulagrangianflowmap-parameters',
    'gpufinitetimelyapunovexponent',
    'gpufinitetimelyapunovexponent-parameters',
    'gpuscalarfieldridges',
    'gpuscalarfieldridges-parameters'
  ]) {
    expect(page).toContain(`<a id="${fragment}"></a>`);
    expect(page).toContain(`./flow-map-and-particle-diagnostics.md#${fragment}`);
  }
  expect(page).not.toContain('{/* parameters:');
});

it('REVIEW typechecks every owned family-guide fragment independently of concurrently edited pages', () => {
  const failures = getSnippetDiagnostics(FAMILY_GUIDES.flatMap(page => extractSnippets(page)));
  expect(failures.map(failure => `${failure.page}:${failure.line} ${failure.message}`)).toEqual([]);
}, 120000);
