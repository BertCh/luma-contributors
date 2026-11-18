// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {maskWGSLComments} from '@luma.gl/shaders/wgsl';
import {describe, expect, it} from 'vitest';
import {expectGridCoordinateReaderSource} from '../utils/grid-coordinate-pins.test-utils';

// Math correctness review, luma-official-02. AGENTS.md forbids copied luma.gl source. A few kernels
// still hold a private copy of a WGSL function that published luma.gl does not export. The five
// grid readers now interpolate the public export; their generated-source pins remain below.
// Each remaining copy is a registered
// workaround (`dev-docs/upstream-workarounds.md`, section 2) and needs a pin: a spec that compares
// the copy with the text of the official kernel, so a luma.gl upgrade that changes one side fails a
// test instead of silently changing results next to cell edges.
//
// This spec is the list of every known copy. It fails when
// - a listed copy has no pin spec, or its pin spec does not name both files,
// - a WGSL function in `modules/analysis/src` is identical to a function in the published
//   `@luma.gl/graph` or `@luma.gl/shaders` kernels and is not listed here,
// - a listed copy no longer exists (remove the row here and in the register).
// Kernel files are only read as text; the public WGSL comment masker handles their comments.

const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const THIS_SPEC = 'modules/analysis/test/analysis-core/official-wgsl-copies.node.spec.ts';
const ANALYSIS_SOURCE = 'modules/analysis/src';
const OFFICIAL_ROOTS = ['node_modules/@luma.gl/graph/dist', 'node_modules/@luma.gl/shaders/dist'];
const GRID_AGGREGATION_KERNEL = 'node_modules/@luma.gl/graph/dist/gpu-core/gpu-grid-aggregation.js';

/**
 * A function shorter than this (whitespace removed) is an idiom, not a copy: the one-line
 * exponent-mask finite test `(bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u` is 87 characters and
 * is written the same way in about thirty local kernels and in several official ones.
 */
const MINIMUM_COPY_LENGTH = 100;

type OfficialWGSLCopy = {
  /** WGSL function name in the local file. */
  functionName: string;
  /** Local file that holds the copy. */
  localFile: string;
  /** Official kernels whose function of the same name the copy must equal, character for character. */
  officialFiles: string[];
  /** The spec that compares the texts: a per-copy spec, or this file. */
  pinSpec: string;
  /** Why the copy exists and what equality protects. */
  reason: string;
};

const OFFICIAL_WGSL_COPIES: OfficialWGSLCopy[] = [
  {
    functionName: 'encodeOrderedFloat',
    localFile:
      'modules/analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts',
    officialFiles: [GRID_AGGREGATION_KERNEL],
    pinSpec: THIS_SPEC,
    reason:
      'Order-preserving float key of the fused zone reduction, the same text as the official min and max aggregation; the local kernel decodes its own keys, pinned so a luma.gl upgrade shows the drift.'
  }
];

type WGSLFunctionText = {
  name: string;
  /** `fn name(...) -> type { ... }` with all whitespace removed. */
  text: string;
  /** The same text without `fn name`, to recognise a renamed copy. */
  signatureAndBody: string;
};

/** Every WGSL `fn name(...) { ... }` in `source`, with whitespace removed. */
function getWGSLFunctionTexts(source: string): WGSLFunctionText[] {
  source = maskWGSLComments(source);
  const functions: WGSLFunctionText[] = [];
  const pattern = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g;
  for (let match = pattern.exec(source); match !== null; match = pattern.exec(source)) {
    const open = source.indexOf('{', match.index);
    let depth = 0;
    for (let index = open; index >= 0 && index < source.length; index++) {
      if (source[index] === '{') {
        depth++;
      } else if (source[index] === '}') {
        depth--;
        if (depth === 0) {
          const text = source.slice(match.index, index + 1).replace(/\s+/g, '');
          functions.push({
            name: match[1],
            text,
            signatureAndBody: text.slice(`fn${match[1]}`.length)
          });
          break;
        }
      }
    }
  }
  return functions;
}

function listFiles(directory: string, namePattern: RegExp): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const filePath = join(directory, entry.name);
    return entry.isDirectory()
      ? listFiles(filePath, namePattern)
      : namePattern.test(entry.name)
        ? [filePath]
        : [];
  });
}

function readRepositoryFile(repositoryPath: string): string {
  return readFileSync(resolve(REPOSITORY_ROOT, repositoryPath), 'utf8');
}

function extractFunctionText(source: string, functionName: string, label: string): string {
  const declarations = [
    ...maskWGSLComments(source).matchAll(/\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g)
  ].filter(match => match[1] === functionName);
  expect(declarations, `${label}: exactly one active fn ${functionName}`).toHaveLength(1);
  const functions = getWGSLFunctionTexts(source).filter(
    wgslFunction => wgslFunction.name === functionName
  );
  expect(functions, `${label}: exactly one active fn ${functionName}`).toHaveLength(1);
  return functions[0].text;
}

function getFunctionText(repositoryPath: string, functionName: string): string {
  return extractFunctionText(readRepositoryFile(repositoryPath), functionName, repositoryPath);
}

function extractConstantText(source: string, constantName: string, label: string): string {
  const activeSource = maskWGSLComments(source);
  const declarations = [...activeSource.matchAll(/\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\b/g)].filter(
    match => match[1] === constantName
  );
  expect(declarations, `${label}: exactly one active const ${constantName}`).toHaveLength(1);
  const start = declarations[0].index;
  const end = activeSource.indexOf(';', start);
  expect(end, `${label}: const ${constantName} must end with a semicolon`).toBeGreaterThan(start);
  return activeSource.slice(start, end + 1).replace(/\s+/g, '');
}

function expectSemanticPin(
  source: string,
  functionName: string,
  expectedFunction: string,
  expectedConstants: readonly string[],
  label: string
): void {
  expect(extractFunctionText(source, functionName, label), label).toBe(expectedFunction);
  for (const constant of expectedConstants) {
    const constantName = /^const([A-Za-z_][A-Za-z0-9_]*):/.exec(constant)![1];
    expect(extractConstantText(source, constantName, label), label).toBe(constant);
  }
}

function getCopyKey(localFile: string, functionName: string): string {
  return `${localFile}#${functionName}`;
}

describe('private copies of official WGSL', () => {
  it('lists each copy once', () => {
    const keys = OFFICIAL_WGSL_COPIES.map(copy => getCopyKey(copy.localFile, copy.functionName));
    expect(new Set(keys).size).toBe(keys.length);
  });

  it('pins point-cloud normals to exponent-scaled division instead of the former official copy', () => {
    // Metal fast math turns value / scale into value * (1 / scale), whose reciprocal can
    // flush to zero for a large finite scale. The local function now delegates that quotient
    // to the shared exponent-scaled helper, so equality with the official text is incorrect.
    // Numeric oracles for large/small scales live in gpu-pointcloud/pointcloud-runtime-chunks.spec.ts.
    const localFile =
      'modules/analysis/src/pointcloud-analysis/operations/pointcloud-kernel-nodes.ts';
    const local = getFunctionText(localFile, 'safeNormal');
    expect(local).toBe(
      `fn safeNormal(value: vec3f) -> vec3f {
        let scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
        if (!(scale > 0.0) || scale > 3.402823e38) { return vec3f(0.0); }
        let scaled = getPointCloudQuotient3(value, scale);
        let magnitude = length(scaled);
        if (!(magnitude > 0.0)) { return vec3f(0.0); }
        return scaled / magnitude;
      }`.replace(/\s+/g, '')
    );
    const quotient = getFunctionText(localFile, 'getPointCloudQuotient3');
    for (const axis of ['x', 'y', 'z']) {
      expect(quotient).toContain(`getScaledQuotient(value.${axis},divisor,1.0,0)`);
    }
    expect(local).not.toBe(
      getFunctionText(
        'node_modules/@luma.gl/graph/dist/utils/gpu-spatial-operation.js',
        'safeNormal'
      )
    );
  });

  it.each([
    [
      'encounters',
      'gpu-spatial-analysis/trajectory-encounters/encounter-kernels.ts',
      'gpu-spatial-analysis/trajectory-encounters/encounter-grid-coordinate-pin.node.spec.ts'
    ],
    [
      'regions',
      'gpu-spatial-analysis/region-statistics/region-statistics-grid.ts',
      'gpu-spatial-analysis/region-statistics/region-statistics-grid-coordinate-pin.node.spec.ts'
    ],
    [
      'interpolation',
      'gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting.ts',
      'gpu-spatial-analysis/spatial-interpolation/grid-coordinate-pin.node.spec.ts'
    ],
    [
      'density',
      'gpu-spatial-analysis/point-density/point-density-kernels.ts',
      'gpu-spatial-analysis/point-density/point-density-grid-coordinate-pin.node.spec.ts'
    ],
    [
      'flow',
      'gpu-network/flow-aggregation/flow-aggregation-kernels.ts',
      'gpu-network/flow-aggregation/flow-aggregation-grid-coordinate-pin.node.spec.ts'
    ]
  ] as const)('%s uses the upstream grid export without a second getCoordinate and retains its standalone pin', (reader, localFile, pinSpec) => {
    const source = readRepositoryFile(`modules/analysis/src/${localFile}`);
    expect(source).toContain('${GPU_GRID_COORDINATE_WGSL}');
    expect(source).not.toMatch(/\bfn\s+getCoordinate\s*\(/);
    expect(readRepositoryFile(`modules/analysis/test/${pinSpec}`)).toContain(
      `testGridCoordinateReader('${reader}')`
    );
    expectGridCoordinateReaderSource(reader);
  });

  it.each(OFFICIAL_WGSL_COPIES)('$functionName in $localFile still exists and has a pin', copy => {
    expect(
      getFunctionText(copy.localFile, copy.functionName),
      `${copy.localFile} no longer holds fn ${copy.functionName}: remove the row here and in dev-docs/upstream-workarounds.md`
    ).toBeDefined();
    for (const officialFile of copy.officialFiles) {
      expect(
        getFunctionText(officialFile, copy.functionName),
        `${officialFile} no longer holds fn ${copy.functionName}: the official kernel moved, repoint the pin`
      ).toBeDefined();
    }
    const local = getFunctionText(copy.localFile, copy.functionName);
    for (const officialFile of copy.officialFiles) {
      expect(local, `${copy.localFile} against ${officialFile}`).toBe(
        getFunctionText(officialFile, copy.functionName)
      );
    }
    if (copy.pinSpec === THIS_SPEC) {
      return;
    }
    expect(
      existsSync(resolve(REPOSITORY_ROOT, copy.pinSpec)),
      `missing pin spec ${copy.pinSpec}`
    ).toBe(true);
    const pinSpecText = readRepositoryFile(copy.pinSpec);
    expect(pinSpecText, `${copy.pinSpec} must read the local copy`).toContain(copy.localFile);
    for (const officialFile of copy.officialFiles) {
      expect(pinSpecText, `${copy.pinSpec} must read the official kernel`).toContain(officialFile);
    }
    expect(pinSpecText, `${copy.pinSpec} must compare the two texts`).toMatch(/\.toBe\(/);
  });

  it('finds no copy of an official WGSL function that is not listed', () => {
    const officialNamesByText = new Map<string, string>();
    const officialNamesBySignatureAndBody = new Map<string, string>();
    for (const officialRoot of OFFICIAL_ROOTS) {
      const directory = resolve(REPOSITORY_ROOT, officialRoot);
      for (const filePath of listFiles(directory, /\.js$/)) {
        // The two bundles at the root of `dist` repeat every kernel.
        if (dirname(filePath) === directory) {
          continue;
        }
        const origin = relative(REPOSITORY_ROOT, filePath);
        for (const wgslFunction of getWGSLFunctionTexts(readFileSync(filePath, 'utf8'))) {
          if (wgslFunction.text.length < MINIMUM_COPY_LENGTH) {
            continue;
          }
          if (!officialNamesByText.has(wgslFunction.text)) {
            officialNamesByText.set(wgslFunction.text, `${origin} fn ${wgslFunction.name}`);
          }
          if (!officialNamesBySignatureAndBody.has(wgslFunction.signatureAndBody)) {
            officialNamesBySignatureAndBody.set(
              wgslFunction.signatureAndBody,
              `${origin} fn ${wgslFunction.name}`
            );
          }
        }
      }
    }
    // The scan has something to compare with: the published kernels are installed and readable.
    expect(officialNamesByText.size).toBeGreaterThan(1000);

    const listedKeys = new Set(
      OFFICIAL_WGSL_COPIES.map(copy => getCopyKey(copy.localFile, copy.functionName))
    );
    const unlistedCopies: string[] = [];
    const foundKeys = new Set<string>();
    for (const filePath of listFiles(resolve(REPOSITORY_ROOT, ANALYSIS_SOURCE), /\.ts$/)) {
      const localFile = relative(REPOSITORY_ROOT, filePath);
      for (const wgslFunction of getWGSLFunctionTexts(readFileSync(filePath, 'utf8'))) {
        const official =
          officialNamesByText.get(wgslFunction.text) ??
          officialNamesBySignatureAndBody.get(wgslFunction.signatureAndBody);
        // The cell-coordinate function is listed by name too, so a copy that has already drifted
        // from the official text cannot leave the list unnoticed.
        const isCopy =
          (official !== undefined && wgslFunction.text.length >= MINIMUM_COPY_LENGTH) ||
          wgslFunction.name === 'getCoordinate';
        if (!isCopy) {
          continue;
        }
        const key = getCopyKey(localFile, wgslFunction.name);
        foundKeys.add(key);
        if (!listedKeys.has(key)) {
          unlistedCopies.push(`${key} == ${official ?? 'a cell-coordinate function by name'}`);
        }
      }
    }
    expect(
      unlistedCopies,
      'add each to OFFICIAL_WGSL_COPIES with a pin spec, and to dev-docs/upstream-workarounds.md'
    ).toEqual([]);
    // Every listed copy is still an exact copy today (the per-copy pins say which official file).
    expect([...listedKeys].filter(key => !foundKeys.has(key))).toEqual([]);
  });
});

const DGGS_KERNEL = 'node_modules/@luma.gl/shaders/dist/modules/geospatial/dggs/dggs-wgsl.js';

type OfficialWGSLSemanticCopy = {
  functionName: string;
  localFile: string;
  /** The local function, whitespace removed, as reviewed. */
  localText: string;
  officialFunctionName: string;
  officialFile: string;
  /** The official function, whitespace removed, as reviewed against `localText`. */
  officialText: string;
  /** Unique active WGSL constant declarations each side relies on, whitespace removed. */
  localConstants: string[];
  officialConstants: string[];
  reason: string;
};

/**
 * Copies that compute the same f32 result as an official function with different text (renamed
 * constants, an equivalent exact rewrite), so the text scan above does not see them. Both texts are
 * pinned as reviewed: a luma.gl upgrade that changes the official function, or a local edit, fails
 * here and needs a new review of the arithmetic.
 */
const OFFICIAL_WGSL_SEMANTIC_COPIES: OfficialWGSLSemanticCopy[] = [
  'modules/analysis/src/gpu-spatial-analysis/cell-cover/cell-cover-kernels.ts',
  'modules/analysis/src/gpu-spatial-analysis/cell-cover/cell-cover-scanline.ts'
].map(localFile => ({
  functionName: 'coverTileLatitude',
  localFile,
  // `row * ROW_SCALE` with ROW_SCALE = 2^(1 - z) and `2.0 * tileY / tileScale` with
  // tileScale = 2^z are both exact in f32; PI and DGGS_PI round to the same f32; the degree factor is
  // the same literal. The result is bit-identical to the official function.
  localText:
    'fncoverTileLatitude(row:f32)->f32{lett=PI*(1.0-row*ROW_SCALE);return(2.0*atan(exp(t))-PI*0.5)*57.29577951308232;}',
  officialFunctionName: 'dggs_web_mercator_tile_y_to_latitude',
  officialFile: DGGS_KERNEL,
  officialText:
    'fndggs_web_mercator_tile_y_to_latitude(tileY:f32,tileScale:f32)->f32{letmercatorY=DGGS_PI*(1.0-2.0*tileY/tileScale);return(2.0*atan(exp(mercatorY))-DGGS_PI*0.5)*DGGS_RADIANS_TO_DEGREES;}',
  localConstants: [
    'constPI:f32=3.14159265358979;',
    'constROW_SCALE:f32=${getWGSLFloatLiteral(2**(1-resolution))};'
  ],
  officialConstants: [
    'constDGGS_PI:f32=3.141592653589793;',
    'constDGGS_RADIANS_TO_DEGREES:f32=57.29577951308232;'
  ],
  reason:
    'Quadbin cover must agree with the tile edges GPUCellGeometry draws with the official function; including the 70 KB dggs module in kernels that do not need it otherwise costs compile time.'
}));

describe('private semantic copies of official WGSL', () => {
  it.each(
    OFFICIAL_WGSL_SEMANTIC_COPIES
  )('$functionName in $localFile is pinned to $officialFunctionName', copy => {
    expectSemanticPin(
      readRepositoryFile(copy.localFile),
      copy.functionName,
      copy.localText,
      copy.localConstants,
      copy.localFile
    );
    expectSemanticPin(
      readRepositoryFile(copy.officialFile),
      copy.officialFunctionName,
      copy.officialText,
      copy.officialConstants,
      `${copy.officialFile}: review ${copy.localFile} again`
    );
    expect(Math.fround(3.14159265358979)).toBe(Math.fround(3.141592653589793));
  });
});

describe('WGSL pin extraction', () => {
  const functionSource = `fn coverTileLatitude(row: f32) -> f32 {
    let t = PI * (1.0 - row * ROW_SCALE);
    return (2.0 * atan(exp(t)) - PI * 0.5) * 57.29577951308232;
  }`;
  const constantSources = [
    'const PI: f32 = 3.14159265358979;',
    'const ROW_SCALE: f32 = ${getWGSLFloatLiteral(2 ** (1 - resolution))};'
  ];
  const source = [...constantSources, functionSource].join('\n');
  const copy = OFFICIAL_WGSL_SEMANTIC_COPIES[0];
  const checkPin = (candidate: string) =>
    expectSemanticPin(candidate, copy.functionName, copy.localText, copy.localConstants, 'fixture');
  const commentStyles = [
    {
      name: 'line',
      comment: (text: string) =>
        text
          .split('\n')
          .map(line => `// ${line}`)
          .join('\n')
    },
    {name: 'block', comment: (text: string) => `/* ${text} */`},
    {name: 'nested block', comment: (text: string) => `/* /* nested */ ${text} */`}
  ];

  it('accepts the reviewed active declarations with comments inside their bodies', () => {
    checkPin(source);
    checkPin(source.replace('let t', '/* } fn coverTileLatitude() { /* nested */ */ let t'));
  });

  it.each(
    commentStyles
  )('rejects an original function in a $name comment followed by changed code', ({comment}) => {
    const changedFunction = functionSource.replace('row * ROW_SCALE', 'row + ROW_SCALE');
    const candidate = [...constantSources, comment(functionSource), changedFunction].join('\n');
    expect(extractFunctionText(candidate, copy.functionName, 'fixture')).toBe(
      changedFunction.replace(/\s+/g, '')
    );
    expect(() => checkPin(candidate)).toThrow();
  });

  it.each(
    commentStyles
  )('rejects an original constant in a $name comment followed by a changed value', ({comment}) => {
    const candidate = source.replace(
      constantSources[0],
      `${comment(constantSources[0])}\nconst PI: f32 = 3.0;`
    );
    expect(extractConstantText(candidate, 'PI', 'fixture')).toBe('constPI:f32=3.0;');
    expect(() => checkPin(candidate)).toThrow();
  });

  it.each([
    functionSource,
    ...constantSources
  ])('rejects duplicate active declarations: %s', declaration => {
    expect(() => checkPin(`${source}\n${declaration}`)).toThrow(/exactly one active/);
  });

  it.each([
    functionSource,
    ...constantSources
  ])('rejects missing active declarations: %s', declaration => {
    expect(() => checkPin(source.replace(declaration, `/* ${declaration} */`))).toThrow(
      /exactly one active/
    );
  });
});
