// Census of contributor standards across modules/analysis/src.
// Usage: node census.mjs <srcRoot> <outJson>
import {readdirSync, readFileSync, statSync, writeFileSync} from 'node:fs';
import {join, relative} from 'node:path';

const [sourceRoot, outputPath] = process.argv.slice(2);

function walk(directory, files = []) {
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) walk(path, files);
    else if (name.endsWith('.ts')) files.push(path);
  }
  return files;
}

const files = walk(sourceRoot);
const rows = [];
const wgslFunctions = new Map();
const maxima = [];

for (const file of files) {
  const text = readFileSync(file, 'utf8');
  const path = relative(sourceRoot, file);
  const topDirectory = path.split('/')[0];
  const family = path.split('/').slice(0, 2).join('/');

  // WGSL function definitions (inside template strings; TS functions use `function`).
  for (const match of text.matchAll(/^\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/gm)) {
    const name = match[1];
    if (!wgslFunctions.has(name)) wgslFunctions.set(name, new Set());
    wgslFunctions.get(name).add(path);
  }
  for (const match of text.matchAll(
    /^(?:export )?const ((?:MAXIMUM|MAX|DEFAULT_MAXIMUM)_[A-Z0-9_]+)\s*=\s*([^;]+);/gm
  )) {
    maxima.push({path, name: match[1], value: match[2].trim().slice(0, 60)});
  }

  const classes = [...text.matchAll(/^export class ([A-Za-z0-9_]+)([^{]*)\{/gm)];
  const recipes = [...text.matchAll(/^export function (add[A-Za-z0-9_]+Recipe)/gm)];
  if (classes.length === 0 && recipes.length === 0) continue;

  const count = pattern => (text.match(pattern) || []).length;
  rows.push({
    path,
    topDirectory,
    family,
    lines: text.split('\n').length,
    classes: classes.map(match => match[1]),
    producer: classes.some(match => /GPUCommandNodeProducer/.test(match[2])),
    recipes: recipes.map(match => match[1]),
    hasSpdx: /SPDX-License-Identifier/.test(text),
    hasCopyrightText: /SPDX-FileCopyrightText/.test(text),
    spatialContextProp: /spatialContext\??:/.test(text),
    resolvesSpatialContext: /resolveGPUSpatialContext|resolve[A-Za-z]*SpatialContext\(/.test(text),
    takesPositions:
      /positions\??:|GPUPointGeometryPort|GPULineGeometryPort|GPUPolygonGeometryPort|GPUTrajectoryGeometryPort|GPUFloat32Positions/.test(
        text
      ),
    statusPortTypes: [
      ...new Set(
        [...text.matchAll(/\b(GPU[A-Za-z]*Status[A-Za-z]*)\b/g)]
          .map(match => match[1])
          .filter(name => /^GPU(Bounded|Iteration|Cardinality|Approximation|Recipe|Validity|Capacity)/.test(name))
      )
    ],
    statusWord: count(/\bstatus\b/gi),
    overflowWord: count(/overflow/gi),
    compactOutput: /GPUCompactOutput/.test(text),
    bareAssert: count(/\bassert\(/g),
    throwError: count(/throw new (Error|RangeError|TypeError)\(/g),
    validateCalls: count(/\bvalidate[A-Z][A-Za-z0-9]*\(/g),
    validateCapacity: count(/\bvalidate(Capacity|MaximumCount)\(/g),
    bindingGuard: count(/validateGPUStorageBindingByteLength|getStorageBindingElementLimit/g),
    packedViewOnly: count(/validatePackedView\(/g),
    kernelNodes: count(/createWGSLKernelNode\(/g),
    dispatchGate: count(/createWGSLDispatchGate\(/g),
    parameterBuffer: /GPUParameterBuffer/.test(text),
    perFrameDoc: /per-frame|Per-frame/.test(text),
    compileTimeDoc: /compile-time|Compile-time|topology prop/i.test(text),
    atomics: count(/atomic(Add|Max|Min|Store|Load|CompareExchangeWeak|Or|And)\(/g),
    forLoopsTs: count(/^\s*for \(/gm),
    readback: count(/readAsync|mapAsync|readBuffer|\.read\(\)/g),
    publishedPrimitives: [
      ...new Set(
        [...text.matchAll(/\b(GPUScan|GPUSort|GPUCompaction|GPUReduction|GPUGridIndex|GPUBVH|GPUHashIndex|GPUSegmentedSort|GPUBatchSort|GPURunLengthEncode|GPUSegmentedBVH|GPUCrossfilter|GPUHistogram|GPUSpMV|GPUAdaptiveSpMV|GPUGather|GPUScatter)\b/g)].map(
          match => match[1]
        )
      )
    ],
    maskProp: [
      ...new Set(
        [...text.matchAll(/^\s+((?:[a-z][A-Za-z]*)?(?:Mask|mask|Validity|validity|valid|Valid|nodata|noData|NoData)[A-Za-z]*)\??:/gm)].map(
          match => match[1]
        )
      )
    ],
    todo: count(/TODO|FIXME|HACK|XXX/g),
    notImplemented: count(/not (yet )?(implemented|supported)/gi),
    deprecated: count(/@deprecated/g),
    anyType: count(/\bas any\b|: any\b|<any>/g),
    tsIgnore: count(/@ts-(ignore|expect-error|nocheck)/g),
    float32Only: /float32x2/.test(text),
    unrolled: count(/for \(let [a-zA-Z]+ = 0; [a-zA-Z]+ < [^;]*(iteration|round|pass|sweep|step)[A-Za-z]*;/gi),
    serialKernel: count(/workgroupCount:\s*\[1,\s*1,\s*1\]|dispatch:\s*\[1,\s*1,\s*1\]|invocationCount:\s*1\b/g)
  });
}

const duplicateWgsl = [...wgslFunctions.entries()]
  .filter(([, paths]) => paths.size > 1)
  .map(([name, paths]) => ({name, fileCount: paths.size, files: [...paths].slice(0, 12)}))
  .sort((left, right) => right.fileCount - left.fileCount);

writeFileSync(
  outputPath,
  JSON.stringify({fileCount: files.length, rows, duplicateWgsl, wgslFunctionCount: wgslFunctions.size, maxima}, null, 1)
);
console.log(`files ${files.length} contributor files ${rows.length} wgsl fns ${wgslFunctions.size} duplicated ${duplicateWgsl.length} maxima ${maxima.length}`);
