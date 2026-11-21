// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// No install, registry access, build or lifecycle scripts: test precisely the packed bytes.
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {createRequire} from 'node:module';
import {basename, dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import MarkdownIt from 'markdown-it';
import semver from 'semver';
import typescript from 'typescript';
import {checkPackedArchiveFiles} from './packed-file-inventory.mjs';
import {getIsolatedNodeHeapArguments} from './verification-policy.mjs';

const WORKSPACE_DIRECTORY = fileURLToPath(new URL('../', import.meta.url));
// Qualification must not silently shrink when the candidate manifest drops a public family.
const EXPECTED_ANALYSIS_EXPORT_KEYS = [
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

function validatePublicEntryInventory(manifest) {
  for (const key of EXPECTED_ANALYSIS_EXPORT_KEYS) {
    if (!Object.hasOwn(manifest.exports ?? {}, key)) {
      throw new Error(`missing required public entry ${JSON.stringify(key)}`);
    }
  }
}

function getArchiveIdentity(bytes) {
  return {sha256: createHash('sha256').update(bytes).digest('hex'), byteLength: bytes.byteLength};
}

function verifyExtractedArchive(tarball, identity) {
  const actual = getArchiveIdentity(readFileSync(tarball));
  if (actual.sha256 !== identity.sha256 || actual.byteLength !== identity.byteLength)
    throw new Error(`retained tarball ${tarball} differs from the extracted tarball`);
}

function runCommand(command, arguments_, options) {
  const result = spawnSync(command, arguments_, {
    encoding: 'utf8',
    timeout: 60_000,
    maxBuffer: 16 * 1024 * 1024,
    ...options
  });
  if (result.error || result.status !== 0) {
    throw new Error(
      `${command} failed: ${result.error?.message ?? ''}\n${result.stdout ?? ''}${result.stderr ?? ''}`
    );
  }
  return result.stdout;
}

function isOutsidePackage(offset) {
  return offset === '..' || offset.startsWith(`..${sep}`) || isAbsolute(offset);
}

function resolvePackedFile(packageDirectory, target, description) {
  if (typeof target !== 'string' || !target.startsWith('./')) {
    throw new Error(`${description}: expected a relative package target, received ${target}`);
  }
  const path = resolve(packageDirectory, target);
  const offset = relative(packageDirectory, path);
  if (isOutsidePackage(offset)) {
    throw new Error(`${description}: target escapes tarball: ${target}`);
  }
  let actualPath;
  try {
    actualPath = realpathSync(path);
    if (!statSync(actualPath).isFile()) throw new Error('not a file');
  } catch {
    throw new Error(`${description}: missing packed file ${target}`);
  }
  const actualOffset = relative(realpathSync(packageDirectory), actualPath);
  if (isOutsidePackage(actualOffset)) {
    throw new Error(`${description}: target escapes tarball: ${target}`);
  }
  return actualPath;
}

function checkReadmeLinks(packageDirectory) {
  const readme = readFileSync(resolvePackedFile(packageDirectory, './README.md', 'README'), 'utf8');
  const markdown = new MarkdownIt({html: true});
  const destinations = [];
  function collectDestinations(tokens) {
    for (const token of tokens) {
      if (token.type === 'link_open') destinations.push(token.attrGet('href'));
      if (token.type === 'image') destinations.push(token.attrGet('src'));
      if (token.type === 'html_inline' || token.type === 'html_block') {
        for (const match of token.content.matchAll(
          /\b(?:href|src)\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s"'=<>`]+))/gi
        )) {
          destinations.push(markdown.utils.unescapeAll(match[1] ?? match[2] ?? match[3]));
        }
      }
      if (token.children) collectDestinations(token.children);
    }
  }
  collectDestinations(markdown.parse(readme, {}));
  for (const destination of destinations) {
    const href = destination;
    if (/^[a-z][a-z\d+.-]*:/i.test(href)) {
      new URL(href);
      continue;
    }
    const target = fileURLToPath(new URL(href, pathToFileURL(join(packageDirectory, 'README.md'))));
    const offset = relative(packageDirectory, target);
    if (isOutsidePackage(offset)) {
      throw new Error(`README link escapes tarball: ${href}`);
    }
    let actualPath;
    try {
      actualPath = realpathSync(target);
    } catch {
      throw new Error(`README link missing from tarball: ${href}`);
    }
    const actualOffset = relative(realpathSync(packageDirectory), actualPath);
    if (isOutsidePackage(actualOffset)) {
      throw new Error(`README link escapes tarball: ${href}`);
    }
  }
  process.stdout.write(`verified ${destinations.length} packed README links\n`);
}

function collectSourceFiles(directory) {
  return readdirSync(directory).flatMap(name => {
    const path = join(directory, name);
    return statSync(path).isDirectory() ? collectSourceFiles(path) : [path];
  });
}

function checkLicenseInventory(installedPackage, packageDirectory, manifest) {
  for (const file of [
    'LICENSE',
    'LICENSES/Apache-2.0.txt',
    'LICENSES/BSD-3-Clause.txt',
    'LICENSES/H3-NOTICE.txt'
  ]) {
    resolvePackedFile(installedPackage, `./${file}`, 'licence');
  }
  const inventory = readFileSync(
    resolvePackedFile(installedPackage, './THIRD-PARTY-NOTICES.md', 'notices'),
    'utf8'
  );
  const sourceDirectory = join(packageDirectory, 'src');
  const sources = collectSourceFiles(sourceDirectory).filter(
    source => !source.startsWith(`${join(sourceDirectory, 'testing')}${sep}`)
  );
  if (!sources.length) throw new Error('zero build source files checked for notices');
  for (const source of sources) {
    const text = readFileSync(source, 'utf8');
    const expression = text
      .match(/SPDX-License-Identifier: ([^\r\n]+)/)?.[1]
      ?.replace(/\s*(?:\*\/|-->)\s*$/, '')
      .trim();
    if (!expression && !/\.(?:[cm]?ts|tsx)$/.test(source)) continue;
    if (!expression || !inventory.includes(`\`${expression}\``)) {
      throw new Error(
        `unlisted SPDX expression ${expression}: ${relative(packageDirectory, source)}`
      );
    }
    for (const identifier of expression.split(/\s+(?:AND|OR)\s+/)) {
      if (!manifest.license?.split(/\s+(?:AND|OR)\s+/).includes(identifier)) {
        throw new Error(
          `manifest licence omits ${identifier}: ${relative(packageDirectory, source)}`
        );
      }
    }
    if (expression !== 'MIT') {
      const path = relative(packageDirectory, source).split('\\').join('/');
      const line = inventory.split('\n').find(candidate => candidate.includes(`\`${path}\``));
      if (!line?.includes(`\`${expression}\``)) throw new Error(`unlisted source notice: ${path}`);
      for (const holder of text.matchAll(/SPDX-FileCopyrightText: ([^\r\n]+)/g)) {
        if (!inventory.includes(holder[1]))
          throw new Error(`unlisted copyright holder: ${path}: ${holder[1]}`);
      }
    }
  }
  process.stdout.write(`verified packed notices against ${sources.length} build source files\n`);
}

function checkPackedEntryPointMaps(installedPackage, packageDirectory, manifest) {
  const entry = manifest.exports['.'];
  const sourcePath = join(packageDirectory, 'src/index.ts');
  const source = readFileSync(sourcePath, 'utf8');
  for (const target of [entry.import, entry.require]) {
    const mapTarget = `${target}.map`;
    const map = JSON.parse(
      readFileSync(resolvePackedFile(installedPackage, mapTarget, 'entry-point source map'), 'utf8')
    );
    const expectedSource = relative(dirname(resolve(packageDirectory, mapTarget)), sourcePath)
      .split(sep)
      .join('/');
    const sourceIndex = Array.isArray(map?.sources) ? map.sources.indexOf(expectedSource) : -1;
    if (
      map?.version !== 3 ||
      typeof map.mappings !== 'string' ||
      !map.mappings ||
      sourceIndex < 0 ||
      !Array.isArray(map.sourcesContent) ||
      map.sourcesContent[sourceIndex] !== source
    ) {
      throw new Error(`entry-point source map ${mapTarget} must embed src/index.ts`);
    }
  }
  process.stdout.write('verified packed ESM and CommonJS root source maps\n');
}

const NODE_NEXT_OPTIONS = {
  module: typescript.ModuleKind.NodeNext,
  moduleResolution: typescript.ModuleResolutionKind.NodeNext
};

const TYPESCRIPT_EXTENSIONS = typescript.getSupportedExtensions(NODE_NEXT_OPTIONS);

function isTypeScriptFile(file, host) {
  const canonicalFile = host.getCanonicalFileName(file);
  return TYPESCRIPT_EXTENSIONS.flat().some(extension => canonicalFile.endsWith(extension));
}

function isPackedTypeScriptFile(packageRoot, file, host) {
  if (!file || !isTypeScriptFile(file, host)) return false;
  try {
    const target = realpathSync(file);
    return statSync(target).isFile() && !isOutsidePackage(relative(packageRoot, target));
  } catch {
    return false;
  }
}

function resolveDeclarationPathReference(reference, containingFile, host) {
  const target = typescript.resolveTripleslashReference(reference.fileName, containingFile);
  // Match the compiler's getSourceFileFromReferenceWorker: an explicit extension must be
  // supported; extensionless references try the first supported extension group in order.
  const candidates = basename(target).includes('.')
    ? [target]
    : TYPESCRIPT_EXTENSIONS[0].map(extension => `${target}${extension}`);
  return candidates.find(
    candidate => isTypeScriptFile(candidate, host) && host.fileExists(candidate)
  );
}

/** Relative module specifiers of one declaration file, with their string-literal nodes. */
function collectRelativeDeclarationSpecifiers(sourceFile) {
  const literals = [];
  function visit(node) {
    if (
      (typescript.isImportDeclaration(node) || typescript.isExportDeclaration(node)) &&
      node.moduleSpecifier &&
      typescript.isStringLiteral(node.moduleSpecifier)
    ) {
      literals.push(node.moduleSpecifier);
    } else if (
      typescript.isImportTypeNode(node) &&
      typescript.isLiteralTypeNode(node.argument) &&
      typescript.isStringLiteral(node.argument.literal)
    ) {
      literals.push(node.argument.literal);
    } else if (
      typescript.isExternalModuleReference(node) &&
      typescript.isStringLiteral(node.expression)
    ) {
      literals.push(node.expression);
    } else if (
      typescript.isModuleDeclaration(node) &&
      typescript.isStringLiteral(node.name) &&
      node.name.text.startsWith('.')
    ) {
      literals.push(node.name);
    }
    typescript.forEachChild(node, visit);
  }
  visit(sourceFile);
  return literals.filter(literal => literal.text.startsWith('.'));
}

/**
 * Resolves every relative module reference of every packed declaration file with native
 * NodeNext rules, in the resolution mode TypeScript uses for that reference. This is independent
 * of the consumer examples: one unresolved `import("../x")` type degrades a public type to `any`
 * without a diagnostic under `skipLibCheck`.
 */
export function checkDeclarationEdges(
  packageDirectory,
  consumerDirectory = packageDirectory,
  system = typescript.sys
) {
  const packageRoot = realpathSync(packageDirectory);
  const consumerRoot = realpathSync(consumerDirectory);
  const host = {
    ...system,
    getCurrentDirectory: () => consumerRoot,
    getCanonicalFileName: typescript.createGetCanonicalFileName(system.useCaseSensitiveFileNames)
  };
  const declarations = collectSourceFiles(packageRoot).filter(path =>
    /\.d\.[cm]?ts$/.test(host.getCanonicalFileName(path))
  );
  if (!declarations.length) throw new Error('zero packed declaration files checked');
  const failures = [];
  let edges = 0;
  for (const path of declarations) {
    const sourceFile = typescript.createSourceFile(
      path,
      readFileSync(path, 'utf8'),
      {
        languageVersion: typescript.ScriptTarget.ES2022,
        impliedNodeFormat: typescript.getImpliedNodeFormatForFile(
          path,
          undefined,
          host,
          NODE_NEXT_OPTIONS
        )
      },
      true
    );
    const location = literal => {
      const {line} = sourceFile.getLineAndCharacterOfPosition(literal.getStart(sourceFile));
      return `${relative(packageRoot, path).split(sep).join('/')}:${line + 1}`;
    };
    for (const literal of collectRelativeDeclarationSpecifiers(sourceFile)) {
      edges++;
      const mode = typescript.getModeForUsageLocation(sourceFile, literal, NODE_NEXT_OPTIONS);
      const resolved = typescript.resolveModuleName(
        literal.text,
        path,
        NODE_NEXT_OPTIONS,
        host,
        undefined,
        undefined,
        mode
      ).resolvedModule?.resolvedFileName;
      if (!isPackedTypeScriptFile(packageRoot, resolved, host)) {
        failures.push(`${location(literal)} ${JSON.stringify(literal.text)}`);
      }
    }
    for (const reference of sourceFile.referencedFiles) {
      edges++;
      const target = resolveDeclarationPathReference(reference, path, host);
      if (!isPackedTypeScriptFile(packageRoot, target, host)) {
        const {line} = sourceFile.getLineAndCharacterOfPosition(reference.pos);
        failures.push(
          `${relative(packageRoot, path).split(sep).join('/')}:${line + 1} reference ${JSON.stringify(reference.fileName)}`
        );
      }
    }
    for (const reference of sourceFile.typeReferenceDirectives) {
      if (!reference.fileName.startsWith('.')) continue;
      edges++;
      const mode = typescript.getModeForFileReference(reference, sourceFile.impliedNodeFormat);
      const resolved = typescript.resolveTypeReferenceDirective(
        reference.fileName,
        path,
        NODE_NEXT_OPTIONS,
        host,
        undefined,
        undefined,
        mode
      ).resolvedTypeReferenceDirective?.resolvedFileName;
      if (!isPackedTypeScriptFile(packageRoot, resolved, host)) {
        const {line} = sourceFile.getLineAndCharacterOfPosition(reference.pos);
        failures.push(
          `${relative(packageRoot, path).split(sep).join('/')}:${line + 1} reference types ${JSON.stringify(reference.fileName)}`
        );
      }
    }
  }
  if (failures.length) {
    throw new Error(
      `${failures.length} of ${edges} relative declaration references do not resolve under NodeNext in ${new Set(failures.map(failure => failure.split(':')[0])).size} files:\n${failures.join('\n')}`
    );
  }
  process.stdout.write(
    `resolved ${edges} relative references in ${declarations.length} packed declaration files\n`
  );
  return {declarationCount: declarations.length, edgeCount: edges};
}

function checkPackedDeclarations(consumerDirectory, manifest) {
  validatePublicEntryInventory(manifest);
  const entries = Object.keys(manifest.exports ?? {});
  if (!entries.length || entries.some(key => key.includes('*'))) {
    throw new Error('zero exports or unsupported export patterns in the native type consumer');
  }
  const imports = entries.map((key, index) => {
    const specifier = key === '.' ? manifest.name : `${manifest.name}/${key.slice(2)}`;
    return `import * as entry${index} from ${JSON.stringify(specifier)}; void entry${index};`;
  });
  // The same entries from an ES module and from a CommonJS module: `require` consumers resolve
  // their types through the `require` condition and must type-check on their own.
  const consumerPaths = ['mts', 'cts'].map(extension => {
    const source = readFileSync(new URL(`./packed-consumer.${extension}`, import.meta.url), 'utf8');
    const consumerPath = join(consumerDirectory, `native-consumer.${extension}`);
    writeFileSync(consumerPath, `${source}\n${imports.join('\n')}\n`);
    return consumerPath;
  });
  const options = {
    module: typescript.ModuleKind.NodeNext,
    moduleResolution: typescript.ModuleResolutionKind.NodeNext,
    target: typescript.ScriptTarget.ES2022,
    skipLibCheck: true,
    types: [],
    strict: true,
    noEmit: true
  };
  const host = typescript.createCompilerHost(options);
  host.getCurrentDirectory = () => consumerDirectory;
  const program = typescript.createProgram(consumerPaths, options, host);
  const diagnostics = typescript.getPreEmitDiagnostics(program);
  if (diagnostics.length) {
    throw new Error(
      `native declaration consumer failed:\n${typescript.formatDiagnostics(diagnostics, {
        getCanonicalFileName: file => file,
        getCurrentDirectory: () => consumerDirectory,
        getNewLine: () => '\n'
      })}`
    );
  }
  process.stdout.write('native NodeNext declaration consumer passed (.mts and .cts)\n');
}

async function checkExportEntries(packageDirectory, manifest) {
  validatePublicEntryInventory(manifest);
  const entries = Object.entries(manifest.exports ?? {});
  if (!entries.length) throw new Error('zero exports checked');
  const require = createRequire(import.meta.url);
  const failures = [];
  let checked = 0;
  for (const [key, target] of entries) {
    const specifier = key === '.' ? manifest.name : `${manifest.name}/${key.slice(2)}`;
    try {
      if (
        (key !== '.' && !key.startsWith('./')) ||
        key.includes('*') ||
        !target ||
        typeof target !== 'object'
      ) {
        throw new Error(
          `unsupported export ${key}; declare explicit types, import and require targets`
        );
      }
      resolvePackedFile(packageDirectory, target.types, `${specifier} types target`);
      resolvePackedFile(packageDirectory, target.import, `${specifier} import target`);
      resolvePackedFile(packageDirectory, target.require, `${specifier} require target`);
      const imported = await import(specifier);
      const required = require(specifier);
      const importedNames = Object.keys(imported).sort();
      const requiredNames = Object.keys(required).sort();
      if (!importedNames.length) throw new Error('zero exported names');
      if (JSON.stringify(importedNames) !== JSON.stringify(requiredNames)) {
        throw new Error(
          `export names differ: ESM ${importedNames.join(', ')}; CommonJS ${requiredNames.join(', ')}`
        );
      }
      checked++;
    } catch (error) {
      failures.push(`${specifier}: ${error.message}`);
    }
  }
  if (!checked || failures.length)
    throw new Error(
      `verified ${checked} of ${entries.length} packed entry points\n${failures.join('\n')}`
    );
  process.stdout.write(`verified ${checked} packed entry points (ESM, CommonJS, types)\n`);
}

async function runDocumentedComputation(manifest) {
  // docs/api/operation-forms-and-cpu-twins.md, "Variogram fitting without a graph".
  const {computeVariogramModelFitOnCPU} = await import(`${manifest.name}/dataframe-analysis`);
  const distances = Float32Array.from({length: 20}, (_, index) => (index + 1) * 0.2);
  const result = computeVariogramModelFitOnCPU(
    {
      distances,
      semivariances: Float32Array.from(
        distances,
        distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
      ),
      pairCounts: Uint32Array.from({length: 20}, (_, index) => 10 + index)
    },
    {models: ['gaussian', 'exponential', 'spherical'], weighting: 'pairs'}
  );
  if (result.parameters.length !== 12 || result.status.length !== 3) {
    throw new Error(
      `CPU variogram result lengths: expected parameters 12 and status 3, received ${result.parameters.length} and ${result.status.length}`
    );
  }
  if (result.status[1] !== 0)
    throw new Error(`CPU variogram status: expected 0, received ${result.status[1]}`);
  const expected = [0.2, 1.5, 2.2, 0];
  for (let index = 0; index < expected.length; index++) {
    const actual = result.parameters[4 + index];
    if (!Number.isFinite(actual) || Math.abs(actual - expected[index]) > 1e-4) {
      throw new Error(
        `CPU variogram parameter ${index}: expected ${expected[index]}, received ${actual}`
      );
    }
  }
  process.stdout.write('documented CPU variogram fit passed\n');
}

async function checkCrossFormatContracts(manifest) {
  const {default: assert} = await import('node:assert/strict');
  const {Buffer} = await import('@luma.gl/core');
  const require = createRequire(import.meta.url);
  const imported = await import(`${manifest.name}/analysis-core`);
  const required = require(`${manifest.name}/analysis-core`);
  const importedRecords = await import(`${manifest.name}/computation-record`);
  const requiredRecords = require(`${manifest.name}/computation-record`);
  const importedSpatial = await import(`${manifest.name}/gpu-spatial-analysis`);
  const requiredSpatial = require(`${manifest.name}/gpu-spatial-analysis`);
  const importedGraph = await import('@luma.gl/graph/gpu-core');
  const requiredGraph = require('@luma.gl/graph/gpu-core');
  const importedData = await import('@luma.gl/graph/gpu-data');
  const requiredData = require('@luma.gl/graph/gpu-data');
  const importedDevice = await import('@luma.gl/core/test-utils');
  const requiredDevice = require('@luma.gl/core/test-utils');
  for (const [label, producer, reader, records, spatial, graphEntry, dataEntry, deviceEntry] of [
    [
      'import-to-require',
      imported,
      required,
      requiredRecords,
      requiredSpatial,
      importedGraph,
      importedData,
      importedDevice
    ],
    [
      'require-to-import',
      required,
      imported,
      importedRecords,
      importedSpatial,
      requiredGraph,
      requiredData,
      requiredDevice
    ]
  ]) {
    const device = new deviceEntry.NullDevice({});
    // The null device supports declaration and borrowed buffers; no pipeline or GPU execution.
    Object.defineProperty(device, 'type', {value: 'webgpu'});
    Object.defineProperty(device, 'limits', {
      value: {
        ...device.limits,
        maxComputeWorkgroupsPerDimension: 65535,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxStorageBuffersPerShaderStage: 8,
        maxStorageBufferBindingSize: 134217728
      }
    });
    const buffers = [];
    try {
      const graph = new graphEntry.GPUCommandGraph(device);
      const chunks = [3, 7].map(value => {
        const buffer = device.createBuffer({
          data: Float32Array.of(value),
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return new dataEntry.GPUData({buffer, format: 'float32', length: 1});
      });
      const vector = graph.importGPUVector(
        'two',
        new dataEntry.GPUVector({type: 'data', name: 'two', format: 'float32', data: chunks})
      );
      assert.deepEqual(
        reader.getGraphViewChunks(vector).map(chunk => chunk.buffer?.id),
        ['two-chunk-0', 'two-chunk-1'],
        `${label}: ordered chunks`
      );
      assert.equal(reader.getGraphViewChunks(vector)[0], vector.data[0]);
      reader.validateGraphViewsBelongToGraph(label, graph, [vector]);
      const empty = graph.importGPUVector(
        'empty',
        new dataEntry.GPUVector({type: 'data', name: 'empty', format: 'float32', data: []})
      );
      assert.equal(reader.getGraphViewChunks(empty).length, 0, `${label}: empty vector`);
      reader.validateGraphViewsBelongToGraph(label, graph, [empty]);

      function importPackedVector(id, format, data) {
        const buffer = device.createBuffer({
          data,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return graph.importGPUVector(
          id,
          new dataEntry.GPUVector({
            type: 'data',
            name: id,
            format,
            data: [new dataEntry.GPUData({buffer, format, length: data.length})]
          })
        );
      }
      const importPackedView = (id, format, data) => importPackedVector(id, format, data).data[0];
      const featureRows = importPackedView('feature-rows', 'uint32', Uint32Array.of(0, 0));
      const sums = importPackedView('sums', 'float32', Float32Array.of(0));
      const zonal = new spatial.GPUZonalStatistics({
        id: `zonal-${label}`,
        features: {kind: 'feature-rows', pointFeatureRows: featureRows, featureCount: 1},
        values: vector,
        output: {sums}
      });
      assert.equal(zonal.sumOrder, 'atomic', `${label}: chunked zonal sumOrder`);
      let aggregationNodes;
      assert.doesNotThrow(() => {
        aggregationNodes = zonal.getCommandNodes(graph);
      }, `${label}: zonal aggregation declaration`);
      assert.ok(aggregationNodes.length > 0, `${label}: zonal aggregation nodes`);
      const getResourceBuffer = resource => resource.buffer?.buffer ?? resource.buffer;
      for (const [chunkIndex, chunk] of vector.data.entries()) {
        assert.ok(
          aggregationNodes.some(
            node =>
              node.type === 'compute' &&
              node.resources?.some(
                resource =>
                  getResourceBuffer(resource) === chunk.buffer &&
                  ['storage-read', 'storage-read-write'].includes(resource.usage)
              ) &&
              node.resources?.some(
                resource =>
                  getResourceBuffer(resource) === sums.buffer &&
                  resource.usage === 'storage-read-write'
              )
          ),
          `${label}: zonal accumulation reads chunk ${chunkIndex} and updates sums`
        );
      }

      const recorder = new records.GPUComputationRecorder();
      assert.equal(recorder.recordKernels(graph), graph);
      const body = 'let packedConsumerMarker = 17u;';
      const node = producer.createWGSLKernelNode(graph, {
        id: `packed-${label}`,
        operation: 'PackedConsumer',
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [],
        body
      });
      const record = reader.getWGSLKernelNodeRecord(graph, node.id);
      assert.ok(record, `${label}: opposite-format kernel was not recorded`);
      assert.equal(record.id, node.id);
      assert.equal(record.id, `packed-${label}`);
      assert.equal(record.body, body);
      assert.ok(record.source.includes(body), `${label}: recorded WGSL body`);
      assert.match(record.source, /@compute/);
      assert.match(record.source, /fn main\(/);
      assert.equal(record.invocationCount, 1);
      assert.deepEqual(record.workgroupSize, [2, 1, 1]);
      assert.deepEqual(record.workgroups, [1, 1, 1]);

      const selectedInput = importPackedVector('selected-input', 'float32', Float32Array.of(3));
      const selectedOutput = importPackedVector('selected-output', 'uint32', Uint32Array.of(0));
      const contributor = {getCommandNodes: () => []};
      let factoryCalls = 0;
      const operation = new producer.GPUContributorOperation({
        id: `operation-${label}`,
        inputs: {selectedInput: 'input-value'},
        outputs: {selectedOutput: {value: 'output-value', format: 'uint32', length: 1}},
        createContributor: getView => {
          factoryCalls++;
          const input = getView('selectedInput', 'float32');
          const output = getView('selectedOutput', 'uint32');
          assert.equal(input.format, 'float32', `${label}: selected input format`);
          assert.equal(output.format, 'uint32', `${label}: selected output format`);
          assert.equal(input, selectedInput.data[0], `${label}: selected input identity`);
          assert.equal(output, selectedOutput.data[0], `${label}: selected output identity`);
          return contributor;
        }
      });
      const lowered = [];
      const resolvedValues = [];
      const context = {
        lower: value => lowered.push(value),
        resolveVector: value => {
          resolvedValues.push(value);
          assert.ok(
            ['input-value', 'output-value'].includes(value),
            `${label}: selected graph value`
          );
          return value === 'input-value' ? selectedInput : selectedOutput;
        }
      };
      reader.GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(operation, context);
      assert.equal(factoryCalls, 1, `${label}: genuine operation reaches its factory`);
      assert.deepEqual(
        [...new Set(resolvedValues)].sort(),
        ['input-value', 'output-value'],
        `${label}: selected port resolution`
      );
      assert.deepEqual(lowered, [contributor]);
      assert.equal(lowered[0], contributor, `${label}: lower receives the operation's contributor`);
      assert.throws(
        () =>
          reader.GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(
            {
              id: 'type-string-impostor',
              type: operation.type,
              inputs: {},
              outputs: {},
              createContributor: () => contributor,
              destroy() {}
            },
            context
          ),
        /not a GPUContributorOperation/
      );
      assert.equal(lowered.length, 1, `${label}: impostor must not reach lower`);
      operation.destroy();
      process.stdout.write(`native cross-format ${label} passed\n`);
    } finally {
      for (const buffer of buffers) buffer.destroy();
      device.destroy();
    }
  }
  process.stdout.write(
    'native cross-format vectors, recording and lowering passed (both directions)\n'
  );
}

function main() {
  const heapArguments = getIsolatedNodeHeapArguments(
    [
      process.env.NODE_OPTIONS ?? '',
      ...process.execArgv.map(argument => JSON.stringify(argument))
    ].join(' ')
  );
  let packageDirectory = join(WORKSPACE_DIRECTORY, 'modules/analysis');
  let dependencyDirectory = join(WORKSPACE_DIRECTORY, 'node_modules');
  let keepArtifact = false;
  for (let index = 2; index < process.argv.length; index++) {
    const argument = process.argv[index];
    if (argument === '--keep-artifact') keepArtifact = true;
    else if (argument === '--package-directory' || argument === '--dependency-directory') {
      const value = process.argv[++index];
      if (!value || value.startsWith('--')) throw new Error(`${argument} requires a directory`);
      if (argument === '--package-directory') packageDirectory = resolve(value);
      else dependencyDirectory = resolve(value);
    } else throw new Error(`unknown argument ${argument}`);
  }
  const temporaryRoot = realpathSync(tmpdir());
  const temporaryOffset = relative(realpathSync(WORKSPACE_DIRECTORY), temporaryRoot);
  if (
    !temporaryOffset ||
    (temporaryOffset.split(sep)[0] !== '..' && !isAbsolute(temporaryOffset))
  ) {
    throw new Error('temporary directory must be outside the workspace');
  }
  const temporaryDirectory = mkdtempSync(join(temporaryRoot, 'analysis-packed-consumer-'));
  try {
    const manifest = JSON.parse(readFileSync(join(packageDirectory, 'package.json'), 'utf8'));
    if (!/^(@[^/]+\/)?[^/]+$/.test(manifest.name) || manifest.name.includes('..'))
      throw new Error('invalid package name');
    const packed = JSON.parse(
      runCommand(
        'npm',
        [
          'pack',
          '--offline',
          '--ignore-scripts',
          '--json',
          '--workspaces=false',
          '--pack-destination',
          temporaryDirectory,
          '--cache',
          join(temporaryDirectory, 'npm-cache')
        ],
        {cwd: packageDirectory}
      )
    );
    if (
      packed.length !== 1 ||
      !packed[0].filename ||
      basename(packed[0].filename) !== packed[0].filename
    ) {
      throw new Error('npm pack did not produce exactly one tarball');
    }
    const tarball = join(temporaryDirectory, packed[0].filename);
    const consumerDirectory = join(temporaryDirectory, 'consumer');
    const installedPackage = join(consumerDirectory, 'node_modules', manifest.name);
    mkdirSync(installedPackage, {recursive: true});
    // Hash and extract the same immutable buffer, rather than reopening a replaceable pathname.
    const archiveBytes = readFileSync(tarball);
    const archiveIdentity = getArchiveIdentity(archiveBytes);
    checkPackedArchiveFiles(archiveBytes);
    runCommand('tar', ['-xzf', '-', '--strip-components=1', '-C', installedPackage], {
      input: archiveBytes
    });
    verifyExtractedArchive(tarball, archiveIdentity);
    const installedManifest = JSON.parse(
      readFileSync(join(installedPackage, 'package.json'), 'utf8')
    );
    checkReadmeLinks(installedPackage);
    checkLicenseInventory(installedPackage, packageDirectory, installedManifest);
    checkDeclarationEdges(installedPackage, consumerDirectory);
    const linkedDependencies = new Set();
    for (const [name, range] of [
      ...Object.entries(installedManifest.dependencies ?? {}),
      ...Object.entries(installedManifest.peerDependencies ?? {})
    ]) {
      if (!/^(@[^/]+\/)?[^/]+$/.test(name) || name.includes('..'))
        throw new Error(`invalid dependency name ${name}`);
      let installed;
      try {
        installed = realpathSync(join(dependencyDirectory, name));
        if (!statSync(installed).isDirectory()) throw new Error('not a directory');
      } catch {
        throw new Error(`missing installed dependency ${name} in ${dependencyDirectory}`);
      }
      const dependency = JSON.parse(readFileSync(join(installed, 'package.json'), 'utf8'));
      if (
        !semver.validRange(range) ||
        !semver.valid(dependency.version) ||
        !semver.satisfies(dependency.version, range)
      ) {
        throw new Error(`${name}@${dependency.version} does not satisfy ${range}`);
      }
      if (linkedDependencies.has(name)) continue;
      const destination = join(consumerDirectory, 'node_modules', name);
      mkdirSync(dirname(destination), {recursive: true});
      symlinkSync(installed, destination, 'dir');
      linkedDependencies.add(name);
    }
    const worker = join(consumerDirectory, 'check-consumer.mjs');
    writeFileSync(
      worker,
      `
      import {readFileSync, realpathSync, statSync} from 'node:fs';
      import {createRequire} from 'node:module';
      import {isAbsolute, join, relative, resolve, sep} from 'node:path';
      ${isOutsidePackage.toString()}
      ${resolvePackedFile.toString()}
      const EXPECTED_ANALYSIS_EXPORT_KEYS = ${JSON.stringify(EXPECTED_ANALYSIS_EXPORT_KEYS)};
      ${validatePublicEntryInventory.toString()}
      ${checkExportEntries.toString()}
      ${runDocumentedComputation.toString()}
      ${checkCrossFormatContracts.toString()}
      const packageDirectory = process.argv[2];
      const manifest = JSON.parse(readFileSync(join(packageDirectory, 'package.json'), 'utf8'));
      await checkExportEntries(packageDirectory, manifest);
      await runDocumentedComputation(manifest);
      await checkCrossFormatContracts(manifest);
    `
    );
    process.stdout.write(
      runCommand(process.execPath, [...heapArguments, worker, installedPackage], {
        cwd: consumerDirectory,
        env: {...process.env, NODE_PATH: '', NODE_OPTIONS: ''}
      })
    );
    checkPackedDeclarations(consumerDirectory, installedManifest);
    checkPackedEntryPointMaps(installedPackage, packageDirectory, installedManifest);
    verifyExtractedArchive(tarball, archiveIdentity);
    if (keepArtifact)
      process.stdout.write(
        `checked tarball: ${tarball}\nchecked tarball identity: ${JSON.stringify(archiveIdentity)}\n`
      );
  } finally {
    if (!keepArtifact) rmSync(temporaryDirectory, {recursive: true, force: true});
    else process.stdout.write(`retained consumer directory: ${temporaryDirectory}\n`);
  }
}

if (
  process.argv[1] &&
  realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url))
) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`packed-consumer check failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}
