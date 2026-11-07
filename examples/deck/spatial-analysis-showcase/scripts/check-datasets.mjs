// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Verify shipped dataset contracts without downloading or executing dataset modules.
// A clean result verifies bytes and metadata, not the scientific validity of a source.
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {PNG} from 'pngjs';
import ts from 'typescript';

const SHOWCASE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DATA_ROOT = path.join(SHOWCASE_ROOT, 'public/data');
const CATALOG_ROOT = path.join(SHOWCASE_ROOT, 'src/data/datasets');
const ELEMENT_BYTES = {float64: 8, float32: 4, int32: 4, uint32: 4, int16: 2, uint16: 2, int8: 1, uint8: 1};
const reports = [];

for (const filename of fs.readdirSync(CATALOG_ROOT).filter(name => name.endsWith('.dataset.ts')).sort()) {
  const identifier = filename.replace('.dataset.ts', '');
  const report = {id: identifier, bundled: false, files: 0, bytes: 0, errors: [], notes: []};
  reports.push(report);
  const catalog = readCatalog(path.join(CATALOG_ROOT, filename));
  if (catalog.id !== identifier) report.errors.push('Catalog id does not match its filename.');
  for (const field of ['title', 'description', 'license', 'attribution', 'sourceUrl']) {
    if (typeof catalog[field] !== 'string' || !catalog[field].trim()) report.errors.push(`Missing catalog ${field}.`);
  }
  if (catalog.sourceUrl && !/^https?:\/\//.test(catalog.sourceUrl)) report.errors.push('Source URL must use HTTP(S).');
  const datasetDirectory = path.join(DATA_ROOT, identifier);
  const manifestPath = path.join(datasetDirectory, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    report.notes.push('No bundled manifest; remote or builtin loader requires separate verification.');
    continue;
  }
  report.bundled = true;
  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    if (manifest.id !== identifier) report.errors.push('Manifest id does not match catalog.');
    if (!isBounds(manifest.bbox)) report.errors.push('Manifest bbox must contain finite geographic bounds.');
    const checkedFiles = new Map();
    const readAsset = relativePath => {
      if (typeof relativePath !== 'string' || !relativePath) throw new Error('Missing asset filename.');
      const assetPath = path.resolve(datasetDirectory, relativePath);
      if (!assetPath.startsWith(`${datasetDirectory}${path.sep}`)) throw new Error(`Asset escapes its dataset: ${relativePath}`);
      if (checkedFiles.has(relativePath)) return checkedFiles.get(relativePath);
      const bytes = fs.readFileSync(assetPath);
      checkedFiles.set(relativePath, bytes);
      report.files++;
      report.bytes += bytes.length;
      return bytes;
    };
    for (const [name, column] of Object.entries(manifest.columns ?? {})) {
      const bytes = readAsset(column.file);
      const components = column.components ?? 1;
      if (!(column.dtype in ELEMENT_BYTES) || !Number.isSafeInteger(column.length) || column.length < 0 || !Number.isSafeInteger(components) || components < 1) {
        report.errors.push(`Column ${name}: invalid dtype, length or components.`);
        continue;
      }
      const expectedBytes = column.length * components * ELEMENT_BYTES[column.dtype];
      if (bytes.length !== expectedBytes) report.errors.push(`Column ${name}: expected ${expectedBytes} bytes, found ${bytes.length}.`);
    }
    const rasters = {...manifest.rasters};
    if (manifest.raster) rasters['primary'] = manifest.raster;
    for (const [name, raster] of Object.entries(rasters)) {
      const bytes = readAsset(raster.file);
      if (![raster.width, raster.height, raster.depth ?? 1].every(value => Number.isSafeInteger(value) && value > 0)) {
        report.errors.push(`Raster ${name}: invalid dimensions.`);
        continue;
      }
      if (!isBounds(raster.bounds)) report.errors.push(`Raster ${name}: missing geographic bounds.`);
      if (raster.encoding?.endsWith('-bin')) {
        const elementBytes = ELEMENT_BYTES[raster.encoding.slice(0, -4)];
        const expectedBytes = raster.width * raster.height * (raster.depth ?? 1) * elementBytes;
        if (bytes.length !== expectedBytes) report.errors.push(`Raster ${name}: expected ${expectedBytes} bytes, found ${bytes.length}.`);
      } else if (raster.file.endsWith('.png')) {
        const decoded = PNG.sync.read(bytes);
        if (decoded.width !== raster.width || decoded.height !== raster.height) report.errors.push(`Raster ${name}: PNG dimensions differ from its manifest.`);
      }
    }
    for (const entry of [manifest.geometry, manifest.table]) if (entry?.file) readAsset(entry.file);
    for (const field of ['attributesCsv', 'namesCsv']) if (typeof manifest[field] === 'string') readAsset(manifest[field]);
    // New datasets may declare artifact hashes; source hashes are deliberately not
    // compared with derived runtime bytes, which are a different artifact.
    const checksums = manifest.checksums ?? manifest.properties?.checksums;
    if (checksums) {
      for (const [filename, expected] of Object.entries(checksums)) {
        if (typeof expected !== 'string' || !/^[a-f0-9]{64}$/i.test(expected)) {
          report.errors.push(`Unsupported SHA-256 checksum for ${filename}.`);
          continue;
        }
        const actual = createHash('sha256').update(readAsset(filename)).digest('hex');
        if (actual !== expected.toLowerCase()) report.errors.push(`SHA-256 mismatch: ${filename}.`);
      }
    } else {
      report.notes.push('No runtime artifact checksums declared.');
    }
    const builderDirectory = path.join(SHOWCASE_ROOT, 'scripts/data', identifier);
    if (!fs.existsSync(builderDirectory)) report.notes.push('No dedicated builder directory; inspect shared acquisition pipeline.');
    if (!manifest.properties?.source && !manifest.properties?.provenance && !manifest.source && !manifest.provenance) {
      report.notes.push('Source is catalog-only; inspect acquisition provenance before interpreting values.');
    }
  } catch (error) {
    report.errors.push(error.message);
  }
}

const catalogIdentifiers = new Set(reports.map(report => report.id));
for (const entry of fs.readdirSync(DATA_ROOT, {withFileTypes: true})) {
  if (entry.isDirectory() && fs.existsSync(path.join(DATA_ROOT, entry.name, 'manifest.json')) && !catalogIdentifiers.has(entry.name)) {
    reports.push({id: entry.name, bundled: true, files: 0, bytes: 0, errors: ['Bundled manifest has no catalog entry.'], notes: []});
  }
}
const errors = reports.flatMap(report => report.errors.map(message => `${report.id}: ${message}`));
const summary = {
  datasets: reports.length,
  bundled: reports.filter(report => report.bundled).length,
  verifiedFiles: reports.reduce((total, report) => total + report.files, 0),
  verifiedBytes: reports.reduce((total, report) => total + report.bytes, 0),
  errors: errors.length,
  provenanceNotes: reports.reduce((total, report) => total + report.notes.length, 0)
};
if (process.argv.includes('--json')) console.log(JSON.stringify({summary, datasets: reports}, null, 2));
else {
  console.log(`Dataset integrity: ${summary.bundled}/${summary.datasets} bundled datasets, ${summary.verifiedFiles} assets, ${summary.errors} errors.`);
  for (const error of errors) console.error(`  ${error}`);
  console.log(`${summary.provenanceNotes} provenance/remote-loader notes; inspect --json for details. Byte checks do not establish scientific validity.`);
}
process.exitCode = errors.length ? 1 : 0;

function isBounds(value) {
  return Array.isArray(value) && value.length === 4 && value.every(Number.isFinite) && value[0] >= -180 && value[2] <= 180 && value[1] >= -90 && value[3] <= 90 && value[1] <= value[3];
}

function readCatalog(filename) {
  const source = ts.createSourceFile(filename, fs.readFileSync(filename, 'utf8'), ts.ScriptTarget.Latest, true);
  const declaration = source.statements.find(ts.isExportAssignment);
  let expression = declaration?.expression;
  while (expression && (ts.isSatisfiesExpression(expression) || ts.isAsExpression(expression) || ts.isParenthesizedExpression(expression))) expression = expression.expression;
  if (!expression || !ts.isObjectLiteralExpression(expression)) return {};
  const values = {};
  for (const property of expression.properties) {
    if (!ts.isPropertyAssignment(property)) continue;
    if (ts.isStringLiteralLike(property.initializer)) values[property.name.getText(source).replace(/^['"]|['"]$/g, '')] = property.initializer.text;
  }
  return values;
}
