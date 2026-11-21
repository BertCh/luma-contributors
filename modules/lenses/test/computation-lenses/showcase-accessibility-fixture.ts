// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Evaluate the actual application constructor expression with small caller-owned buffers. */
export function createApplicationContributor(
  source: string,
  bindings: Record<string, unknown>,
  constructorName = 'GPUNetworkAccessibility'
) {
  const start = source.indexOf(`new ${constructorName}({`);
  if (start < 0) throw new Error('Application contributor constructor missing');
  let depth = 0;
  let end = source.indexOf('(', start);
  do {
    if (source[end] === '(') depth++;
    if (source[end] === ')') depth--;
    end++;
  } while (depth > 0 && end < source.length);
  if (depth !== 0) throw new Error('Application contributor constructor unbalanced');
  return new Function(...Object.keys(bindings), `return ${source.slice(start, end)};`)(
    ...Object.values(bindings)
  );
}

/** Execute the real publication guard and statistics after a small GPU score readback. */
export function createApplicationScorePublisher(route: 'job' | 'explorer', source: string) {
  let body: string;
  if (route === 'job') {
    const start = source.indexOf('    const process = (bytes: ArrayBuffer) => {');
    const end = source.indexOf('\n    return {', start);
    if (start < 0 || end < start) throw new Error('Job score publisher missing');
    body = source
      .slice(start, end)
      .replace('(bytes: ArrayBuffer)', '(bytes)')
      .replace('(value: number)', '(value)');
  } else {
    const marker = source.indexOf('    const readScores = async (');
    const start = source.indexOf('        if (destroyed) return;', marker);
    const end = source.indexOf('\n      } catch', start);
    if (marker < 0 || start < 0 || end < start) throw new Error('Explorer score publisher missing');
    body = `const process = bytes => {${source.slice(start, end)}};`;
  }
  return new Function(`
    const values = new Map();
    const ctx = {options: {measure: 'two-step', transit: false},
      setReadout: (key, value) => values.set(key, value), requestLayers() {}};
    const context = {updateLayers() {}};
    const destroyed = false, nodeCount = 2, roadNodeCount = 2;
    const activeScoreGeneration = 1, comparisonGeneration = 1, generation = 1, scoreGeneration = 1;
    let scoresReady = true, latestScores = null, scoreMaximum = 1;
    const comparisonSnapshots = new Map();
    const comparisonKey = () => 'current';
    function publishComparisons() {}
    const cumulativeBuffer = {}, twoStepBuffer = {};
    const getScoreBuffer = () => twoStepBuffer;
    const formatInteger = String, formatCount = String, SCORE_PERCENTILE = 0.98;
    const reachedReadout = {setValue: value => values.set('reached', value)};
    const scoreRangeReadout = {setValue: value => values.set('range', value)};
    ${body}
    return {publish: process, ready: () => scoresReady, values};
  `)() as {
    publish: (bytes: ArrayBuffer | Uint8Array) => void;
    ready: () => boolean;
    values: Map<string, unknown>;
  };
}

/** Read the application's real copy sources, rather than rebuilding its packet from expectations. */
export async function readApplicationScorePacket(
  route: 'job' | 'explorer',
  source: string,
  bindings: Record<string, unknown>,
  readBuffer: (buffer: unknown, size: number) => Promise<Uint8Array>
): Promise<ArrayBuffer> {
  type Copy = {sourceBuffer: unknown; size: number; destinationOffset: number};
  const copies: Copy[] = [];
  let byteLength = 0;
  if (route === 'job') {
    const marker = source.indexOf("'access-score-columns',");
    const start = source.indexOf('[', marker);
    const end = source.indexOf('],', start) + 1;
    if (marker < 0 || start < 0 || end <= start) throw new Error('Job score copy list missing');
    const sections = new Function(...Object.keys(bindings), `return ${source.slice(start, end)};`)(
      ...Object.values(bindings)
    ) as {buffer: unknown; size: number}[];
    for (const section of sections) {
      copies.push({
        sourceBuffer: section.buffer,
        size: section.size,
        destinationOffset: byteLength
      });
      byteLength += section.size;
    }
  } else {
    const marker = source.indexOf('    const readScores = async (');
    const start = source.indexOf('      const generation =', marker);
    const end = source.indexOf('      try {', start);
    const selectionStart = source.indexOf('    const getScoreBuffer =');
    const selectionEnd = source.indexOf(';', selectionStart) + 1;
    if (marker < 0 || start < 0 || end < start || selectionStart < 0) {
      throw new Error('Explorer score copy sequence missing');
    }
    const ticket = {
      buffer: {},
      markEncoded: (range: {byteLength: number}) => {
        byteLength = range.byteLength;
      }
    };
    const locals = {
      ...bindings,
      measure: 'two-step',
      scoreGeneration: 1,
      scoreReadbackPending: false,
      scoreRing: {tryAcquire: () => ticket},
      commandEncoder: {copyBufferToBuffer: (copy: Copy) => copies.push(copy)}
    };
    const body = source.slice(selectionStart, selectionEnd).replace('(): Buffer', '()');
    new Function(...Object.keys(locals), `${body}\n${source.slice(start, end)}`)(
      ...Object.values(locals)
    );
  }
  const bytes = new Uint8Array(byteLength);
  for (const copy of copies)
    bytes.set(await readBuffer(copy.sourceBuffer, copy.size), copy.destinationOffset);
  return bytes.buffer;
}
