// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUSpatialJoinPreparedMemoryPlan} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {readDocPage} from './doc-program';
import {getDocSections} from './doc-heading-sections.test-utils';

it('GEO-115 documents whole-domain padded BVH binding sizes and distinct generation storage', () => {
  const markdown = readDocPage('gpu-spatial-analysis/concepts.md');
  expect(markdown).toContain('getGPUSpatialJoinPreparedMemoryPlan');
  const limits = {maxStorageBufferBindingSize: 128 * 1024 ** 2, maxBufferSize: 256 * 1024 ** 2};
  // Closed form: 8*(2*2^23-1)=134217720; one extra feature pads leaves to 2^24.
  for (const [featureCount, leafCapacity, bytes, fits] of [
    [8_388_608, 8_388_608, 134_217_720, true],
    [8_388_609, 16_777_216, 268_435_448, false]
  ] as const) {
    const plan = getGPUSpatialJoinPreparedMemoryPlan(
      {kind: 'points', featureCount, vertexCount: featureCount},
      limits
    );
    expect(plan.leafCapacity).toBe(leafCapacity);
    expect(plan.bindingByteLengths['BVH node bounds/children']).toBe(bytes);
    expect(
      plan.constraints.find(constraint => constraint.name === 'BVH node bounds/children')?.satisfied
    ).toBe(fits);
    expect(markdown).toContain(featureCount.toLocaleString('en-US'));
    expect(markdown).toContain(bytes.toLocaleString('en-US'));
  }
  expect(markdown).toContain('1,072');
  expect(markdown).toContain('the same retained right positions/offsets');
  expect(markdown).toContain('Source-ID contents remain live for output mapping');
});

it('REVIEW S3 limits memory planning claims to byte constraints', () => {
  const markdown = readDocPage('gpu-spatial-analysis/concepts.md').replace(/\s+/g, ' ');
  expect(markdown).not.toContain('every other binding and device constraint');
  expect(markdown).toContain('Graph declaration separately checks kernel support');
  expect(markdown).toContain('eight storage bindings');
});

function checkGatherGuidance(markdown: string, guide: string) {
  const normalized = markdown.replace(/\s+/g, ' ');
  expect(normalized).toContain('Per-query evidence withholds affected zero-count queries');
  expect(normalized).toContain(
    'supplied unattributed global loss withholds **all** derived unmatched rows'
  );
  const heading = '### `GPUPairGather`, `GPUOffsetExpansion`, and `GPUBoundsFilter`';
  const section = guide.split(heading)[1]?.split(/\n#{1,3} /)[0];
  expect(section, 'linked gather section').toBeDefined();
  const fences = [...section!.matchAll(/^```ts\n([\s\S]*?)^```/gm)].map(match => match[1]);
  expect(
    fences.filter(code => code.includes('new GPUPairGather(') && code.includes('unmatchedLeft:'))
  ).toHaveLength(1);
  expect(
    fences.filter(code => code.includes('new GPUPairGather(') && code.includes('neighbors: {'))
  ).toHaveLength(1);
}

it('REVIEW S4 rejects inverted loss guidance and examples moved outside the linked section', () => {
  const markdown = readDocPage('gpu-spatial-analysis/concepts.md');
  const guide = readDocPage('gpu-spatial-analysis/operators-joins.md');
  checkGatherGuidance(markdown, guide);
  const inverted = markdown.replace('global loss withholds **all**', 'global loss emits **all**');
  expect(inverted).not.toBe(markdown);
  expect(() => checkGatherGuidance(inverted, guide)).toThrow();
  const moved =
    guide.replace(
      'pairs: join.outputPairs!, unmatchedLeft: unmatched,',
      'pairs: join.outputPairs!,'
    ) + '\n### Unrelated\n```ts\nunmatchedLeft: unmatched\n```';
  expect(moved).not.toBe(guide);
  expect(() => checkGatherGuidance(markdown, moved)).toThrow();
});

it('GEO-113 qualifies unmatched claims and links the checked gather examples', () => {
  const markdown = readDocPage('gpu-spatial-analysis/concepts.md');
  for (const text of [
    'neighborRows',
    'neighborCounts',
    'candidateTruncated',
    'candidateOverflowSources',
    'output.status.count',
    'unattributed global loss'
  ])
    expect(markdown).toContain(text);
  const anchor = 'gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter';
  expect(markdown).toContain(`./operators-joins.md#${anchor}`);
  const section = getDocSections(readDocPage('gpu-spatial-analysis/operators-joins.md')).find(
    candidate => candidate.anchor === anchor
  );
  expect(section).toBeDefined();
  checkGatherGuidance(markdown, readDocPage('gpu-spatial-analysis/operators-joins.md'));
});

it('W5-18 distinguishes Knox from Mantel optional descriptor validation', () => {
  const markdown = readDocPage('gpu-spatial-analysis/temporal-context.md');
  const rows = markdown.split('\n').filter(line => line.startsWith('|'));
  const mantel = rows.find(line => line.includes('`GPUMantelTest`'))!;
  expect(mantel).not.toContain('GPUKnoxTest');
  expect(mantel).toContain('temporalContext');
  expect(mantel).toContain('standalone CPU');
  expect(mantel).toContain('without converting');
  expect(rows.find(line => line.includes('`GPUKnoxTest`'))).toContain('no context prop');
});
