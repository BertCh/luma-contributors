// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';

function assertStructureGuidance(guide: string): void {
  const section = guide
    .split('## Directed components and undirected resilience\n')[1]
    ?.split('\n## ')[0];
  expect(section).toBeDefined();
  const paragraphs = section!.split(/\n\s*\n/);
  const exact = paragraphs.find(paragraph => /mode:\s*['"]exact['"]/.test(paragraph));
  expect(exact, 'explicit exact mode must name both construction bounds').toBeDefined();
  expect(exact).toMatch(/1,?024/);
  expect(exact).toMatch(/65,?536/);
  const automatic = paragraphs.find(
    paragraph =>
      /auto/.test(paragraph) && /strongComponents|SCC/.test(paragraph) && /parallel/.test(paragraph)
  );
  expect(automatic, 'auto SCC must describe switching when either bound is exceeded').toBeDefined();
  expect(automatic).toMatch(/1,?024/);
  expect(automatic).toMatch(/65,?536/);
  expect(automatic?.replace(/\s+/g, ' ')).toMatch(
    /\band switches to parallel SCC when either bound is exceeded\./
  );
  expect(
    paragraphs.some(paragraph =>
      /^auto mode uses parallel resilience even for small graphs\./i.test(
        paragraph.trim().replace(/\s+/g, ' ')
      )
    )
  ).toBe(true);
  expect(section).not.toMatch(/bounds[^.]*apply to the resilience outputs and to the\s+SCC path/);
}

const VALID_GUIDE = `## Directed components and undirected resilience

With mode: 'exact', construction supports at most 1,024 nodes and 65,536 CSR slots.

In mode: 'auto', strongComponents uses exact SCC within 1,024 nodes and 65,536 slots,
and switches to parallel SCC when either bound is exceeded.

auto mode uses parallel resilience even for small graphs.

## Unrelated section
`;

it('accepts explicit independent SCC bounds and parallel resilience on small graphs', () => {
  expect(() => assertStructureGuidance(VALID_GUIDE)).not.toThrow();
});

it.each([
  VALID_GUIDE.replace(
    'when either bound is exceeded',
    'only when both 1,024 vertices and 65,536 slots are exceeded'
  ),
  VALID_GUIDE.replace(
    'auto mode uses parallel resilience even for small graphs',
    'auto resilience never uses parallel on small graphs'
  )
])('rejects opposite automatic routing rules: %s', guide => {
  expect(() => assertStructureGuidance(guide)).toThrow();
});

it('documents exact bounds, automatic SCC selection and parallel resilience separately (DOC-104)', () => {
  const guide = readFileSync(
    new URL('../../../../../docs/api/gpu-network/expansion.md', import.meta.url),
    'utf8'
  );
  assertStructureGuidance(guide);
});

it('runs the documented CPU first run including the unreachable OD entry (UNW-14)', async () => {
  const {runInNewContext} = await import('node:vm');
  const {computeNetworkReachabilityOnCPU} = await import(
    '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation'
  );
  const {computeNetworkCostMatrixOnCPU} = await import(
    '../../../src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation'
  );
  const guide = readFileSync(
    new URL('../../../../../docs/api/gpu-network/README.md', import.meta.url),
    'utf8'
  );
  const example = [...guide.matchAll(/```ts\n([\s\S]*?)```/g)]
    .map(match => match[1])
    .find(
      code =>
        code.includes('computeNetworkReachabilityOnCPU') &&
        code.includes('computeNetworkCostMatrixOnCPU')
    );
  expect(example, 'CPU first-run code fence').toBeDefined();
  const logged: unknown[] = [];
  runInNewContext(example!.replace(/import\s*\{[\s\S]*?\}\s*from\s*['"][^'"]+['"];?/, ''), {
    Uint32Array,
    Float32Array,
    computeNetworkReachabilityOnCPU,
    computeNetworkCostMatrixOnCPU,
    console: {log: (value: unknown) => logged.push(value)}
  });
  expect(logged).toEqual([
    [0, 2, 5],
    [2, 5, Infinity, 0]
  ]);
});
