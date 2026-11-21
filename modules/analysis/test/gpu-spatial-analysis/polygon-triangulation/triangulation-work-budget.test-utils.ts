// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getTriangulationWGSL} from '../../../src/analysis-core/polygon-triangulation/polygon-triangulation-earcut-wgsl';

/** Execute the production work-budget transition at uint32 boundaries without billions of steps. */
export function getTriangulationSpendWGSL(limit: number): string {
  const source = getTriangulationWGSL({
    vertexCount: 3,
    polygonCount: 1,
    ringCount: 1,
    indexLength: 3,
    nodeCount: 16,
    useZOrderHash: false,
    maximumWork: limit,
    writeTriangleCount: false,
    minimumVertices: 4096
  });
  const spend = source.match(/fn spend\(amount: u32\) \{[\s\S]*?\n\}/)?.[0];
  if (!spend) throw new Error('Missing production triangulation spend function');
  return `const WORK_LIMIT: u32 = ${limit}u;\nvar<private> workDone: u32;\nvar<private> failed: bool;\n${spend}`;
}

export const WORK_BUDGET_CASES = [
  {
    name: 'uint32 maximum exact endpoint then one more',
    limit: 0xffffffff,
    start: 0xfffffffe,
    amounts: [1, 1],
    expected: [0xffffffff, 1]
  },
  {
    name: 'uint32 maximum oversized increment',
    limit: 0xffffffff,
    start: 0xfffffffe,
    amounts: [3],
    expected: [0xfffffffe, 1]
  },
  {
    name: 'ordinary exact endpoint then one more',
    limit: 5,
    start: 4,
    amounts: [1, 1],
    expected: [5, 1]
  },
  {name: 'ordinary oversized first increment', limit: 3, start: 0, amounts: [4], expected: [0, 1]}
] as const;
