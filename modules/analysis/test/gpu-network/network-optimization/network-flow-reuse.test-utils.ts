// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Hand-worked diamond: edge rows 0,2 and 1,3 are the two unit paths from 0 to 3. */
export const FLOW_REUSE_EDGES = Uint32Array.of(0, 1, 0, 2, 1, 3, 2, 3);
// Integral path costs are 0 for rows [0,2] and 1 for rows [1,3], with no solver tie.
export const FLOW_REUSE_COSTS = Uint32Array.of(0, 1, 0, 0);
export const FLOW_REUSE_CAPACITIES = [
  Uint32Array.of(1, 1, 1, 1),
  Uint32Array.of(1, 0, 1, 0),
  Uint32Array.of(0, 0, 0, 0),
  Uint32Array.of(1, 1, 1, 1)
];

export function expectFlowReuseRows(
  expectRows: (name: string, expected: number[]) => void,
  stage: number,
  pathCapacity: number,
  limited: boolean
): void {
  const invalid = Number(stage === 3);
  const incomplete = Number(stage === 0 && limited);
  const required = stage === 0 ? (limited ? 1 : 2) : stage === 1 ? 1 : 0;
  const count = Math.min(required, pathCapacity);
  const flow = stage === 0 && !limited ? [1, 1, 1, 1] : stage <= 1 ? [1, 0, 1, 0] : [0, 0, 0, 0];
  expectRows('flow', flow);
  expectRows('pathEdges', [
    ...(count > 0 ? [0, 2, 0xffffffff] : [0xffffffff, 0xffffffff, 0xffffffff]),
    ...(pathCapacity === 2
      ? count > 1
        ? [1, 3, 0xffffffff]
        : [0xffffffff, 0xffffffff, 0xffffffff]
      : [])
  ]);
  expectRows(
    'pathLengths',
    pathCapacity === 2 ? [count > 0 ? 2 : 0, count > 1 ? 2 : 0] : [count > 0 ? 2 : 0]
  );
  expectRows(
    'pathVolumes',
    pathCapacity === 2 ? [Number(count > 0), Number(count > 1)] : [Number(count > 0)]
  );
  expectRows('pathCount', [count]);
  expectRows('pathRequired', [required]);
  expectRows('pathOverflow', [Number(required > pathCapacity)]);
  for (const name of ['invalid', 'pathInvalid']) expectRows(name, [invalid]);
  for (const name of ['converged', 'pathConverged']) expectRows(name, [1 - invalid - incomplete]);
  for (const name of ['limited', 'pathLimited']) expectRows(name, [incomplete]);
}
