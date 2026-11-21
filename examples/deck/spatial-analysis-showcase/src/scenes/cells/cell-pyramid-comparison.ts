// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Provenance captured when a comparison copy is encoded, rather than when it finishes. */
export type CellComparisonIdentity = {
  key: string;
  inputGeneration: number;
  measure: 'count' | 'sum';
  resolution: number;
};

export type CellComparisonSnapshot = CellComparisonIdentity & {
  rows: number;
  cells: Uint32Array;
  before: Float32Array;
  after: Float32Array;
  delta: Float32Array;
  zScore: Float32Array;
  beforeCells: number;
  afterCells: number;
  overflow: boolean;
  candidateOverflow: boolean;
  sumOverflow: boolean;
};

/** Decode the scene's comparison publication, retaining its producing variant. */
export function readCellComparisonSnapshot(
  bytes: ArrayBuffer,
  capacity: number,
  produced: CellComparisonIdentity,
  current: CellComparisonIdentity | null
): CellComparisonSnapshot | null {
  if (
    !current ||
    produced.key !== current.key ||
    produced.inputGeneration !== current.inputGeneration ||
    produced.measure !== current.measure ||
    produced.resolution !== current.resolution
  )
    return null;
  const words = new Uint32Array(bytes);
  const rows = Math.min(words[0], capacity);
  const base = 6;
  return {
    ...produced,
    rows,
    cells: words.slice(base, base + rows * 2),
    before: new Float32Array(bytes.slice((base + capacity * 2) * 4, (base + capacity * 3) * 4)),
    after: new Float32Array(bytes.slice((base + capacity * 3) * 4, (base + capacity * 4) * 4)),
    delta: new Float32Array(bytes.slice((base + capacity * 4) * 4, (base + capacity * 5) * 4)),
    zScore: new Float32Array(bytes.slice((base + capacity * 5) * 4, (base + capacity * 6) * 4)),
    beforeCells: words[2],
    afterCells: words[3],
    overflow: words[1] !== 0,
    candidateOverflow: words[4] !== 0,
    sumOverflow: words[5] !== 0
  };
}
