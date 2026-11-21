// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type OffsetAssemblyFixture = {
  name: string;
  fragments: number[];
  lines: number[];
  expected: number[];
  sources: number[];
  components: number[];
  capacity?: number;
  rows?: number[];
};
export const OFFSET_ASSEMBLY_SCENES: OffsetAssemblyFixture[] = [
  {
    name: 'roots before cycles',
    fragments: [0, 0, 1, 0, 10, 0, 11, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    lines: [0, 1, 0, 0],
    expected: [10, 0, 11, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    sources: [1, 1, 0, 0, 2, 3],
    components: [0, 2, 1, 0, 2, 4, 0, 0]
  },
  {
    name: 'first successor at a branch',
    fragments: [0, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 5, 0, 0, 0],
    lines: [0, 0, 0, 0],
    expected: [5, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1],
    sources: [3, 3, 0, 1, 2, 2],
    components: [0, 4, 0, 0, 4, 2, 0, 0]
  },
  {
    name: 'ordered duplicate and opposite cancellation',
    fragments: [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    lines: [0, 0, 0, 0],
    expected: [0, 0, 1, 0],
    sources: [3, 3],
    components: [0, 2, 0, 0]
  }
];
OFFSET_ASSEMBLY_SCENES.push({
  ...OFFSET_ASSEMBLY_SCENES[0],
  name: 'sparse compacted rows preserve root and cycle order',
  capacity: 257,
  rows: [0, 1, 128, 256]
});

export function packOffsetAssemblyFixture(fixture: OffsetAssemblyFixture) {
  const capacity = fixture.capacity ?? 4;
  const fragments = new Float32Array(capacity * 4);
  const metadata = new Uint32Array(capacity * 4);
  fixture.lines.forEach((line, index) => {
    const row = fixture.rows?.[index] ?? index;
    fragments.set(fixture.fragments.slice(index * 4, index * 4 + 4), row * 4);
    metadata.set([line, index, 1, 0], row * 4);
  });
  return {capacity, fragments, metadata};
}
