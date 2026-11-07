// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type WatershedForestFixture = {
  width: number;
  height: number;
  values: Float32Array;
  markers: Uint32Array;
  validity?: Uint32Array;
  connectivity?: 4 | 8;
};

/** Independent sequential rooted Kruskal oracle: no Boruvka selections or pointer-jump rounds. */
export function calculateWatershedForestReference(fixture: WatershedForestFixture): number[] {
  const {width, height, values, markers, validity} = fixture;
  const count = width * height;
  const parents = Array.from({length: count}, (_, index) => index);
  const roots = Array.from(markers);
  const usable = (index: number) =>
    Number.isFinite(values[index]) && (validity?.[index] ?? 1) !== 0 && markers[index] < 0xfffffffe;
  const findRoot = (index: number): number => {
    let root = index;
    while (parents[root] !== root) root = parents[root];
    while (parents[index] !== index) {
      const next = parents[index];
      parents[index] = root;
      index = next;
    }
    return root;
  };
  const edges: {source: number; target: number; cost: number; identity: number}[] = [];
  for (let source = 0; source < count; source++) {
    if (!usable(source)) continue;
    const column = source % width;
    const row = Math.floor(source / width);
    for (const [direction, columnOffset, rowOffset] of [
      [0, 1, 0],
      [1, -1, 1],
      [2, 0, 1],
      [3, 1, 1]
    ]) {
      if ((fixture.connectivity ?? 4) === 4 && (direction === 1 || direction === 3)) continue;
      const targetColumn = column + columnOffset;
      const targetRow = row + rowOffset;
      if (targetColumn < 0 || targetColumn >= width || targetRow >= height) continue;
      const target = targetRow * width + targetColumn;
      if (usable(target))
        edges.push({
          source,
          target,
          cost: Math.max(values[source], values[target]),
          identity: source * 4 + direction
        });
    }
  }
  edges.sort((first, second) => first.cost - second.cost || first.identity - second.identity);
  for (const edge of edges) {
    const first = findRoot(edge.source);
    const second = findRoot(edge.target);
    if (first === second || (roots[first] !== 0 && roots[second] !== 0)) continue;
    parents[second] = first;
    roots[first] ||= roots[second];
  }
  return Array.from({length: count}, (_, index) => (usable(index) ? roots[findRoot(index)] : 0));
}

/** Deterministic tied/negative/invalid test data, independent of production kernel construction. */
export function makeWatershedForestFixture(
  width: number,
  height: number,
  seed = 1
): WatershedForestFixture {
  let state = seed;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
  const count = width * height;
  const values = Float32Array.from({length: count}, () => (random() % 13) - 6);
  const markers = Uint32Array.from({length: count}, () =>
    random() % 19 === 0 ? (random() % 7) + 1 : 0
  );
  const validity = Uint32Array.from({length: count}, () => Number(random() % 17 !== 0));
  markers[0] = 1;
  markers[count - 1] = 2;
  validity[0] = 1;
  validity[count - 1] = 1;
  return {width, height, values, markers, validity};
}
