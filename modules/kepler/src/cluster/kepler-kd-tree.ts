// luma.gl
// SPDX-License-Identifier: MIT AND ISC
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright (c) Vladimir Agafonkin

/**
 * Typed-array port of the kdbush 3.0 tree that supercluster 7 uses (Float32 coordinates, node size
 * 64). The selection sort, the median split and the traversal order are copied exactly, because the
 * order of `within` and `range` results decides merge order and floating point centroid sums.
 */
export class KeplerKDTree {
  readonly ids: Uint16Array | Uint32Array;
  readonly coordinates: Float32Array;
  private readonly nodeSize: number;
  private readonly stack: number[] = [];

  constructor(x: Float32Array, y: Float32Array, count: number, nodeSize = 64) {
    this.nodeSize = nodeSize;
    this.ids = count < 65536 ? new Uint16Array(count) : new Uint32Array(count);
    this.coordinates = new Float32Array(count * 2);
    for (let index = 0; index < count; index++) {
      this.ids[index] = index;
      this.coordinates[2 * index] = x[index];
      this.coordinates[2 * index + 1] = y[index];
    }
    sortTree(this.ids, this.coordinates, nodeSize, 0, count - 1, 0);
  }

  get byteLength(): number {
    return this.ids.byteLength + this.coordinates.byteLength;
  }

  /** Appends ids inside the rectangle to `result` (cleared first), in kdbush traversal order. */
  range(minX: number, minY: number, maxX: number, maxY: number, result: number[]): void {
    const {ids, coordinates, nodeSize, stack} = this;
    stack.length = 0;
    result.length = 0;
    stack.push(0, ids.length - 1, 0);
    while (stack.length) {
      const axis = stack.pop() as number;
      const right = stack.pop() as number;
      const left = stack.pop() as number;
      if (right - left <= nodeSize) {
        for (let index = left; index <= right; index++) {
          const x = coordinates[2 * index];
          const y = coordinates[2 * index + 1];
          if (x >= minX && x <= maxX && y >= minY && y <= maxY) result.push(ids[index]);
        }
        continue;
      }
      const middle = Math.floor((left + right) / 2);
      const x = coordinates[2 * middle];
      const y = coordinates[2 * middle + 1];
      if (x >= minX && x <= maxX && y >= minY && y <= maxY) result.push(ids[middle]);
      const nextAxis = (axis + 1) % 2;
      if (axis === 0 ? minX <= x : minY <= y) stack.push(left, middle - 1, nextAxis);
      if (axis === 0 ? maxX >= x : maxY >= y) stack.push(middle + 1, right, nextAxis);
    }
  }

  /** True when a node other than `selfId` lies within `radius` of the point (same test as `within`). */
  hasOtherWithin(queryX: number, queryY: number, radius: number, selfId: number): boolean {
    const {ids, coordinates, nodeSize, stack} = this;
    stack.length = 0;
    stack.push(0, ids.length - 1, 0);
    const radiusSquared = radius * radius;
    while (stack.length) {
      const axis = stack.pop() as number;
      const right = stack.pop() as number;
      const left = stack.pop() as number;
      if (right - left <= nodeSize) {
        for (let index = left; index <= right; index++) {
          const dx = coordinates[2 * index] - queryX;
          const dy = coordinates[2 * index + 1] - queryY;
          if (dx * dx + dy * dy <= radiusSquared && ids[index] !== selfId) return true;
        }
        continue;
      }
      const middle = Math.floor((left + right) / 2);
      const x = coordinates[2 * middle];
      const y = coordinates[2 * middle + 1];
      const dx = x - queryX;
      const dy = y - queryY;
      if (dx * dx + dy * dy <= radiusSquared && ids[middle] !== selfId) return true;
      const nextAxis = (axis + 1) % 2;
      if (axis === 0 ? queryX - radius <= x : queryY - radius <= y)
        stack.push(left, middle - 1, nextAxis);
      if (axis === 0 ? queryX + radius >= x : queryY + radius >= y)
        stack.push(middle + 1, right, nextAxis);
    }
    return false;
  }

  /** Appends ids within `radius` of the point to `result` (cleared first). */
  within(queryX: number, queryY: number, radius: number, result: number[]): void {
    const {ids, coordinates, nodeSize, stack} = this;
    stack.length = 0;
    result.length = 0;
    stack.push(0, ids.length - 1, 0);
    const radiusSquared = radius * radius;
    while (stack.length) {
      const axis = stack.pop() as number;
      const right = stack.pop() as number;
      const left = stack.pop() as number;
      if (right - left <= nodeSize) {
        for (let index = left; index <= right; index++) {
          const dx = coordinates[2 * index] - queryX;
          const dy = coordinates[2 * index + 1] - queryY;
          if (dx * dx + dy * dy <= radiusSquared) result.push(ids[index]);
        }
        continue;
      }
      const middle = Math.floor((left + right) / 2);
      const x = coordinates[2 * middle];
      const y = coordinates[2 * middle + 1];
      const dx = x - queryX;
      const dy = y - queryY;
      if (dx * dx + dy * dy <= radiusSquared) result.push(ids[middle]);
      const nextAxis = (axis + 1) % 2;
      if (axis === 0 ? queryX - radius <= x : queryY - radius <= y)
        stack.push(left, middle - 1, nextAxis);
      if (axis === 0 ? queryX + radius >= x : queryY + radius >= y)
        stack.push(middle + 1, right, nextAxis);
    }
  }
}

function sortTree(
  ids: Uint16Array | Uint32Array,
  coordinates: Float32Array,
  nodeSize: number,
  left: number,
  right: number,
  depth: number
): void {
  if (right - left <= nodeSize) return;
  const middle = (left + right) >> 1;
  selectItem(ids, coordinates, middle, left, right, depth % 2);
  sortTree(ids, coordinates, nodeSize, left, middle - 1, depth + 1);
  sortTree(ids, coordinates, nodeSize, middle + 1, right, depth + 1);
}

function selectItem(
  ids: Uint16Array | Uint32Array,
  coordinates: Float32Array,
  k: number,
  leftStart: number,
  rightStart: number,
  axis: number
): void {
  let left = leftStart;
  let right = rightStart;
  while (right > left) {
    if (right - left > 600) {
      const n = right - left + 1;
      const m = k - left + 1;
      const z = Math.log(n);
      const s = 0.5 * Math.exp((2 * z) / 3);
      const sd = 0.5 * Math.sqrt((z * s * (n - s)) / n) * (m - n / 2 < 0 ? -1 : 1);
      const newLeft = Math.max(left, Math.floor(k - (m * s) / n + sd));
      const newRight = Math.min(right, Math.floor(k + ((n - m) * s) / n + sd));
      selectItem(ids, coordinates, k, newLeft, newRight, axis);
    }
    const t = coordinates[2 * k + axis];
    let i = left;
    let j = right;
    swapItems(ids, coordinates, left, k);
    if (coordinates[2 * right + axis] > t) swapItems(ids, coordinates, left, right);
    while (i < j) {
      swapItems(ids, coordinates, i, j);
      i++;
      j--;
      while (coordinates[2 * i + axis] < t) i++;
      while (coordinates[2 * j + axis] > t) j--;
    }
    if (coordinates[2 * left + axis] === t) swapItems(ids, coordinates, left, j);
    else {
      j++;
      swapItems(ids, coordinates, j, right);
    }
    if (j <= k) left = j + 1;
    if (k <= j) right = j - 1;
  }
}

function swapItems(
  ids: Uint16Array | Uint32Array,
  coordinates: Float32Array,
  i: number,
  j: number
): void {
  const id = ids[i];
  ids[i] = ids[j];
  ids[j] = id;
  const x = coordinates[2 * i];
  const y = coordinates[2 * i + 1];
  coordinates[2 * i] = coordinates[2 * j];
  coordinates[2 * i + 1] = coordinates[2 * j + 1];
  coordinates[2 * j] = x;
  coordinates[2 * j + 1] = y;
}
