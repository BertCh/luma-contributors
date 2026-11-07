// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent bounded scan oracle for the original FIFO priority-flood contract. */
export function calculateWatershedScanReference(props: {
  width: number;
  height: number;
  values: ArrayLike<number>;
  markers: ArrayLike<number>;
  validity?: ArrayLike<number>;
  connectivity: 4 | 8;
  maximumSteps: number;
  noDataValue?: number;
  scale?: number;
  offset?: number;
}) {
  const count = props.width * props.height;
  const values = Float32Array.from(
    {length: count},
    (_, cell) => props.values[cell] * (props.scale ?? 1) + (props.offset ?? 0)
  );
  const valid = Array.from(
    {length: count},
    (_, cell) =>
      (props.validity?.[cell] ?? 1) !== 0 &&
      Number.isFinite(props.values[cell]) &&
      props.values[cell] !== props.noDataValue &&
      Number.isFinite(values[cell])
  );
  const labels = new Array<number>(count).fill(0);
  const costs = new Float64Array(count).fill(Infinity);
  const ages = new Float64Array(count).fill(Infinity);
  const visited = valid.map(value => !value);
  let age = 0;
  for (let cell = 0; cell < count; cell++) {
    if (valid[cell] && props.markers[cell] > 0 && props.markers[cell] < 0xfffffffe) {
      labels[cell] = props.markers[cell];
      costs[cell] = values[cell];
      ages[cell] = age++;
    }
  }
  let converged = false;
  for (let step = 0; step <= Math.min(props.maximumSteps, count); step++) {
    let best = -1;
    for (let cell = 0; cell < count; cell++) {
      if (visited[cell] || labels[cell] === 0) continue;
      if (
        best < 0 ||
        costs[cell] < costs[best] ||
        (costs[cell] === costs[best] && ages[cell] < ages[best])
      )
        best = cell;
    }
    if (best < 0) {
      converged = true;
      break;
    }
    if (step === Math.min(props.maximumSteps, count)) break;
    visited[best] = true;
    for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
      for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
        if (
          (rowOffset === 0 && columnOffset === 0) ||
          (props.connectivity === 4 && Math.abs(rowOffset) + Math.abs(columnOffset) > 1)
        )
          continue;
        const column = (best % props.width) + columnOffset;
        const row = Math.floor(best / props.width) + rowOffset;
        if (column < 0 || row < 0 || column >= props.width || row >= props.height) continue;
        const neighbor = row * props.width + column;
        if (visited[neighbor] || props.markers[neighbor] !== 0) continue;
        const candidate = Math.max(costs[best], values[neighbor]);
        if (candidate < costs[neighbor]) {
          costs[neighbor] = candidate;
          labels[neighbor] = labels[best];
          ages[neighbor] = age++;
        }
      }
    }
  }
  if (!converged)
    for (let cell = 0; cell < count; cell++) {
      if (valid[cell] && props.markers[cell] === 0) labels[cell] = 0xfffffffe;
    }
  return {labels, converged: [Number(converged)]};
}
