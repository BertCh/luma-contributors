// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL `getTheilSenSlope(series: ptr<function, array<f32, seriesCapacity>>, count: u32) -> f32`: the
 * exact median of the `count (count - 1) / 2` pairwise slopes `(series[j] - series[i]) / (j - i)`,
 * selected in a bounded max-heap of the `pairs / 2 + 1` smallest slopes (no sort of all pairs), the
 * average of the two middle slopes for an even pair count, 0 for `count < 2`.
 *
 * @internal
 */
export function getTheilSenSlopeWGSL(seriesCapacity: number): string {
  const pairCapacity = (seriesCapacity * (seriesCapacity - 1)) / 2;
  const heapCapacity = Math.floor(pairCapacity / 2) + 1;
  return /* wgsl */ `
fn getTheilSenSlope(series: ptr<function, array<f32, ${seriesCapacity}>>, count: u32) -> f32 {
  let pairCount = count * (count - 1u) / 2u;
  if (pairCount == 0u) {
    return 0.0;
  }
  let heapTarget = pairCount / 2u + 1u;
  var heap: array<f32, ${heapCapacity}>;
  var size = 0u;
  for (var first = 0u; first + 1u < count; first++) {
    for (var second = first + 1u; second < count; second++) {
      let slope = ((*series)[second] - (*series)[first]) / f32(second - first);
      if (size < heapTarget) {
        var child = size;
        heap[child] = slope;
        size++;
        while (child > 0u) {
          let parent = (child - 1u) / 2u;
          if (heap[parent] >= heap[child]) {
            break;
          }
          let swapped = heap[parent];
          heap[parent] = heap[child];
          heap[child] = swapped;
          child = parent;
        }
      } else if (slope < heap[0]) {
        heap[0] = slope;
        var parent = 0u;
        loop {
          let left = 2u * parent + 1u;
          if (left >= size) {
            break;
          }
          var larger = left;
          if (left + 1u < size && heap[left + 1u] > heap[left]) {
            larger = left + 1u;
          }
          if (heap[parent] >= heap[larger]) {
            break;
          }
          let swapped = heap[parent];
          heap[parent] = heap[larger];
          heap[larger] = swapped;
          parent = larger;
        }
      }
    }
  }
  var median = heap[0];
  if (pairCount % 2u == 0u && size >= 2u) {
    var secondLargest = heap[1];
    if (size > 2u) {
      secondLargest = max(secondLargest, heap[2]);
    }
    median = 0.5 * (median + secondLargest);
  }
  return median;
}`;
}
