// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const NONE = 0xffffffff;

/** Deterministic vertex-order execution of the GPU trim/colour/backward/finalize steps. @internal */
export function computeBudgetedNetworkStrongComponents(
  offsets: Uint32Array,
  neighbors: Uint32Array,
  maximumIterations: number
) {
  const nodeCount = offsets.length - 1;
  const strongComponents = new Uint32Array(nodeCount).fill(NONE);
  const colors = Uint32Array.from({length: nodeCount}, (_, vertex) => vertex);
  const flags = new Uint32Array(nodeCount);
  let phase: 'trim' | 'color' | 'back' | 'finalize' | 'done' = 'trim';
  let iterationCount = 0;
  while (phase !== 'done' && iterationCount < maximumIterations) {
    let changed = false;
    let remaining = 0;
    if (phase === 'trim') {
      for (let vertex = 0; vertex < nodeCount; vertex++) {
        if (strongComponents[vertex] !== NONE) continue;
        for (let slot = offsets[vertex]; slot < offsets[vertex + 1]; slot++) {
          const neighbor = neighbors[slot];
          if (neighbor >= nodeCount || neighbor === vertex || strongComponents[neighbor] !== NONE)
            continue;
          flags[neighbor] |= 1;
          flags[vertex] |= 2;
        }
      }
      for (let vertex = 0; vertex < nodeCount; vertex++) {
        if (strongComponents[vertex] !== NONE) continue;
        if ((flags[vertex] & 3) !== 3) {
          strongComponents[vertex] = vertex;
          changed = true;
        } else remaining++;
        flags[vertex] &= ~3;
      }
      if (remaining === 0) phase = 'done';
      else if (!changed) phase = 'color';
    } else if (phase === 'color') {
      for (let vertex = 0; vertex < nodeCount; vertex++) {
        if (strongComponents[vertex] !== NONE) continue;
        const color = colors[vertex];
        for (let slot = offsets[vertex]; slot < offsets[vertex + 1]; slot++) {
          const neighbor = neighbors[slot];
          if (neighbor >= nodeCount || neighbor === vertex || strongComponents[neighbor] !== NONE)
            continue;
          if (colors[neighbor] > color) {
            colors[neighbor] = color;
            changed = true;
          }
        }
      }
      if (!changed) phase = 'back';
    } else if (phase === 'back') {
      for (let vertex = 0; vertex < nodeCount; vertex++) {
        if (strongComponents[vertex] !== NONE || (flags[vertex] & 4) !== 0) continue;
        const color = colors[vertex];
        let reachesRoot = color === vertex;
        for (let slot = offsets[vertex]; slot < offsets[vertex + 1] && !reachesRoot; slot++) {
          const neighbor = neighbors[slot];
          if (neighbor >= nodeCount || neighbor === vertex) continue;
          reachesRoot = (flags[neighbor] & 4) !== 0 && colors[neighbor] === color;
        }
        if (reachesRoot) {
          flags[vertex] |= 4;
          changed = true;
        }
      }
      if (!changed) phase = 'finalize';
    } else {
      for (let vertex = 0; vertex < nodeCount; vertex++) {
        if (strongComponents[vertex] !== NONE) continue;
        if ((flags[vertex] & 4) !== 0) {
          strongComponents[vertex] = colors[vertex];
          flags[vertex] = 0;
        } else {
          colors[vertex] = vertex;
          remaining++;
        }
      }
      phase = remaining === 0 ? 'done' : 'trim';
    }
    iterationCount++;
  }
  const converged = phase === 'done';
  if (!converged) strongComponents.fill(NONE);
  return {
    strongComponents,
    strongComponentsStatus: {
      converged: Uint32Array.of(Number(converged)),
      iterationCount: Uint32Array.of(iterationCount),
      iterationLimitReached: Uint32Array.of(Number(!converged))
    }
  };
}
