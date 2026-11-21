// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateS2CoveringOptions} from './s2-covering-option-validation';
import {snapshotS2CapOperationProps} from './s2-covering-construction-snapshot';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from '../global-coordinates/spherical-limits';
import {
  GPU_S2_CAP_COVERING_MINIMUM_TOLERANCE,
  GPU_S2_CAP_COVERING_MAXIMUM_TOLERANCE
} from './gpu-s2-cap-covering';
import {
  GPUS2CapCoveringOperation,
  type S2CapCoveringOperationProps
} from './gpu-s2-cap-covering-operation';
import {
  CPU_S2_COVERING_MAXIMUM_LEVEL,
  type CPUCoveringVector,
  getCoveringCellGeometry,
  getCoveringCellDistance,
  getCoveringDot,
  negateCoveringVector
} from './cpu-s2-covering-geometry';

/** CPU geometry; an explicit frontierCapacity applies the GPU refinement bound. */
export type CPUS2CapCoveringOptions = Omit<
  S2CapCoveringOperationProps,
  'id' | 'caps' | 'capCount' | 'output'
>;
/** Each cap owns capacity key rows, [written, required] counts and one status word. */
export type CPUS2CapCoveringResult = {cells: Uint32Array; counts: Uint32Array; status: Uint32Array};

/**
 * Exact hierarchy traversal for the contributor's fixed-level spherical-cap cover. Cells are
 * ordered by ascending ID within each cap; unused output rows are zero.
 *
 * What differs from the GPU contributor:
 * - Float64 geometry replaces float32; classifications within GPU rounding error may differ.
 *   The exterior expansion/interior shrink tolerance and inclusive bounds are retained.
 *   The empty-interior guard compares radius with the float32-encoded tolerance, as on GPU.
 * - Omitted frontierCapacity retains complete adaptive traversal, including level-synchronous
 *   requests. An explicit frontierCapacity honors the GPU refinement bound and pruning slack;
 *   frontier exhaustion writes zero cells/counts and status bit 4. The GPU defaults to a bound.
 * - Levels up to 30 are accepted, dropping both GPU level caps. The uint32 output-count layout
 *   remains; a complete count above 0xffffffff throws instead of wrapping.
 * - Empty arrays are accepted by the pure function; graph operations require at least one cap.
 *   Invalid cap rows return status 1 and zero outputs. The radius upper bound is the GPU's
 *   float32 pi bound, so a float32 encoding of pi remains valid.
 *
 * @throws Error naming caps for incomplete packed rows, or requiredCount for unrepresentable counts.
 */
export function computeS2CapCoveringOnCPU(
  caps: Float32Array,
  options: CPUS2CapCoveringOptions
): CPUS2CapCoveringResult {
  validateOptions('s2-cap-covering', options);
  if (caps.length % 4 !== 0) throw new Error('caps must hold float32x4 rows');
  const capCount = caps.length / 4;
  const {capacity, level: targetLevel} = options;
  validateMaximumCount('s2-cap-covering', 'caps times capacity', capCount * capacity, 0xffffffff);
  const result: CPUS2CapCoveringResult = {
    cells: new Uint32Array(capCount * capacity * 2),
    counts: new Uint32Array(capCount * 2),
    status: new Uint32Array(capCount)
  };
  const tolerance = options.tolerance ?? 1e-5;
  for (let cap = 0; cap < capCount; cap++) {
    const center: CPUCoveringVector = [caps[cap * 4], caps[cap * 4 + 1], caps[cap * 4 + 2]];
    const squared = getCoveringDot(center, center);
    const radius = caps[cap * 4 + 3];
    if (
      !Number.isFinite(squared) ||
      !Number.isFinite(radius) ||
      Math.abs(squared - 1) > 1e-4 ||
      radius < 0 ||
      radius > Math.fround(3.1415928)
    ) {
      result.status[cap] = 1;
      continue;
    }
    // The GPU compares the packed radius to a float32 tolerance before expanding
    // its frontier. Equal encoded values describe an empty interior even when the
    // caller's binary64 tolerance is slightly smaller than its float32 encoding.
    if (options.mode === 'interior' && radius <= Math.fround(tolerance)) continue;
    const threshold = radius + (options.mode === 'interior' ? -tolerance : tolerance);
    if (threshold < 0 || (options.mode === 'interior' && threshold === 0)) continue;
    const opposite = negateCoveringVector(center);
    if (options.refinement === 'level-synchronous' && options.frontierCapacity !== undefined) {
      const frontierCapacity = options.frontierCapacity;
      let frontier = Array.from({length: 6}, (_, face) => BigInt(2 * face + 1) << 60n);
      for (let level = 0; level <= targetLevel; level++) {
        const selected = frontier.filter(cell => {
          const geometry = getCoveringCellGeometry(cell, level);
          return level === targetLevel && options.mode === 'interior'
            ? Math.PI - getCoveringCellDistance(geometry, opposite) <= threshold
            : getCoveringCellDistance(geometry, center) <=
                threshold + (level < targetLevel ? 1e-6 : 0);
        });
        if (level === targetLevel) {
          const required = selected.length;
          selected.slice(0, capacity).forEach((cell, row) => {
            result.cells[(cap * capacity + row) * 2] = Number(cell & 0xffffffffn);
            result.cells[(cap * capacity + row) * 2 + 1] = Number(cell >> 32n);
          });
          result.counts.set([Math.min(required, capacity), required], cap * 2);
          result.status[cap] = Number(required > capacity) * 2;
        } else {
          if (selected.length * 4 > frontierCapacity) {
            result.status[cap] = 4;
            break;
          }
          const childMarker = 1n << BigInt(58 - 2 * level);
          frontier = selected.flatMap(cell =>
            Array.from(
              {length: 4},
              (_, child) => cell - 4n * childMarker + childMarker + BigInt(child) * 2n * childMarker
            )
          );
        }
      }
      continue;
    }
    const stack: {cell: bigint; level: number}[] = [];
    for (let face = 5; face >= 0; face--) stack.push({cell: BigInt(2 * face + 1) << 60n, level: 0});
    let required = 0;
    const emitDescendants = (cell: bigint, level: number) => {
      const descendantCount = 4 ** (targetLevel - level);
      validateMaximumCount(
        's2-cap-covering',
        'requiredCount',
        required + descendantCount,
        0xffffffff
      );
      const marker = 1n << BigInt(60 - 2 * level);
      const leafMarker = 1n << BigInt(60 - 2 * targetLevel);
      const first = cell - marker + leafMarker;
      const written = Math.min(descendantCount, Math.max(0, capacity - required));
      for (let descendant = 0; descendant < written; descendant++) {
        const key = first + BigInt(descendant) * 2n * leafMarker;
        const offset = (cap * capacity + required + descendant) * 2;
        result.cells[offset] = Number(key & 0xffffffffn);
        result.cells[offset + 1] = Number(key >> 32n);
      }
      required += descendantCount;
    };
    while (stack.length) {
      const {cell, level} = stack.pop()!;
      const geometry = getCoveringCellGeometry(cell, level);
      const nearest = getCoveringCellDistance(geometry, center);
      if (nearest > threshold) continue;
      // A fully contained ancestor selects every descendant under either definition. Count
      // those descendants directly, even after output capacity is exhausted.
      const farthest = Math.PI - getCoveringCellDistance(geometry, opposite);
      if (farthest <= threshold || (level === targetLevel && options.mode !== 'interior')) {
        emitDescendants(cell, level);
      } else if (level < targetLevel) {
        const marker = 1n << BigInt(60 - 2 * level);
        const childMarker = marker >> 2n;
        for (let child = 3; child >= 0; child--)
          stack.push({
            cell: cell - marker + childMarker + BigInt(child) * 2n * childMarker,
            level: level + 1
          });
      }
    }
    result.counts.set([Math.min(required, capacity), required], cap * 2);
    result.status[cap] = Number(required > capacity) * 2;
  }
  return result;
}

/** Complete CPU cover unless an explicit level-synchronous frontier budget is supplied. */
export class CPUS2CapCoveringOperation extends GPUCPUOperation {
  constructor(props: S2CapCoveringOperationProps) {
    props = snapshotS2CapOperationProps(props);
    const operation = new GPUS2CapCoveringOperation(props);
    const {id} = operation;
    validateOptions(id, props);
    validateCapacity(id, 'capCount', props.capCount);
    validateMaximumCount(
      id,
      'capCount times capacity',
      props.capCount * props.capacity,
      0xffffffff
    );
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const caps = getCPUOperationFloat32Rows(id, inputs, 'caps', 'float32x4');
        if (caps.length !== props.capCount * 4)
          throw new Error(`${id} caps length must equal capCount * 4`);
        const result = computeS2CapCoveringOnCPU(caps, props);
        return {
          [props.output.cells]: result.cells,
          [props.output.counts]: result.counts,
          [props.output.status]: result.status
        };
      }
    });
  }
}

function validateOptions(id: string, options: CPUS2CapCoveringOptions): void {
  validateS2CoveringOptions(id, options, ['uniform', 'level-synchronous']);
  validateCapacity(id, 'level', options.level, 0, CPU_S2_COVERING_MAXIMUM_LEVEL);
  validateCapacity(
    id,
    'capacity',
    options.capacity,
    1,
    Math.min(0xffffffff, 6 * 4 ** options.level)
  );
  if (options.refinement === 'level-synchronous' && options.frontierCapacity !== undefined)
    validateCapacity(
      id,
      'frontierCapacity',
      options.frontierCapacity,
      Math.max(6, options.capacity),
      Math.floor(0xffffffff / 2)
    );
  else if (options.refinement !== 'level-synchronous')
    validateRequirement(
      id,
      options.frontierCapacity === undefined,
      "frontierCapacity needs refinement 'level-synchronous'"
    );
  validateAngularTolerance(
    id,
    options.tolerance ?? 1e-5,
    GPU_S2_CAP_COVERING_MINIMUM_TOLERANCE,
    GPU_S2_CAP_COVERING_MAXIMUM_TOLERANCE
  );
  resolveGPUSpatialContext(
    id,
    options.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
    GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
  );
}
