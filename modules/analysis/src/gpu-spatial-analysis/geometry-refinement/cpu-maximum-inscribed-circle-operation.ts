// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRefinementOperationProps} from './refinement-construction-snapshot';

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  areCPUPolygonVerticesCollinear,
  findCPUPolygonPole,
  publishCPUPolygonPole,
  readCPUPolygonRows
} from '../label-point/cpu-label-point-operation';
import {
  getCPUPolygonRings,
  getCPUPolygonDistance,
  validateCPUPolygonRows,
  type CPUPolygonRows
} from '../shared/cpu-polygon-rows';
import {
  GPUMaximumInscribedCircleOperation,
  type MaximumInscribedCircleOperationProps
} from './gpu-maximum-inscribed-circle-operation';

/** Circle, witness and status rows, in GPUMaximumInscribedCircle's layout. */
export type CPUMaximumInscribedCircleResult = {
  circles: Float32Array;
  /** Clearance at the published float32 center, also used as circle radius. */
  publishedCenterClearance: Float32Array;
  /** Per feature: center publication reduced clearance or the absolute witness lost precision. */
  precisionLimited: Uint32Array;
  witnesses: Float32Array;
  /** Nearest boundary witness minus the published center, in position units (float32x2). */
  witnessOffsets: Float32Array;
  status: Uint32Array;
};

type CircleWorkOptions = Pick<
  MaximumInscribedCircleOperationProps,
  'maximumIterations' | 'cellCapacity'
>;

function findBoundedCircle(
  input: CPUPolygonRows,
  feature: number,
  tolerance: number,
  options: CircleWorkOptions
) {
  const rings = getCPUPolygonRings(input, feature);
  let minimumX = Infinity,
    minimumY = Infinity,
    maximumX = -Infinity,
    maximumY = -Infinity;
  let centerX = NaN,
    centerY = NaN;
  let invalid = false;
  for (const ring of rings)
    for (let row = ring.start; row < ring.end; row++) {
      centerX = input.positions[2 * row];
      centerY = input.positions[2 * row + 1];
      invalid ||= !Number.isFinite(centerX) || !Number.isFinite(centerY);
      minimumX = Math.min(minimumX, centerX);
      maximumX = Math.max(maximumX, centerX);
      minimumY = Math.min(minimumY, centerY);
      maximumY = Math.max(maximumY, centerY);
    }
  if (invalid || minimumX === Infinity)
    return {
      centerX: NaN,
      centerY: NaN,
      radius: NaN,
      witnessX: NaN,
      witnessY: NaN,
      error: NaN,
      status: invalid ? 3 : 2
    };
  let best = {
    centerX,
    centerY,
    radius: 0,
    witnessX: centerX,
    witnessY: centerY,
    error: 0,
    status: 0
  };
  if (areCPUPolygonVerticesCollinear(input.positions, rings)) return best;
  const maximumRadius = 0.5 * Math.min(maximumX - minimumX, maximumY - minimumY);
  const createCell = (positionX: number, positionY: number, halfSize: number) => {
    const distance = getCPUPolygonDistance(input.positions, rings, positionX, positionY);
    if (distance.distance > best.radius)
      best = {
        ...best,
        centerX: positionX,
        centerY: positionY,
        radius: distance.distance,
        witnessX: distance.witnessX,
        witnessY: distance.witnessY
      };
    return {
      centerX: positionX,
      centerY: positionY,
      halfSize,
      bound: Math.min(maximumRadius, distance.distance + Math.SQRT2 * halfSize)
    };
  };
  let cells = [
    createCell(
      0.5 * minimumX + 0.5 * maximumX,
      0.5 * minimumY + 0.5 * maximumY,
      0.5 * Math.max(maximumX - minimumX, maximumY - minimumY)
    )
  ];
  let iterations = 0;
  while (cells.length) {
    let winner = 0;
    for (let cell = 1; cell < cells.length; cell++)
      if (cells[cell].bound > cells[winner].bound) winner = cell;
    const parent = cells[winner];
    best.error = Math.max(0, parent.bound - best.radius);
    if (best.error <= tolerance) break;
    if (
      iterations >= (options.maximumIterations ?? Infinity) ||
      cells.length + 3 > (options.cellCapacity ?? Infinity) ||
      parent.halfSize * 0.5 === parent.halfSize
    ) {
      best.status = 1;
      break;
    }
    cells[winner] = cells[cells.length - 1];
    cells.pop();
    const halfSize = parent.halfSize * 0.5;
    for (let child = 0; child < 4; child++)
      cells.push(
        createCell(
          parent.centerX + halfSize * (child & 1 ? 1 : -1),
          parent.centerY + halfSize * (child & 2 ? 1 : -1),
          halfSize
        )
      );
    cells = cells.filter(cell => cell.bound > best.radius);
    if (cells.length === 0) {
      best.error = 0;
      break;
    }
    iterations++;
  }
  return best;
}

/**
 * Best-first even-odd polygon circle search with dynamic live-cell storage in binary64.
 *
 * What differs from the GPU contributor:
 * - Cell storage is dynamic, but explicit cellCapacity and maximumIterations bind before
 *   subdivision at the same point as the contributor. Omitted limits allow search completion.
 * - The remaining-error field bounds the completed binary64 search, stopping at tolerance or
 *   binary64 resolution. Status 1 means a work/storage budget or binary64 resolution
 *   prevented subdivision before the requested tolerance was reached.
 * - Exactly collinear float32 vertices have an exact zero-radius result and zero remaining
 *   error on every budget route. Half the shorter envelope side caps every search bound.
 * - Float32 arithmetic overflow does not set status 3; only non-finite input does. Finite float32
 *   coordinates are evaluated in binary64 and outputs are rounded once. Radius and witness are
 *   recomputed at the published float32 center; precisionLimited reports lost clearance or
 *   absolute witness precision. Exterior rounded centers fall back to a source vertex. The
 *   fourth circle component remains unresolved search error, excluding publication loss. The
 *   center-relative witnessOffsets remains useful when the absolute witness rounds to the center.
 * - Live-slot winner ties, child order, last-vertex fallback and first closest-segment witness
 *   follow the GPU search. Different arithmetic can still resolve near ties differently.
 *
 * @throws If tolerance is not positive and finite, or input arrays are inconsistent.
 */
export function computeMaximumInscribedCircleOnCPU(
  input: CPUPolygonRows,
  tolerance: number,
  options: CircleWorkOptions = {}
): CPUMaximumInscribedCircleResult {
  validateCPUPolygonRows(input);
  if (options.maximumIterations !== undefined)
    validateCapacity('maximum-inscribed-circle', 'maximumIterations', options.maximumIterations, 0);
  if (options.cellCapacity !== undefined)
    validateCapacity('maximum-inscribed-circle', 'cellCapacity', options.cellCapacity, 1);
  if (!(Number.isFinite(tolerance) && tolerance > 0)) {
    throw new Error('tolerance must be positive and finite');
  }
  const featureCount = input.featureOffsets.length - 1;
  const circles = new Float32Array(featureCount * 4);
  const witnesses = new Float32Array(featureCount * 2);
  const status = new Uint32Array(featureCount);
  const witnessOffsets = new Float32Array(featureCount * 2);
  const publishedCenterClearance = new Float32Array(featureCount);
  const precisionLimited = new Uint32Array(featureCount);
  for (let feature = 0; feature < featureCount; feature++) {
    const pole =
      options.maximumIterations === undefined && options.cellCapacity === undefined
        ? findCPUPolygonPole(input.positions, getCPUPolygonRings(input, feature), tolerance)
        : findBoundedCircle(input, feature, tolerance, options);
    const published = publishCPUPolygonPole(
      input.positions,
      getCPUPolygonRings(input, feature),
      pole
    );
    circles.set(
      [published.centerX, published.centerY, published.clearance, pole.error],
      feature * 4
    );
    publishedCenterClearance[feature] = published.clearance;
    witnesses.set([published.witnessX, published.witnessY], feature * 2);
    witnessOffsets.set(
      [published.witnessX - published.centerX, published.witnessY - published.centerY],
      feature * 2
    );
    const witnessRounded =
      witnesses[feature * 2] !== published.witnessX ||
      witnesses[feature * 2 + 1] !== published.witnessY;
    precisionLimited[feature] =
      published.precisionLimited || (Number.isFinite(published.clearance) && witnessRounded)
        ? 1
        : 0;
    status[feature] = pole.status;
  }
  return {circles, witnesses, witnessOffsets, status, publishedCenterClearance, precisionLimited};
}

/** CPU named circle form, including additive publication diagnostics. */
export type CPUMaximumInscribedCircleOperationProps = Omit<
  MaximumInscribedCircleOperationProps,
  'output'
> & {
  /** CPU host interpretation, defaulting to resident-capacity for graph inputs. */
  storage?: CPUPolygonRows['storage'];
  output: MaximumInscribedCircleOperationProps['output'] & {
    publishedCenterClearance?: string;
    precisionLimited?: string;
    witnessOffsets?: string;
  };
};

/** CPU twin; explicit cellCapacity and maximumIterations bind as on the contributor. */
export class CPUMaximumInscribedCircleOperation extends GPUCPUOperation {
  constructor(props: CPUMaximumInscribedCircleOperationProps) {
    props = snapshotRefinementOperationProps(props, 'cpu-circle');
    const storage = props.storage;
    const extraOutput = snapshotConstructionFields(props.output, [
      'publishedCenterClearance',
      'precisionLimited',
      'witnessOffsets'
    ]);
    const operation = new GPUMaximumInscribedCircleOperation(props);
    const {id} = operation;
    validateRequirement(
      id,
      storage === undefined || ['compact', 'resident-capacity'].includes(storage),
      `storage must be compact or resident-capacity; received ${storage}`
    );
    const inputNames = [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
      props.featureOffsets
    ];
    const outputNames = [
      props.output.circles,
      props.output.witnesses,
      props.output.status,
      ...Object.values(extraOutput).filter((name): name is string => name !== undefined)
    ];
    validateRequirement(
      id,
      new Set(outputNames).size === outputNames.length &&
        outputNames.every(
          name => typeof name === 'string' && name.length > 0 && !inputNames.includes(name)
        ),
      'publication outputs must use distinct destinations, disjoint from polygon input names'
    );
    validateCapacity(id, 'featureCount', props.featureCount, 0);
    validateFiniteNumber(id, 'tolerance', props.tolerance);
    validateRequirement(id, props.tolerance > 0, 'tolerance must be greater than 0');
    if (props.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', props.maximumIterations, 0);
    if (props.cellCapacity !== undefined)
      validateCapacity(id, 'cellCapacity', props.cellCapacity, 1);
    resolveGPUSpatialContext(id, props.spatialContext);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries([
        ...Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ]),
        ...Object.entries(extraOutput)
          .filter(([, value]) => value !== undefined)
          .map(([port, value]) => [
            value!,
            gpuGraph.vector(
              port === 'witnessOffsets'
                ? 'float32x2'
                : port === 'precisionLimited'
                  ? 'uint32'
                  : 'float32',
              {length: props.featureCount}
            )
          ])
      ]),
      execute({inputs}) {
        const input = readCPUPolygonRows(id, inputs, storage);
        validateRequirement(
          id,
          input.featureOffsets.length === props.featureCount + 1,
          'featureOffsets length must equal featureCount + 1'
        );
        const result = computeMaximumInscribedCircleOnCPU(input, props.tolerance, props);
        return {
          [props.output.circles]: result.circles,
          [props.output.witnesses]: result.witnesses,
          [props.output.status]: result.status,
          ...Object.fromEntries(
            Object.entries(extraOutput)
              .filter(([, value]) => value !== undefined)
              .map(([port, value]) => [
                value!,
                result[port as keyof CPUMaximumInscribedCircleResult]
              ])
          )
        };
      }
    });
  }
}
