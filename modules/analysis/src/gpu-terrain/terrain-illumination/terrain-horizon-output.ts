// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  TERRAIN_HORIZON_UNORM16_SCALE,
  type GPUTerrainHorizonFormat
} from './terrain-illumination-utils';

/** Props of {@link getTerrainHorizonSectorOutput}. @internal */
export type TerrainHorizonSectorOutputProps = {
  /** Sector index in `[0, directionCount)`; ignored when {@link sectorWGSL} is set. */
  sector: number;
  /**
   * WGSL `u32` expression of the sector, read at run time (library rule 12: the sector index then
   * never reaches the WGSL text). Sector 0 still assigns the sums and later sectors add.
   */
  sectorWGSL?: string;
  /** Number of sectors. */
  directionCount: number;
  /** Storage format of `horizon`. Defaults to `'float32'`. */
  horizonFormat?: GPUTerrainHorizonFormat;
  /** Horizon map written in `'zenith'` mode (`read_write`, binding name `horizon`). */
  horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Per-pixel sum of `sin(max(h, 0))` (binding `sineSum`), `'zenith'` mode. */
  sineSum?: GraphDataView<'float32'>;
  /** Per-pixel sum of `h` (binding `angleSum`), `'zenith'` mode. */
  angleSum?: GraphDataView<'float32'>;
  /**
   * Per-pixel weighted sum of `sin(max(h, 0))` (binding `anisotropicSum`), `'zenith'` mode. The
   * caller's kernel must bind `settings` (`array<f32>`, 12 values, slots 8 to 10 hold the RVT main direction,
   * level and minimum weight) and declare {@link TERRAIN_ILLUMINATION_WGSL_CONSTANTS}.
   */
  anisotropicSum?: GraphDataView<'float32'>;
  /** Per-pixel sum of the nadir angle `h_nadir` (binding `nadirSum`), `'nadir'` mode only. */
  nadirSum?: GraphDataView<'float32'>;
  /**
   * `'zenith'` stores the horizon and accumulates the sine, angle and anisotropic sums. `'nadir'`
   * only accumulates `nadirSum`, so a caller running zenith and nadir as separate nodes emits one
   * helper per node.
   */
  mode: 'zenith' | 'nadir';
  /** Storage bindings the caller adds besides the outputs. Defaults to 3 (values, validity, settings). */
  reservedBindingCount?: number;
  /** Device storage-buffer limit per stage. Defaults to 8. */
  maximumBindingCount?: number;
};

/**
 * Generates the per-sector output of horizon kernels: horizon store (float32 or unorm16), the
 * sine, angle, anisotropic and nadir accumulations (sector 0 assigns, later sectors add). One
 * implementation serves the ray march and the sweep.
 *
 * `wgsl` consumes `pixel: u32` and `horizonAngle: f32` (degrees, NaN for an invalid center) and
 * is wrapped in its own block, so it may be emitted twice into one kernel (zenith and nadir) as
 * long as the nadir call sees the nadir angle as `horizonAngle` in an inner scope. `declarations`
 * are module-scope WGSL (empty unless `anisotropicSum` is bound or `horizonFormat` is
 * `'unorm16'`), `bindings` the `read_write` storage bindings to append after the caller's own.
 *
 * @throws If a mode-specific output is bound in the wrong mode, nothing is bound, or the kernel
 *   would exceed `maximumBindingCount` storage bindings.
 * @internal
 */
export function getTerrainHorizonSectorOutput(props: TerrainHorizonSectorOutputProps): {
  bindings: WGSLKernelBinding[];
  declarations: string;
  wgsl: string;
} {
  const {sector, directionCount, mode} = props;
  const format = props.horizonFormat ?? 'float32';
  if (mode === 'nadir') {
    if (props.horizon || props.sineSum || props.angleSum || props.anisotropicSum) {
      throw new Error('Terrain horizon nadir output only accumulates nadirSum');
    }
  } else if (props.nadirSum) {
    throw new Error('Terrain horizon zenith output cannot write nadirSum');
  }
  const bindings: WGSLKernelBinding[] = [];
  const statements: string[] = [];
  let declarations = '';
  const sectorExpression = props.sectorWGSL ? `(${props.sectorWGSL})` : `${sector}u`;
  // Sector 0 assigns and later sectors add: `select` keeps exactly the bits of either form.
  const accumulate = (target: string, term: string): string =>
    props.sectorWGSL
      ? `${target} = select(${target} + ${term}, ${term}, ${sectorExpression} == 0u);`
      : `${target} ${sector === 0 ? '=' : '+='} ${term};`;
  if (props.horizon) {
    if (format === 'unorm16') {
      bindings.push({name: 'horizon', view: props.horizon, type: 'u32', access: 'read_write'});
      declarations += `fn encodeHorizonUnorm16(angle: f32) -> u32 {
  if (!isFiniteValue(angle)) { return 0u; }
  return 1u + u32(floor(clamp(angle, -90.0, 90.0) * ${getWGSLFloatLiteral(TERRAIN_HORIZON_UNORM16_SCALE)} + 32767.0 + 0.5));
}
`;
      // One sector per dispatch and DIRECTION_COUNT >= 4, so distinct pixels never share a word.
      statements.push(`let element = pixel * ${directionCount}u + ${sectorExpression};
  let wordIndex = horizonOffset + (element >> 1u);
  let shift = (element & 1u) * 16u;
  horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
    (encodeHorizonUnorm16(horizonAngle) << shift);`);
    } else {
      bindings.push({name: 'horizon', view: props.horizon, type: 'f32', access: 'read_write'});
      statements.push(
        `horizon[horizonOffset + pixel * ${directionCount}u + ${sectorExpression}] = horizonAngle;`
      );
    }
  }
  if (props.sineSum) {
    bindings.push({name: 'sineSum', view: props.sineSum, type: 'f32', access: 'read_write'});
    statements.push(
      accumulate(
        'sineSum[sineSumOffset + pixel]',
        'sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS)'
      )
    );
  }
  if (props.angleSum) {
    bindings.push({name: 'angleSum', view: props.angleSum, type: 'f32', access: 'read_write'});
    statements.push(accumulate('angleSum[angleSumOffset + pixel]', 'horizonAngle'));
  }
  if (props.anisotropicSum) {
    bindings.push({
      name: 'anisotropicSum',
      view: props.anisotropicSum,
      type: 'f32',
      access: 'read_write'
    });
    declarations += getTerrainHorizonAnisotropicWeightWGSL(directionCount);
    statements.push(
      accumulate(
        'anisotropicSum[anisotropicSumOffset + pixel]',
        `getAnisotropicWeight(${sectorExpression}) *
    sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS)`
      )
    );
  }
  if (props.nadirSum) {
    bindings.push({name: 'nadirSum', view: props.nadirSum, type: 'f32', access: 'read_write'});
    statements.push(accumulate('nadirSum[nadirSumOffset + pixel]', 'horizonAngle'));
  }
  if (bindings.length === 0) {
    throw new Error('Terrain horizon sector output requires at least one output');
  }
  const total = bindings.length + (props.reservedBindingCount ?? 3);
  const limit = props.maximumBindingCount ?? 8;
  if (total > limit) {
    throw new Error(
      `Terrain horizon sector output needs ${total} storage bindings, more than the limit of ${limit}`
    );
  }
  return {bindings, declarations, wgsl: `{\n  ${statements.join('\n  ')}\n  }`};
}

/**
 * WGSL `fn getAnisotropicWeight(sector: u32) -> f32` for `directionCount` sectors, reading the
 * RVT main direction, level and minimum weight from `settings[8..10]`.
 *
 * Convention, stated in compass terms so it does not depend on RVT: sector `d` looks along
 * compass azimuth `t = d * 360 / n` (clockwise from north) and gets the angle
 * `a = (360 - t) mod 360`, counterclockwise from north. Its weight is
 * `(1 - wMin) * cos((a - A) / 2)^level + wMin` with the signed power of an integer level
 * (`0^0 = 1`, as numpy). The weight is largest for the sector at compass azimuth
 * `(360 - A) mod 360`: `A = 315` weights the north-east sky, `A = 90` the west.
 *
 * This equals `rvt.vis.sky_view_factor_compute` of rvt-py 2.2.1, where direction `i` has angle
 * `i * 360 / n` and looks at the pixel offset `(-cos, -sin) * r` over (rows, columns) of a
 * north-up array: counterclockwise from north, so compass sector `d` is RVT direction
 * `(n - d) mod n`. RVT does not wrap `a - A`, and neither does this kernel, which matters for odd
 * levels. Checked against unmodified rvt-py on walls and ramps at the four compass sides and the
 * diagonals, on north-up and rows-north grids, with both presets and an odd level
 * (`test/oracles/terrain-sky-view-rvt.py`, `gpu-terrain-horizon-rvt.spec.ts`): where RVT and the
 * kernel read the same samples the factors agree to 3e-7.
 *
 * @internal
 */
export function getTerrainHorizonAnisotropicWeightWGSL(directionCount: number): string {
  return `fn getAnisotropicWeight(sector: u32) -> f32 {
  let mainDirection = settings[settingsOffset + 8u];
  let level = settings[settingsOffset + 9u];
  let minimumWeight = settings[settingsOffset + 10u];
  // Compass sector d is RVT direction (n - d) mod n; RVT angles run counterclockwise from north.
  let rvtDirection = f32((${directionCount}u - sector) % ${directionCount}u) * ${getWGSLFloatLiteral(360 / directionCount)};
  let cosine = cos((rvtDirection - mainDirection) * 0.5 * DEGREES_TO_RADIANS);
  var power = 1.0;
  if (level > 0.0) {
    // pow is undefined for a zero or negative base; odd integer levels keep the sign like RVT.
    power = select(0.0, pow(abs(cosine), level), abs(cosine) > 0.0);
    if (cosine < 0.0 && fract(level * 0.5) != 0.0) { power = -power; }
  }
  return (1.0 - minimumWeight) * power + minimumWeight;
}
`;
}
