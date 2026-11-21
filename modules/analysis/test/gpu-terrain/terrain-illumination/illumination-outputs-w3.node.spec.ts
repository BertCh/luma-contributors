// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUTerrainHorizonOutput} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import type {GPUReliefShadingOutput} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import type {GPUSolarPositionOutput} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';
import type {GPUSolarIrradianceOutput} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import type {GPUTerrainCastShadowOutput} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import type {GPUSolarShadowMaskOutput} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask';
import type {GPUTextureShadingOutput} from '../../../src/gpu-terrain/terrain-illumination/gpu-texture-shading';
import type {GPUPointHorizonProfileOutput} from '../../../src/gpu-terrain/point-horizon/gpu-point-horizon-profile';
import {
  GPUTerrainDerivatives,
  type GPUTerrainDerivativesOutput
} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives';
import type {GPUTerrainViewshedOutput} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {createBand} from '../terrain-test-utils';

// Compiled by the lead's test typecheck: all ten empty output declarations must fail.
function checkOutputTypes(
  slope: GraphDataView<'float32'>,
  visibility: GraphDataView<'uint32'>
): void {
  // @ts-expect-error A writable terminal is required.
  const horizon: GPUTerrainHorizonOutput = {};
  // @ts-expect-error A writable terminal is required.
  const relief: GPUReliefShadingOutput = {};
  // @ts-expect-error A writable terminal is required.
  const position: GPUSolarPositionOutput = {};
  // @ts-expect-error A writable terminal is required.
  const irradiance: GPUSolarIrradianceOutput = {};
  // @ts-expect-error A writable terminal is required.
  const shadow: GPUTerrainCastShadowOutput = {};
  // @ts-expect-error A writable terminal is required.
  const mask: GPUSolarShadowMaskOutput = {};
  // @ts-expect-error A writable terminal is required.
  const texture: GPUTextureShadingOutput = {};
  // @ts-expect-error A writable terminal is required.
  const profile: GPUPointHorizonProfileOutput = {};
  // @ts-expect-error A writable terminal is required.
  const derivatives: GPUTerrainDerivativesOutput = {};
  // @ts-expect-error A writable terminal is required.
  const viewshed: GPUTerrainViewshedOutput = {};
  // @ts-expect-error A debug counter alone does not meet the profile's runtime guard.
  const counterOnly: GPUPointHorizonProfileOutput = {samples: visibility};
  // @ts-expect-error Derivatives do not support a nested curvatures selection.
  const nested: GPUTerrainDerivativesOutput = {curvatures: {}};
  const minimal: [
    GPUTerrainHorizonOutput,
    GPUReliefShadingOutput,
    GPUSolarPositionOutput,
    GPUSolarIrradianceOutput,
    GPUTerrainCastShadowOutput,
    GPUSolarShadowMaskOutput,
    GPUTextureShadingOutput,
    GPUPointHorizonProfileOutput,
    GPUTerrainDerivativesOutput,
    GPUTerrainViewshedOutput
  ] = [
    {skyViewFactor: slope},
    {hillshade: slope},
    {altitude: slope},
    {sunHours: slope},
    {horizonAngle: slope},
    {sunVisibility: slope},
    {textureShade: slope},
    {tangent: slope},
    {slope},
    {visibility}
  ];
  void [
    horizon,
    relief,
    position,
    irradiance,
    shadow,
    mask,
    texture,
    profile,
    derivatives,
    viewshed,
    counterOnly,
    nested,
    minimal
  ];
}
void checkOutputTypes;

it('W3-15 accepts a slope-only terminal and keeps the JavaScript empty-selection guard', () => {
  const context = createKernelAuditContext('outputs');
  try {
    const props = {
      grid: makeGPURasterMetadata({width: 3, height: 3}),
      elevation: createBand(context.graph, 'elevation', 9),
      parameters: context.view('float32', 8)
    };
    expect(
      () => new GPUTerrainDerivatives({...props, output: {slope: context.view('float32', 9)}})
    ).not.toThrow();
    expect(() => new GPUTerrainDerivatives({...props, output: {} as never})).toThrow(
      /at least one output/
    );
    expect(() => new GPUTerrainDerivatives({...props, output: {curvatures: {}} as never})).toThrow(
      /at least one output/
    );
  } finally {
    context.device.destroy();
  }
});
