// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma-contributors/graph/gpu-core';
import type {GPURasterBufferBand, GPURasterScalarFormat} from '../../src/gpu-raster/types';
import {
  GPURasterNDWI,
  GPURasterNBR,
  GPURasterEVI,
  GPURasterSAVI
} from '../../src/gpu-raster/gpu-raster-spectral-indices';
import fixture from '../oracles/fixtures/raster-spectral-indices.json';

export type SpectralBandFixture = {
  format: GPURasterScalarFormat;
  values: number[];
  validity?: number[];
  scale?: number;
  offset?: number;
  noDataValue?: number;
};
export type SpectralExpected = {values: (number | null)[]; validity: number[]};
export type SpectralCase = {
  name: string;
  index: string;
  width: number;
  height: number;
  bands: Record<string, SpectralBandFixture>;
  epsilon: number;
  soilAdjustment: number;
  clamp?: [number, number];
  expected: SpectralExpected;
  updatedBands?: Record<string, SpectralBandFixture>;
  updatedExpected?: SpectralExpected;
};

export const SPECTRAL_REFERENCE = fixture;
export const SPECTRAL_CASES = fixture.cases as SpectralCase[];

export type SpectralOutputProps = {
  id?: string;
  width: number;
  height: number;
  output: GraphDataView<'float32'>;
  outputValidity: GraphDataView<'uint32'>;
  epsilon?: number;
  clamp?: readonly [number, number];
};

export function createSpectralIndex(
  index: string,
  props: SpectralOutputProps,
  bands: Record<string, GPURasterBufferBand>,
  soilAdjustment?: number
) {
  switch (index) {
    case 'ndwi-mcfeeters':
      return new GPURasterNDWI({
        ...props,
        method: 'mcfeeters',
        green: bands.green,
        nearInfrared: bands.nearInfrared
      });
    case 'ndwi-gao':
      return new GPURasterNDWI({
        ...props,
        method: 'gao',
        nearInfrared: bands.nearInfrared,
        shortwaveInfrared: bands.shortwaveInfrared
      });
    case 'nbr':
      return new GPURasterNBR({
        ...props,
        nearInfrared: bands.nearInfrared,
        shortwaveInfrared2: bands.shortwaveInfrared2
      });
    case 'evi':
      return new GPURasterEVI({
        ...props,
        nearInfrared: bands.nearInfrared,
        red: bands.red,
        blue: bands.blue
      });
    case 'savi':
      return new GPURasterSAVI({
        ...props,
        nearInfrared: bands.nearInfrared,
        red: bands.red,
        soilAdjustment
      });
    default:
      throw new Error(`Unknown spectral fixture index ${index}`);
  }
}
