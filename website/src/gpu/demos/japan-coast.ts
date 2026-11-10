// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import endpoints from '../data/japan-coast.json';
import {projectRows} from '../data/point-observations';
import {createStorageBuffer, SegmentListLayer} from './kit';

export function createJapanCoast(device: Device, worldWidth: number) {
  const coordinates = projectRows(
    endpoints.map(([longitude, latitude]) => ({longitude, latitude})),
    [126, 28, 146, 42],
    {width: worldWidth, height: 1}
  );
  const segmentCount = endpoints.length / 2;
  const segments = createStorageBuffer(device, coordinates);
  const count = createStorageBuffer(device, Uint32Array.of(segmentCount));
  const layer = new SegmentListLayer(device, segments, count, segmentCount);
  return {
    layer,
    destroy() {
      layer.destroy();
      segments.destroy();
      count.destroy();
    }
  };
}
