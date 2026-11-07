// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUDorlingCartogram} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {S2CaptureSpec, S2Uploads} from './s2-lens-run';

/** Two coincident circles with analytically known radii and separation. */
export async function buildDorlingCartogramScenario(
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Promise<Record<string, S2CaptureSpec>> {
  const positions = uploads.view('initial-centers', new Float32Array([0, 0, 0, 0]), 'float32x2');
  const values = uploads.view('values', new Float32Array([Math.PI, Math.PI]), 'float32');
  const centers = uploads.view('centers', new Float32Array(4), 'float32x2');
  const radii = uploads.view('radii', new Float32Array(2), 'float32');
  const sourceIds = uploads.view('source-ids', new Uint32Array(2), 'uint32');
  const metrics = uploads.view('metrics', new Float32Array(2), 'float32x2');
  const status = uploads.view('status', new Uint32Array(4), 'uint32');
  graph.add(
    new GPUDorlingCartogram({
      positions,
      values,
      maximumIterations: 8,
      output: {positions: centers, radii, sourceIds, metrics, status}
    })
  );
  return {
    positions: {view: positions, role: 'input'},
    values: {view: values, role: 'input'},
    centers: {view: centers, role: 'output'},
    radii: {view: radii, role: 'output'},
    sourceIds: {view: sourceIds, role: 'output'},
    metrics: {view: metrics, role: 'output'},
    status: {view: status, role: 'output'}
  };
}
