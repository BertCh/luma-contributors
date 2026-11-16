// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createElement} from 'react';
import {createRoot} from 'react-dom/client';
import {test, vi} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {getCatalogBinding} from '../src/analysis/catalog-bindings';
import {
  AnalysisSensitivitySession,
  type AnalysisSensitivityProvenance
} from '../src/analysis/analysis-sensitivity';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {AnalysisSensitivityStrip} from '../../../examples/deck/kepler-compute/src/analysis-sensitivity-strip';

function report(condition: unknown, message: string, details: unknown): asserts condition {
  if (!condition)
    throw new Error(`AX-42 sensitivity: ${message}\n${JSON.stringify(details, null, 2)}`);
}
const INPUTS = {
  points: {
    type: 'FeatureCollection' as const,
    features: [
      {id: 'first', geometry: {type: 'Point', coordinates: [0, 0]}},
      {id: 'second', geometry: {type: 'Point', coordinates: [10, 10]}}
    ]
  }
};
const PARAMETERS = {coordinateSystem: 'planar', radius: 2, segmentsPerQuarter: 8};

test('six actual maps use one compiled radius graph within admission and retain the chosen setting', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('geometry-point-buffer')!;
  const plan = binding.planPreview!(INPUTS, PARAMETERS);
  const budget = plan.gpuByteLength + plan.cpuByteLength * 8;
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: budget});
  const compilation = vi.spyOn(GPUCommandGraph.prototype, 'compileAsync');
  const gpuDevice = device.handle as GPUDevice;
  const createBuffer = gpuDevice.createBuffer.bind(gpuDevice);
  let liveGpuBytes = 0;
  let maximumGpuBytes = 0;
  const destroySpies: {mockRestore(): void}[] = [];
  const allocation = vi.spyOn(gpuDevice, 'createBuffer').mockImplementation(properties => {
    const buffer = createBuffer(properties);
    const destroy = buffer.destroy.bind(buffer);
    liveGpuBytes += buffer.size;
    maximumGpuBytes = Math.max(maximumGpuBytes, liveGpuBytes);
    let released = false;
    destroySpies.push(
      vi.spyOn(buffer, 'destroy').mockImplementation(() => {
        if (!released) {
          released = true;
          liveGpuBytes -= buffer.size;
        }
        destroy();
      })
    );
    report(
      liveGpuBytes <= registry.memoryUsage.reservedBytes.gpu,
      'GPU allocation exceeded admission including staging',
      {liveGpuBytes, memory: registry.memoryUsage}
    );
    return buffer;
  });
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  let chosen: AnalysisSensitivityProvenance | undefined;
  try {
    root.render(
      createElement(AnalysisSensitivityStrip, {
        control: {
          binding,
          device,
          registry,
          parameters: PARAMETERS,
          revisionKey: 'points@1:all',
          readInputs: async () => INPUTS,
          isCurrent: () => true,
          onChoose: provenance => {
            chosen = provenance;
          }
        }
      })
    );
    await vi.waitFor(() =>
      report(
        container.querySelector('[data-testid="ax-sensitivity-open"]'),
        'exploration control did not mount',
        {html: container.innerHTML}
      )
    );
    (container.querySelector('[data-testid="ax-sensitivity-open"]') as HTMLButtonElement).click();
    await vi.waitFor(
      () =>
        report(
          container.querySelectorAll('[data-testid="ax-sensitivity-map"]').length === 6,
          'six maps did not render',
          {html: container.innerHTML}
        ),
      {timeout: 60000}
    );
    const figures = Array.from(container.querySelectorAll('[data-testid="ax-sensitivity-map"]'));
    const viewBoxes = figures.map(figure => figure.querySelector('svg')?.getAttribute('viewBox'));
    report(new Set(viewBoxes).size === 1, 'map extents differ', {viewBoxes});
    report(
      new Set(figures.map(figure => figure.querySelector('path')?.getAttribute('d'))).size === 6,
      'uniform changes did not generate six geometries',
      {html: container.innerHTML}
    );
    for (const figure of figures) {
      const radius = Number(figure.getAttribute('data-radius'));
      const path = figure.querySelector('path')?.getAttribute('d') ?? '';
      const vertices = Array.from(path.matchAll(/[ML]([^, ]+),([^ ]+)/g), match => [
        Number(match[1]),
        Number(match[2])
      ]);
      // Bearings start north and increase clockwise; SVG negates the northing. Check
      // cardinal and diagonal points plus closure against an independent circle oracle.
      const vertexCount = PARAMETERS.segmentsPerQuarter * 4;
      report(vertices.length === vertexCount + 1, 'ring was not closed', {radius, vertices});
      for (const vertexIndex of [0, 4, 8, 12, 16, 20, 24, 28, 32]) {
        const bearing = (2 * Math.PI * vertexIndex) / vertexCount;
        const expected = [radius * Math.sin(bearing), -radius * Math.cos(bearing)];
        const actual = vertices[vertexIndex];
        report(
          actual &&
            Math.abs(actual[0] - expected[0]) < 1e-4 &&
            Math.abs(actual[1] - expected[1]) < 1e-4,
          'GPU vertex differs from north-first circle oracle',
          {radius, vertexIndex, actual, expected, path}
        );
      }
      report(figure.querySelectorAll('path').length === 2, 'map dropped a source point', {
        radius,
        html: figure.innerHTML
      });
    }
    report(compilation.mock.calls.length === 1, 'six views recompiled', {
      compilations: compilation.mock.calls.length
    });
    report(registry.memoryUsage.totalBytes === budget, 'retained views/output were not admitted', {
      budget,
      memory: registry.memoryUsage
    });
    report(
      container.textContent?.includes('Exploratory parameter search') &&
        container.textContent.includes('No significance classes'),
      'exploration/inference warning is missing',
      {text: container.textContent}
    );
    (figures[3].querySelector('button') as HTMLButtonElement).click();
    report(
      chosen?.chosenValue === Number(figures[3].getAttribute('data-radius')) &&
        chosen?.values.length === 6 &&
        chosen.population.scope === 'full-population' &&
        chosen.population.rowCount === 2,
      'choice/population was not recorded',
      {chosen}
    );
  } finally {
    root.unmount();
    await vi.waitFor(
      () =>
        report(
          registry.memoryUsage.totalBytes === 0,
          'unmount leaked reservation',
          registry.memoryUsage
        ),
      {timeout: 10000}
    );
    container.remove();
    allocation.mockRestore();
    compilation.mockRestore();
    for (const destroy of destroySpies) destroy.mockRestore();
  }
  report(maximumGpuBytes > 0 && liveGpuBytes === 0, 'GPU storage leaked', {
    maximumGpuBytes,
    liveGpuBytes
  });
}, 90000);

test('sampled and viewport GPU sweeps expose no significance attributes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('geometry-point-buffer')!;
  const plan = binding.planPreview!(INPUTS, PARAMETERS);
  for (const scope of ['sample', 'viewport'] as const) {
    const registry = new KeplerDatasetRegistry({
      memoryBudgetBytes: plan.gpuByteLength + plan.cpuByteLength * 8
    });
    const session = new AnalysisSensitivitySession({
      binding,
      device,
      registry,
      inputs: INPUTS,
      parameters: PARAMETERS,
      parameter: 'radius',
      values: [1, 2, 3, 4, 5, 6],
      revisionKey: 'points@1:sample',
      isCurrent: () => true,
      population: {scope, rowCount: 2, sourceRowCount: 100, selectionKey: 'sample-42'}
    });
    try {
      const frames = await session.run();
      report(
        frames?.length === 6 &&
          frames.every(frame =>
            frame.result.tables.every(table => Object.keys(table.columns).join(',') === 'geometry')
          ),
        'partial-population output exposed nongeometry attributes',
        {scope, frames}
      );
    } finally {
      await session.destroy();
    }
    report(registry.memoryUsage.totalBytes === 0, 'sampled sweep leaked memory', {
      scope,
      memory: registry.memoryUsage
    });
  }
}, 90000);
