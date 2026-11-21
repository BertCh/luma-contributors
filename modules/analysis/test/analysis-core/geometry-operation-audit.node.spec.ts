// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCPUVector, GPUGraphVectorFormat, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {GPUPolygonTriangulationOperation} from '../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
import {CPUPolygonTriangulationOperation} from '../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {GPUPolygonTriangulation} from '../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {GPUConstrainedDelaunayTriangulationOperation} from '../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation-operation';
import {CPUConstrainedDelaunayTriangulationOperation} from '../../src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation';
import {GPUConstrainedDelaunayTriangulation} from '../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation';
import {GPUDelaunayTessellationOperation} from '../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation';
import {CPUDelaunayTessellationOperation} from '../../src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation';
import {GPUDelaunayTessellation} from '../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {GPUMinimumBoundsOperation} from '../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import {CPUMinimumBoundsOperation} from '../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation';
import {GPUMinimumBounds} from '../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds';
import {auditKernelCase, getKernelAuditResultFailures} from './kernel-audit/kernel-audit-runner';
import {selectAnalysisCases} from './kernel-audit/analysis-contributors';
import type {KernelAuditCase} from './kernel-audit/kernel-audit-context';

const polygonProps = {
  positions: 'positions',
  ringOffsets: 'rings',
  polygonOffsets: 'polygons',
  featureOffsets: 'features',
  polygonCount: 1,
  indexCapacity: 6,
  largePolygonMethod: 'none' as const,
  output: {
    indices: 'indices',
    valid: 'valid',
    triangleCount: 'triangleCount',
    triangleOffsets: 'triangleOffsets'
  }
};
const constrainedProps = {
  ...polygonProps,
  maximumFlips: 3,
  output: {...polygonProps.output, status: {converged: 'converged'}}
};
const delaunayProps = {
  positions: 'positions',
  capacity: 2,
  output: {
    triangles: 'triangles',
    status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
  }
};
const boundsProps = {
  positions: 'positions',
  labels: 'labels',
  rowCount: 3,
  groupCount: 1,
  maximumHullVertices: 16,
  output: {
    sourceBounds: 'bounds',
    circles: 'circles',
    hullSizes: 'hullSizes',
    status: {overflow: 'overflow', groupOverflow: 'groupOverflow'}
  }
};

// The four pairs in the census. Each GPU operation delegates to the named contributor audited
// below; these are operation factories, not a claim that the contributor's old case covered them.
const FORMS = [
  {
    name: 'polygon triangulation',
    gpu: () => new GPUPolygonTriangulationOperation(polygonProps),
    cpu: () => new CPUPolygonTriangulationOperation(polygonProps),
    contributor: GPUPolygonTriangulation
  },
  {
    name: 'constrained Delaunay triangulation',
    gpu: () => new GPUConstrainedDelaunayTriangulationOperation(constrainedProps),
    cpu: () => new CPUConstrainedDelaunayTriangulationOperation(constrainedProps),
    contributor: GPUConstrainedDelaunayTriangulation
  },
  {
    name: 'Delaunay tessellation',
    gpu: () => new GPUDelaunayTessellationOperation(delaunayProps),
    cpu: () => new CPUDelaunayTessellationOperation(delaunayProps),
    contributor: GPUDelaunayTessellation
  },
  {
    name: 'minimum bounds',
    gpu: () => new GPUMinimumBoundsOperation(boundsProps),
    cpu: () => new CPUMinimumBoundsOperation(boundsProps),
    contributor: GPUMinimumBounds
  }
];

function auditGeometryOperation(form: (typeof FORMS)[number], operation = form.gpu()) {
  const auditCase: KernelAuditCase = {
    // Use the delegate's public contributor key so selectAnalysisCases cannot silently drop it.
    exportName: form.contributor.name,
    name: `${operation.constructor.name} operation factory`,
    build(context) {
      const fixtureViews: Record<string, GraphDataView> = {
        positions: context.view('float32x2', 3),
        rings: context.view('uint32', 2),
        polygons: context.view('uint32', 2),
        features: context.view('uint32', 2),
        labels: context.view('uint32', 3)
      };
      const views = new Map<string, GraphDataView>(
        Object.entries(operation.inputs).map(([port, value]) => {
          const view = fixtureViews[value];
          if (!view) throw new Error(`Missing fixture graph value ${value}`);
          return [port, view];
        })
      );
      for (const [port, output] of Object.entries(operation.outputs)) {
        views.set(port, context.view(output.format, output.length));
      }
      const contributor = operation.createContributor(
        <Format extends GPUGraphVectorFormat>(port: string) => {
          const view = views.get(port);
          if (!view) throw new Error(`Missing operation port ${port}`);
          return view as GraphDataView<Format>;
        }
      );
      expect(contributor).toBeInstanceOf(form.contributor);
      context.graph.add(contributor);
    }
  };
  expect(selectAnalysisCases([auditCase])).toEqual([auditCase]);
  const result = auditKernelCase(auditCase);
  operation.destroy();
  return result;
}

it.each(FORMS)('GEO-129 $name GPU operation delegates and emits audited modules', form => {
  const result = auditGeometryOperation(form);
  expect(getKernelAuditResultFailures(result)).toEqual([]);
  expect(result.rejected).toBe(false);
  expect(result.nodeCount).toBeGreaterThan(0);
  expect(result.moduleCount).toBeGreaterThan(0);
});

function rows(format: GPUGraphVectorFormat, values: Float32Array | Uint32Array): GPUCPUVector {
  const length = values.length / getGPUVectorFormatInfo(format).components;
  return {format, length, chunks: [{values, length}], pack: () => values.slice()};
}

function getCPUFixtureInputs(declarations: Readonly<Record<string, string>>) {
  const fixtures: Record<string, GPUCPUVector> = {
    positions: rows('float32x2', Float32Array.of(0, 0, 2, 0, 0, 2)),
    rings: rows('uint32', Uint32Array.of(0, 3)),
    polygons: rows('uint32', Uint32Array.of(0, 1)),
    features: rows('uint32', Uint32Array.of(0, 1)),
    labels: rows('uint32', Uint32Array.of(0, 0, 0))
  };
  return Object.fromEntries(
    Object.entries(declarations).map(([port, value]) => {
      if (!fixtures[value]) throw new Error(`Missing fixture graph value ${value}`);
      return [port, fixtures[value]];
    })
  );
}

function validateCPUOutput(values: unknown, format: GPUGraphVectorFormat, length: number) {
  expect(ArrayBuffer.isView(values)).toBe(true);
  if (!ArrayBuffer.isView(values)) throw new Error('Expected packed CPU output');
  const ValueType = format.startsWith('float32')
    ? Float32Array
    : format.startsWith('sint32')
      ? Int32Array
      : Uint32Array;
  expect(values).toBeInstanceOf(ValueType);
  const numbers = Array.from(values as Float32Array | Uint32Array);
  expect(numbers.every(Number.isFinite)).toBe(true);
  expect(numbers).toHaveLength(length * getGPUVectorFormatInfo(format).components);
}

it('review S1 a missing GPU input declaration cannot certify its factory', () => {
  const form = FORMS[0];
  const operation = form.gpu();
  Reflect.set(
    operation,
    'inputs',
    Object.fromEntries(Object.entries(operation.inputs).filter(([port]) => port !== 'positions'))
  );
  expect(getKernelAuditResultFailures(auditGeometryOperation(form, operation))).not.toEqual([]);
});

it('review S1 a missing CPU input declaration cannot reach a hidden fixture input', async () => {
  const operation = FORMS[0].cpu();
  Reflect.set(
    operation,
    'inputs',
    Object.fromEntries(Object.entries(operation.inputs).filter(([port]) => port !== 'positions'))
  );
  await expect(
    Promise.resolve().then(() => operation.execute({inputs: getCPUFixtureInputs(operation.inputs)}))
  ).rejects.toThrow(/positions/);
});

it('review S1 float counts cannot certify a uint32 output contract', () => {
  expect(() => validateCPUOutput(Float32Array.of(1), 'uint32', 1)).toThrow();
});

it.each(
  FORMS
)('GEO-129 $name CPU operation honors its ports and literal triangle contract', async form => {
  const gpu = form.gpu();
  const cpu = form.cpu();
  expect(cpu.inputs).toEqual(gpu.inputs);
  expect(cpu.contract.outputs).toEqual(gpu.contract.outputs);
  const inputs = getCPUFixtureInputs(cpu.inputs);
  const result = await cpu.execute({inputs});
  const cpuOutputs = cpu.contract.outputs;
  if (!cpuOutputs || typeof cpuOutputs === 'function')
    throw new Error('Expected static CPU outputs');
  expect(Object.keys(result).sort()).toEqual(Object.keys(cpuOutputs).sort());
  for (const output of Object.values(gpu.outputs)) {
    const values = result[output.value];
    validateCPUOutput(values, output.format, output.length);
  }
  const values = (name: string) => Array.from(result[name] as Float32Array | Uint32Array);
  // Hand-worked right triangle: one triangle on vertices {0,1,2}; bounds [0,0,2,2];
  // circumcenter (1,1), radius sqrt(2). Neither expectation uses the contributor as an oracle.
  if (form.name === 'minimum bounds') {
    expect(values('bounds')).toEqual([0, 0, 2, 2]);
    expect(values('circles')).toEqual([1, 1, Math.fround(Math.SQRT2)]);
    expect(values('hullSizes')).toEqual([3]);
    expect(values('overflow')).toEqual([0]);
    expect(values('groupOverflow')).toEqual([0]);
  } else if (form.name === 'Delaunay tessellation') {
    expect(values('triangles').slice(0, 3).sort()).toEqual([0, 1, 2]);
    expect(values('count')).toEqual([1]);
    expect(values('requiredCount')).toEqual([1]);
    expect(values('overflow')).toEqual([0]);
  } else {
    expect(values('indices').slice(0, 3).sort()).toEqual([0, 1, 2]);
    expect(values('valid')).toEqual([1]);
    expect(values('triangleCount')).toEqual([1]);
    expect(values('triangleOffsets')).toEqual([0, 1]);
    if (form.name === 'constrained Delaunay triangulation')
      expect(values('converged')).toEqual([1]);
  }
  gpu.destroy();
});
