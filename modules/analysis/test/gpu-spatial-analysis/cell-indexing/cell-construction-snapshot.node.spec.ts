// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  GraphVectorView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPointToCell} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell';
import {GPUCellGeometry} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry';
import {GPUCellMeasures} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {GPUCellTopology} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-topology';
import {GPUCellGridPath} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path';
import {GPUCellCompaction} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-compaction';
import {GPUH3CellIdentities} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-h3-cell-identities';
import {GPUS2CellHierarchy} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover/gpu-cell-cover';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline/gpu-cell-set-outline';
import {GPUHealpixPointToCell} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell';
import {GPUHealpixCellAggregation} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-cell-aggregation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('captures all thirteen owned late-reading cell constructors while retaining live view identities (DGGS-115)', () => {
  const device = createNullWebGPUDevice();
  let serial = 0;
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length = 1
  ) => createTransientView(graph, `snapshot-${serial++}`, format, length);
  const status = () => ({
    count: view('uint32'),
    overflow: view('uint32'),
    requiredCount: view('uint32')
  });
  // Copy configuration records only. Graph data views stay live and must keep identity.
  const copy = (value: unknown): unknown =>
    value instanceof GraphDataView ||
    value instanceof GraphVectorView ||
    !value ||
    typeof value !== 'object'
      ? value
      : Object.fromEntries(Object.entries(value).map(([name, child]) => [name, copy(child)]));
  const mutate = (value: object): void => {
    for (const [name, child] of Object.entries(value)) {
      if (child instanceof GraphDataView || child instanceof GraphVectorView)
        Reflect.set(value, name, undefined);
      else if (child && typeof child === 'object') mutate(child);
      else Reflect.set(value, name, typeof child === 'number' ? NaN : 'changed-after-construction');
    }
  };
  const check = <Props extends object>(
    props: Props,
    create: (props: Props) => GPUCommandNodeProducer & {readonly props: object}
  ) => {
    const expected = copy(props);
    const contributor = create(props);
    expect(contributor.props).not.toBe(props);
    mutate(props);
    expect(contributor.props).toEqual(expected);
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(sources.length).toBeGreaterThan(0);
    expect(sources.every(kernel => !kernel.source.includes('changed-after-construction'))).toBe(
      true
    );
  };
  try {
    check(
      {
        family: 'quadbin' as const,
        resolution: 2,
        positions: view('float32x2'),
        spatialContext: {coordinateSpace: 'longitude-latitude' as const, metric: 'none' as const},
        output: {cells: view('uint32x2'), validity: view('uint32')}
      },
      props => new GPUPointToCell(props)
    );
    check(
      {
        family: 'h3' as const,
        cells: view('uint32x2'),
        maximumVertexCount: 6,
        output: {
          centers: view('float32x2'),
          boundaries: view('float32x2', 6),
          vertexCounts: view('uint32')
        }
      },
      props => new GPUCellGeometry(props)
    );
    check(
      {
        family: 'h3' as const,
        cells: view('uint32x2'),
        maximumEdgeCount: 6,
        precision: 'float32' as const,
        output: {
          areas: view('float32'),
          perimeters: view('float32'),
          edgeLengths: view('float32', 6),
          edgeCounts: view('uint32'),
          pentagons: view('uint32')
        }
      },
      props => new GPUCellMeasures(props)
    );
    check(
      {
        family: 's2' as const,
        operation: {type: 'parent' as const, resolution: 0},
        cells: view('uint32x2'),
        output: {cells: view('uint32x2'), counts: view('uint32')}
      },
      props => new GPUCellTopology(props)
    );
    check(
      {
        family: 'h3' as const,
        origins: view('uint32x2'),
        destinations: view('uint32x2'),
        maximumPathLength: 4,
        output: {cells: view('uint32x2', 4), distances: view('uint32'), counts: view('uint32')}
      },
      props => new GPUCellGridPath(props)
    );
    check(
      {
        family: 'quadbin' as const,
        operation: {type: 'compact' as const, resolution: 1, minimumResolution: 0, sorted: false},
        cells: view('uint32x2', 4),
        output: {cells: view('uint32x2', 4), status: status()}
      },
      props => new GPUCellCompaction(props)
    );
    check(
      {cells: view('uint32x2'), operation: 'vertices' as const, output: view('uint32x2', 6)},
      props => new GPUH3CellIdentities(props)
    );
    check(
      {
        cells: view('uint32x2'),
        operation: 'range' as const,
        output: {cells: view('uint32x2', 2), status: view('uint32')}
      },
      props => new GPUS2CellHierarchy(props)
    );
    check(
      {
        cells: view('uint32x2'),
        precision: 'float32' as const,
        output: {measures: view('float32x2'), status: view('uint32')}
      },
      props => new GPUS2CellMeasures(props)
    );
    check(
      {
        family: 'quadbin' as const,
        resolution: 2,
        positions: view('float32x2', 4),
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 2),
        candidateCapacity: 4,
        output: {
          cells: view('uint32x2', 4),
          featureIds: view('uint32', 4),
          status: {...status(), candidateOverflow: view('uint32')}
        }
      },
      props => new GPUCellCover(props)
    );
    check(
      {
        family: 'h3' as const,
        cells: view('uint32x2'),
        output: {
          cells: view('uint32x2', 8),
          rows: view('uint32', 8),
          edgeIndices: view('uint32', 8),
          endpoints: view('float32x4', 8),
          status: status()
        }
      },
      props => new GPUCellSetOutline(props)
    );
    check(
      {
        order: 0,
        scheme: 'uniq' as const,
        positions: view('float32x2'),
        output: {cells: view('uint32x2'), validity: view('uint32')}
      },
      props => new GPUHealpixPointToCell(props)
    );
    check(
      {
        order: 0,
        scheme: 'uniq' as const,
        cells: view('uint32x2', 2),
        cellCapacity: 2,
        output: {
          cells: view('uint32x2', 2),
          counts: view('uint32', 2),
          status: {...status(), invalidCount: view('uint32')}
        }
      },
      props => new GPUHealpixCellAggregation(props)
    );
  } finally {
    device.destroy();
  }
});
