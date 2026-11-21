// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {validateSinglePackedView} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  GPURasterSkeleton,
  GPURasterMedialAxis
} from '../../../src/raster-analysis/gpu-raster-skeleton';
import {GPURasterRankFilter} from '../../../src/raster-analysis/gpu-raster-rank-filter';
import {GPURasterFillNoData} from '../../../src/raster-analysis/gpu-raster-fill-nodata';
import {
  GPURasterFillHoles,
  GPURasterReconstruction
} from '../../../src/raster-analysis/gpu-raster-reconstruction';
import {GPURasterWatershedSegmentation} from '../../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {GPURasterWalkingCost} from '../../../src/raster-analysis/gpu-raster-walking-cost';
import {GPURasterReprojection} from '../../../src/raster-analysis/gpu-raster-reprojection';
import {GPURasterTileReprojection} from '../../../src/raster-analysis/gpu-raster-tile-reprojection';
import {
  GPURasterFractionalCoverage,
  GPURasterFractionalZonalStatistics
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';

function createFixture(format: 'float32' | 'sint32' | 'uint32' = 'float32') {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  const ports: GraphDataView[] = [];
  const view = <Format extends 'float32' | 'sint32' | 'uint32' | 'float32x2'>(
    format: Format,
    length = 4
  ) => {
    const result = createTransientView(graph, `port-${ports.length}`, format, length);
    ports.push(result);
    return result;
  };
  const band = () =>
    makeGPURasterBand(view(format), grid, {
      id: 'native-band',
      validity: view('uint32')
    });
  const status = () => ({
    converged: view('uint32', 1),
    iterationLimitReached: view('uint32', 1),
    iterationCount: view('uint32', 1)
  });
  const coverage = () => ({
    zoneIds: view('uint32'),
    cellIds: view('uint32'),
    fractions: view('float32'),
    zoneValidity: view('uint32', 1),
    status: {
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    }
  });
  return {device, graph, grid, ports, view, band, status, coverage};
}

type Fixture = ReturnType<typeof createFixture>;
type PreparedContributor = (() => unknown) & {props: Readonly<Record<string, unknown>>};
const cases: {name: string; prepare: (fixture: Fixture) => PreparedContributor}[] = [
  {
    name: 'skeleton',
    prepare: ({grid, view, status}) => {
      const props = {
        grid,
        input: view('uint32'),
        output: {skeleton: view('uint32'), status: status()}
      };
      return Object.assign(() => new GPURasterSkeleton(props), {props});
    }
  },
  {
    name: 'medial axis',
    prepare: ({grid, view}) => {
      const props = {
        grid,
        input: view('uint32'),
        output: {medialAxis: view('uint32'), radius: view('float32')}
      };
      return Object.assign(() => new GPURasterMedialAxis(props), {props});
    }
  },
  {
    name: 'rank filter',
    prepare: ({grid, view, band}) => {
      const props = {
        grid,
        input: band(),
        radius: 1,
        output: {values: view('float32'), validity: view('uint32'), changed: view('uint32')}
      };
      return Object.assign(() => new GPURasterRankFilter(props), {props});
    }
  },
  {
    name: 'fill nodata',
    prepare: ({grid, view, band}) => {
      const props = {
        grid,
        input: band(),
        maximumDistance: 1,
        output: {
          values: view('float32'),
          validity: view('uint32'),
          filled: view('uint32'),
          status: {invalidCount: view('uint32', 1)}
        }
      };
      return Object.assign(() => new GPURasterFillNoData(props), {props});
    }
  },
  {
    name: 'fill holes',
    prepare: ({grid, view, status}) => {
      const props = {
        grid,
        input: view('uint32'),
        output: {values: view('uint32'), status: status()}
      };
      return Object.assign(() => new GPURasterFillHoles(props), {props});
    }
  },
  {
    name: 'reconstruction',
    prepare: ({grid, view, band, status}) => {
      const props = {
        grid,
        marker: band(),
        mask: band(),
        output: {values: view('float32'), validity: view('uint32'), status: status()}
      };
      return Object.assign(() => new GPURasterReconstruction(props), {props});
    }
  },
  {
    name: 'watershed',
    prepare: ({grid, view, band}) => {
      const props = {
        grid,
        input: band(),
        markers: view('uint32'),
        output: {
          labels: view('uint32'),
          status: {converged: view('uint32', 1), iterationLimitReached: view('uint32', 1)}
        }
      };
      return Object.assign(() => new GPURasterWatershedSegmentation(props), {props});
    }
  },
  {
    name: 'walking cost',
    prepare: ({grid, view, band, status}) => {
      const props = {
        grid,
        elevation: band(),
        friction: band(),
        sources: view('uint32'),
        parameters: view('float32', 16),
        output: {
          costs: view('float32'),
          validity: view('uint32'),
          status: {...status(), invalidCount: view('uint32', 1)}
        }
      };
      return Object.assign(() => new GPURasterWalkingCost(props), {props});
    }
  },
  {
    name: 'reprojection',
    prepare: ({grid, view, band}) => {
      const props = {
        sourceGrid: grid,
        destinationGrid: grid,
        input: band(),
        parameters: view('float32', 20),
        output: {values: view('float32'), validity: view('uint32')}
      };
      return Object.assign(() => new GPURasterReprojection(props), {props});
    }
  },
  {
    name: 'tile reprojection',
    prepare: ({grid, view, band}) => {
      const props = {
        sourceGrid: grid,
        destinationGrid: grid,
        sources: [{pixelBounds: [0, 0, 2, 2] as const, input: band()}],
        parameters: view('float32', 20),
        output: {values: view('float32'), validity: view('uint32')}
      };
      return Object.assign(() => new GPURasterTileReprojection(props), {props});
    }
  },
  {
    name: 'fractional coverage',
    prepare: ({grid, view, coverage}) => {
      const props = {
        grid,
        zones: {
          positions: view('float32x2'),
          featureOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          ringOffsets: view('uint32', 2)
        },
        parameters: view('float32', 8),
        output: coverage()
      };
      return Object.assign(() => new GPURasterFractionalCoverage(props), {props});
    }
  },
  {
    name: 'fractional zonal',
    prepare: ({grid, view, band, coverage}) => {
      const props = {
        grid,
        input: band(),
        zoneCount: 1,
        coverage: coverage(),
        quantity: 'intensive' as const,
        parameters: view('float32', 8),
        cellAreas: view('float32'),
        output: {
          totals: view('float32', 1),
          means: view('float32', 1),
          coveredArea: view('float32', 1),
          validArea: view('float32', 1),
          validity: view('uint32', 1)
        }
      };
      return Object.assign(() => new GPURasterFractionalZonalStatistics(props), {props});
    }
  },
  {
    name: 'fractional categorical',
    prepare: ({grid, view, coverage}) => {
      const props = {
        grid,
        input: {
          id: 'classes',
          format: 'uint32' as const,
          storage: {kind: 'buffer' as const, values: view('uint32')},
          validity: view('uint32')
        },
        zoneCount: 1,
        categoryCapacity: 2,
        coverage: coverage(),
        output: {
          coverageSums: view('float32', 2),
          classAreas: view('float32', 2),
          proportions: view('float32', 2),
          coveredArea: view('float32', 1),
          validArea: view('float32', 1),
          missingArea: view('float32', 1),
          majority: view('uint32', 1),
          minority: view('uint32', 1),
          validity: view('uint32', 1)
        }
      };
      return Object.assign(() => new GPURasterFractionalCategoricalStatistics(props), {props});
    }
  }
];

for (const format of ['sint32', 'uint32'] as const) {
  for (const tiled of [false, true]) {
    cases.push({
      name: `${tiled ? 'tile reprojection' : 'reprojection'} native ${format} labels`,
      prepare: ({grid, view}) => {
        const input = makeGPURasterBand(view(format), grid);
        const props = {
          sourceGrid: grid,
          destinationGrid: grid,
          method: 'nearest' as const,
          output: {values: view('float32'), validity: view('uint32'), labels: view(format)}
        };
        if (tiled) {
          const tiledProps = {...props, sources: [{input, pixelBounds: [0, 0, 2, 2] as const}]};
          return Object.assign(() => new GPURasterTileReprojection(tiledProps), {
            props: tiledProps
          });
        }
        const singleProps = {...props, input};
        return Object.assign(() => new GPURasterReprojection(singleProps), {props: singleProps});
      }
    });
  }
}

function getPortReferences(
  object: Readonly<Record<string, unknown>>,
  target: GraphDataView,
  prefix = ''
): {parent: object; key: string; path: string}[] {
  const references: {parent: object; key: string; path: string}[] = [];
  for (const [key, value] of Object.entries(object)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value === target) references.push({parent: object, key, path});
    else if (
      value &&
      typeof value === 'object' &&
      (Array.isArray(value) || Object.getPrototypeOf(value) === Object.prototype)
    )
      references.push(...getPortReferences(value as Record<string, unknown>, target, path));
  }
  return references;
}

for (const entry of cases) {
  it(`${entry.name} rejects wrong scalar and vector formats on every input and output leaf`, () => {
    const fixture = createFixture();
    try {
      const construct = entry.prepare(fixture);
      expect(construct).not.toThrow();
      for (const [index, port] of [...fixture.ports].entries()) {
        const original = port.format;
        const references = getPortReferences(construct.props, port);
        expect(references.length).toBeGreaterThan(0);
        // Allocate actual packed rows in the wrong format; never retag a scalar's stride.
        for (const wrong of [
          original === 'uint32' ? 'float32' : 'uint32',
          original === 'float32x2' ? 'uint32x2' : 'float32x2'
        ] as const) {
          const replacement = createTransientView(
            fixture.graph,
            `wrong-${index}-${wrong}`,
            wrong,
            port.length
          );
          // Control: the former self-format check accepts this genuine packed replacement.
          expect(() =>
            validateSinglePackedView(replacement, [wrong], 'old-check-control')
          ).not.toThrow();
          for (const {parent, key} of references) Reflect.set(parent, key, replacement);
          expect(construct, `${entry.name} port ${index}, ${original} -> ${wrong}`).toThrow(
            new RegExp(references[0].path.replace(/\./g, '\\.'))
          );
          for (const {parent, key} of references) Reflect.set(parent, key, port);
        }
      }
    } finally {
      fixture.device.destroy();
    }
  });
}

for (const format of ['float32', 'sint32', 'uint32'] as const) {
  it(`retains valid native ${format} raster bands`, () => {
    const fixture = createFixture(format);
    try {
      for (const entry of cases.filter(entry =>
        [
          'rank filter',
          'fill nodata',
          'reconstruction',
          'watershed',
          'walking cost',
          'reprojection',
          'tile reprojection',
          'fractional zonal'
        ].includes(entry.name)
      )) {
        expect(entry.prepare(fixture), entry.name).not.toThrow();
      }
    } finally {
      fixture.device.destroy();
    }
  });
}

it('band wrapper rejects malformed formats before borrowing a prefix', () => {
  const fixture = createFixture();
  try {
    const values = fixture.view('float32', 8);
    const validity = fixture.view('uint32', 8);
    expect(() => makeGPURasterBand(fixture.view('float32', 3), fixture.grid)).toThrow(
      /at least one value per grid pixel/
    );
    for (const wrong of ['float32x2', 'uint32x2'] as const) {
      const replacement = createTransientView(fixture.graph, `wrong-values-${wrong}`, wrong, 8);
      expect(() =>
        validateSinglePackedView(replacement, [wrong], 'old-check-control')
      ).not.toThrow();
      expect(() =>
        Reflect.apply(makeGPURasterBand, undefined, [replacement, fixture.grid])
      ).toThrowError(expect.objectContaining({path: 'values', code: 'VIEW_FORMAT'}));
    }
    for (const wrong of ['float32', 'float32x2'] as const) {
      const options = {validity};
      Reflect.set(
        options,
        'validity',
        createTransientView(fixture.graph, `wrong-validity-${wrong}`, wrong, 8)
      );
      expect(() => makeGPURasterBand(values, fixture.grid, options)).toThrowError(
        expect.objectContaining({path: 'validity', code: 'VIEW_FORMAT'})
      );
    }
  } finally {
    fixture.device.destroy();
  }
});

it('band wrapper preserves native values, offsets, exact view identity and graph ownership', () => {
  const fixture = createFixture();
  const foreign = createFixture();
  try {
    for (const format of ['float32', 'sint32', 'uint32'] as const) {
      const allocation = fixture.view(format, 80);
      const values = fixture.graph.createDataView(allocation.buffer, {
        format,
        byteOffset: 256,
        length: 8
      });
      const validityAllocation = fixture.view('uint32', 80);
      const validity = fixture.graph.createDataView(validityAllocation.buffer, {
        format: 'uint32',
        byteOffset: 256,
        length: 8
      });
      const result = makeGPURasterBand(values, fixture.grid, {validity});
      expect(result.storage.values.buffer).toBe(values.buffer);
      expect(result.storage.values.byteOffset).toBe(256);
      expect(result.storage.values.length).toBe(4);
      expect(result.validity?.byteOffset).toBe(256);
      const exact = makeGPURasterBand(result.storage.values, fixture.grid, {
        validity: result.validity
      });
      expect(exact.storage.values).toBe(result.storage.values);
      expect(exact.validity).toBe(result.validity);
      expect(() =>
        makeGPURasterBand(values, fixture.grid, {validity: foreign.view('uint32')})
      ).toThrow(/same graph/);
    }
  } finally {
    fixture.device.destroy();
    foreign.device.destroy();
  }
});
