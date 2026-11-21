// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUNetworkIsochrones,
  type GPUNetworkIsochronesProps
} from '../../../src/gpu-network/network-isochrones/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getResourceBuffer} from '../network-resource-buffers';

function createProps(graph: GPUCommandGraph): GPUNetworkIsochronesProps {
  return {
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 3),
      weights: createTransientView(graph, 'weights', 'float32', 3),
      nodePositions: createTransientView(graph, 'positions', 'float32x2', 4)
    },
    costs: createTransientView(graph, 'costs', 'float32', 4),
    startNodes: createTransientView(graph, 'starts', 'uint32', 1),
    maximumIterations: 1,
    costBreaks: createTransientView(graph, 'breaks', 'float32', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 12),
    raster: {
      width: 2,
      height: 2,
      output: {bandClasses: createTransientView(graph, 'bands', 'uint32', 4)}
    }
  };
}

it.each([
  false,
  true
])('retains the raster output identity supplied as an inherited getter: %s', inherited => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const values = createTransientView(graph, 'caller-values', 'float32', 4);
    const output = inherited
      ? Object.create({
          get values() {
            return values;
          }
        })
      : {values};
    Object.assign(output, {bandClasses: props.raster!.output.bandClasses});
    const contributor = new GPUNetworkIsochrones({
      ...props,
      raster: {...props.raster!, output}
    });
    expect(contributor.props.raster!.output.values).toBe(values);
    const decode = contributor.getCommandNodes(graph).find(node => node.id.endsWith('-decode'))!;
    expect(decode.resources!.map(getResourceBuffer)).toContain(values);
  } finally {
    device.destroy();
  }
});

for (const byteOffset of [0, 4, 16, 256]) {
  it(`checks logical cost/raster output overlap at byte offset ${byteOffset}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const props = createProps(graph);
      const packed = graph.createTransientBuffer({
        id: 'packed',
        byteLength: 512,
        usage: Buffer.STORAGE
      });
      const costs = graph.createDataView(packed, {format: 'float32', length: 4});
      const values = graph.createDataView(packed, {format: 'float32', length: 4, byteOffset});
      const create = () =>
        new GPUNetworkIsochrones({
          ...props,
          costs,
          raster: {...props.raster!, output: {...props.raster!.output, values}}
        });
      if (byteOffset < 16) {
        expect(create).toThrow(/network-isochrones.*costs.*raster.*values.*overlap/);
      } else {
        expect(() => create().getCommandNodes(graph)).not.toThrow();
      }
    } finally {
      device.destroy();
    }
  });
}

function createCellProps(graph: GPUCommandGraph): GPUNetworkIsochronesProps {
  return {
    ...createProps(graph),
    raster: undefined,
    assignments: createTransientView(graph, 'assignments', 'uint32', 4),
    cellOutline: {
      family: 'quadbin',
      resolution: 10,
      byFacility: true,
      tableCapacity: 3,
      table: {
        cells: createTransientView(graph, 'table-cells', 'uint32x2', 2),
        counts: createTransientView(graph, 'table-counts', 'uint32', 2),
        status: {
          count: createTransientView(graph, 'table-count', 'uint32', 1),
          overflow: createTransientView(graph, 'table-overflow', 'uint32', 1),
          candidateOverflow: createTransientView(graph, 'table-candidate-overflow', 'uint32', 1)
        }
      },
      output: {
        rows: createTransientView(graph, 'outline-rows', 'uint32', 16),
        cells: createTransientView(graph, 'outline-cells', 'uint32x2', 16),
        edgeIndices: createTransientView(graph, 'outline-edges', 'uint32', 16),
        endpoints: createTransientView(graph, 'outline-endpoints', 'float32x4', 16),
        groups: createTransientView(graph, 'outline-groups', 'uint32', 16),
        status: {
          count: createTransientView(graph, 'outline-count', 'uint32', 1),
          overflow: createTransientView(graph, 'outline-overflow', 'uint32', 1)
        }
      }
    }
  };
}

it('sizes default facility scratch from the supplied cell table', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createCellProps(graph);
    const create = vi.spyOn(graph, 'createTransientBuffer');
    const contributor = new GPUNetworkIsochrones(props);
    expect(contributor.props.cellOutline!.table!.status.candidateOverflow).toBe(
      props.cellOutline!.table!.status.candidateOverflow
    );
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    for (const name of ['cell-facilities', 'cell-minimum-costs']) {
      const descriptor = create.mock.calls.find(
        ([value]) => value.id === `network-isochrones-${name}`
      )?.[0];
      expect(descriptor?.byteLength).toBe(8);
    }
  } finally {
    device.destroy();
  }
});

it('validates caller facility rows against the supplied table at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createCellProps(graph);
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          cellOutline: {
            ...props.cellOutline!,
            cellFacilities: createTransientView(graph, 'wrong-facilities', 'uint32', 4)
          }
        })
    ).toThrow(/network-isochrones.*cellOutline.cellFacilities.*2/);
  } finally {
    device.destroy();
  }
});

it('rejects an 8192 squared encoded raster before scratch or search declaration', () => {
  const device = createNullWebGPUDevice();
  Object.assign(device.limits, {maxStorageBufferBindingSize: 128 * 1024 * 1024});
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const contributor = new GPUNetworkIsochrones({
      ...props,
      raster: {...props.raster!, width: 8192, height: 8192}
    });
    const create = vi.spyOn(graph, 'createTransientBuffer');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /network-isochrones.*encoded.*268435456/
    );
    expect(create).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('rejects oversized cell table scratch before starting the search', () => {
  const device = createNullWebGPUDevice();
  Object.assign(device.limits, {maxStorageBufferBindingSize: 128 * 1024 * 1024});
  const graph = new GPUCommandGraph(device);
  try {
    const props = createCellProps(graph);
    const contributor = new GPUNetworkIsochrones({
      ...props,
      cellOutline: {...props.cellOutline!, table: undefined, tableCapacity: 20_000_000}
    });
    const create = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /network-isochrones.*cellOutline table.cells.*160000000/
    );
    expect(create).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

for (const cellPath of [false, true]) {
  it(`rejects non-STORAGE caller views before scratch (cell path: ${cellPath})`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const props = cellPath ? createCellProps(graph) : createProps(graph);
      const contributor = new GPUNetworkIsochrones(props);
      Object.defineProperty(props.costBreaks.buffer, 'usage', {value: Buffer.COPY_DST});
      const create = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /network-isochrones.*costBreaks.*STORAGE/
      );
      expect(create).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });
}

it('rejects cost aliases in class-backed raster output maps', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    class RasterOutput {
      bandClasses = props.raster!.output.bandClasses;
      values = props.costs;
    }
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          raster: {...props.raster!, output: new RasterOutput()}
        })
    ).toThrow(/costs.*raster.output.values.*overlap/);
  } finally {
    device.destroy();
  }
});

it('keeps the validated decode view after caller port-map mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const values = createTransientView(graph, 'original-values', 'float32', 4);
    props.raster!.output.values = values;
    const contributor = new GPUNetworkIsochrones(props);
    props.raster!.output.values = props.costs;
    const decode = contributor.getCommandNodes(graph).find(node => node.id.endsWith('-decode'))!;
    const writers = decode
      .resources!.filter(resource => resource.usage === 'storage-read-write')
      .map(getResourceBuffer);
    expect(writers).toContain(values);
    expect(writers).not.toContain(props.costs);
  } finally {
    device.destroy();
  }
});

for (const byteOffset of [16, 256]) {
  it(`checks co-bound labelled decode outputs at byte offset ${byteOffset}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const props = createProps(graph);
      const packed = graph.createTransientBuffer({
        id: 'decode-packed',
        byteLength: 512,
        usage: Buffer.STORAGE
      });
      const values = graph.createDataView(packed, {format: 'float32', length: 4});
      const pixelFacilities = graph.createDataView(packed, {
        format: 'uint32',
        length: 4,
        byteOffset
      });
      const create = () =>
        new GPUNetworkIsochrones({
          ...props,
          assignments: createTransientView(graph, `decode-assignments-${byteOffset}`, 'uint32', 4),
          raster: {...props.raster!, output: {...props.raster!.output, values, pixelFacilities}}
        });
      if (byteOffset === 16) expect(create).toThrow(/aligned binding byte ranges must not overlap/);
      else expect(() => graph.add(create())).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}

it('rejects cell-facility output sharing a physical buffer with table input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createCellProps(graph);
    const packed = graph.createTransientBuffer({
      id: 'facility-packed',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const cells = graph.createDataView(packed, {format: 'uint32x2', length: 2});
    const cellFacilities = graph.createDataView(packed, {
      format: 'uint32',
      length: 2,
      byteOffset: 256
    });
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          cellOutline: {
            ...props.cellOutline!,
            table: {...props.cellOutline!.table!, cells},
            cellFacilities
          }
        })
    ).toThrow(/share buffers with inputs/);
  } finally {
    device.destroy();
  }
});

it('rejects triangle-facility output sharing a physical buffer with triangle input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const packed = graph.createTransientBuffer({
      id: 'triangle-packed',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const triangles = graph.createDataView(packed, {format: 'float32x2', length: 3});
    const triangleFacilities = graph.createDataView(packed, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          assignments: createTransientView(graph, 'triangle-assignments', 'uint32', 4),
          raster: {
            ...props.raster!,
            output: {
              ...props.raster!.output,
              triangles,
              triangleFacilities,
              pixelFacilities: createTransientView(graph, 'triangle-pixel-facilities', 'uint32', 4),
              triangleBands: createTransientView(graph, 'triangle-bands', 'uint32', 1),
              status: {
                count: createTransientView(graph, 'triangle-count', 'uint32', 1),
                overflow: createTransientView(graph, 'triangle-overflow', 'uint32', 1)
              }
            }
          }
        })
    ).toThrow(/share buffers with inputs/);
  } finally {
    device.destroy();
  }
});

it('refuses a potentially lossy transient cell table with no published cell-loss status', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createCellProps(graph);
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          cellOutline: {...props.cellOutline!, table: undefined, tableCapacity: 1}
        })
    ).toThrow(/network-isochrones.*tableCapacity.*node count.*4/);
  } finally {
    device.destroy();
  }
});

it('preflights oversized raster ring edges before search scratch', () => {
  const device = createNullWebGPUDevice();
  Object.assign(device.limits, {maxStorageBufferBindingSize: 128 * 1024 * 1024});
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const contributor = new GPUNetworkIsochrones({
      ...props,
      raster: {
        ...props.raster!,
        rings: {
          edgeCapacity: 8_388_609,
          output: {
            ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 3),
            positions: createTransientView(graph, 'ring-positions', 'float32x2', 8),
            status: {
              count: createTransientView(graph, 'ring-count', 'uint32', 1),
              overflow: createTransientView(graph, 'ring-overflow', 'uint32', 1),
              edgeOverflow: createTransientView(graph, 'edge-overflow', 'uint32', 1)
            }
          }
        }
      }
    });
    const create = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/network-isochrones.*edgeCapacity/);
    expect(create).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('names both a search status port and its geometry alias', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const converged = graph.createDataView(props.raster!.output.bandClasses!.buffer, {
      format: 'uint32',
      length: 1
    });
    expect(() => new GPUNetworkIsochrones({...props, output: {status: {converged}}})).toThrow(
      /output.status.converged.*raster.output.bandClasses/
    );
  } finally {
    device.destroy();
  }
});

it('ignores zero tableCapacity for a supplied table but rejects it for a transient table', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createCellProps(graph);
    const create = vi.spyOn(graph, 'createTransientBuffer');
    expect(() =>
      new GPUNetworkIsochrones({
        ...props,
        cellOutline: {...props.cellOutline!, tableCapacity: 0}
      }).getCommandNodes(graph)
    ).not.toThrow();
    expect(
      create.mock.calls.find(([value]) => value.id === 'network-isochrones-cell-facilities')?.[0]
        .byteLength
    ).toBe(8);
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          cellOutline: {...props.cellOutline!, table: undefined, tableCapacity: 0}
        })
    ).toThrow(/tableCapacity/);
  } finally {
    device.destroy();
  }
});

it.each([
  'startCosts',
  'startCount',
  'costLimit'
] as const)('validates the supplied format of %s at construction (VAL-01)', port => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const wrongFormat = port === 'startCount' ? 'float32' : 'uint32';
    const wrong = createTransientView(graph, 'wrong-search-port', wrongFormat, 1);
    const create = () =>
      new GPUNetworkIsochrones({
        ...props,
        id: 'search-formats',
        [port]: wrong
      } as GPUNetworkIsochronesProps);
    expect(create).toThrow(new RegExp(`search-formats.*${port}.*received ${wrongFormat}`));
    const validFormat = port === 'startCount' ? 'uint32' : 'float32';
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          [port]: createTransientView(graph, 'valid-search-port', validFormat, 1)
        } as GPUNetworkIsochronesProps)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('retains disjoint cell status slices (candidateDestination=%s, B6)', candidateDestination => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createCellProps(graph);
    const storage = graph.createTransientBuffer({
      id: 'shared-cell-status',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    props.cellOutline!.table!.status.overflow = graph.createDataView(storage, {
      format: 'uint32',
      length: 1,
      byteOffset: 0
    });
    const destination = graph.createDataView(storage, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
    if (candidateDestination) props.cellOutline!.output.status.candidateOverflow = destination;
    else props.cellOutline!.output.status.count = destination;
    expect(() => new GPUNetworkIsochrones(props).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('retains the cell polygonCount and sumOverflow writers (B4)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createCellProps(graph);
    const polygonCount = createTransientView(graph, 'polygon-count', 'uint32', 1);
    const sumOverflow = createTransientView(graph, 'sum-overflow', 'uint32', 1);
    props.cellOutline!.table!.status.sumOverflow = sumOverflow;
    props.cellOutline!.rings = {
      output: {
        ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 3),
        positions: createTransientView(graph, 'ring-positions', 'float32x2', 12),
        polygons: {
          kind: 'polygons',
          positions: createTransientView(graph, 'polygon-positions', 'float32x2', 12),
          ringOffsets: createTransientView(graph, 'polygon-ring-offsets', 'uint32', 3),
          polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 3),
          featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 3)
        },
        status: {
          count: createTransientView(graph, 'ring-count', 'uint32', 1),
          overflow: createTransientView(graph, 'ring-overflow', 'uint32', 1),
          polygonCount
        }
      }
    };
    const contributor = new GPUNetworkIsochrones(props);
    expect(contributor.props.cellOutline!.table!.status.sumOverflow === sumOverflow).toBe(true);
    expect(contributor.props.cellOutline!.rings!.output.status.polygonCount === polygonCount).toBe(
      true
    );
    const views = contributor
      .getCommandNodes(graph)
      .flatMap(node => (node.resources ?? []).map(getResourceBuffer));
    expect(views.includes(sumOverflow)).toBe(true);
    expect(views.includes(polygonCount)).toBe(true);
  } finally {
    device.destroy();
  }
});

it.each([
  'startCosts',
  'startCount',
  'costLimit'
] as const)('rejects a wrong row count for %s at construction (S1)', port => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const format = port === 'startCount' ? 'uint32' : 'float32';
    const view = createTransientView(graph, 'two-search-rows', format, 2);
    expect(
      () => new GPUNetworkIsochrones({...props, [port]: view} as GPUNetworkIsochronesProps)
    ).toThrow(new RegExp(`network-isochrones.*${port}.*(one row|startNodes).*2`));
    const valid = createTransientView(graph, 'one-search-row', format, 1);
    expect(
      () => new GPUNetworkIsochrones({...props, [port]: valid} as GPUNetworkIsochronesProps)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('captures the raster completeness and polygon destinations at the owned boundary (B5)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const candidateOverflow = createTransientView(graph, 'raster-ring-candidate', 'uint32', 1);
    const complete = createTransientView(graph, 'raster-ring-complete', 'uint32', 1);
    const polygonCount = createTransientView(graph, 'raster-polygon-count', 'uint32', 1);
    props.raster!.rings = {
      edgeCapacity: 2,
      output: {
        ringOffsets: createTransientView(graph, 'raster-ring-offsets', 'uint32', 3),
        positions: createTransientView(graph, 'raster-ring-positions', 'float32x2', 10),
        polygons: {
          kind: 'polygons',
          positions: createTransientView(graph, 'raster-polygons', 'float32x2', 10),
          ringOffsets: createTransientView(graph, 'raster-polygon-rings', 'uint32', 3),
          polygonOffsets: createTransientView(graph, 'raster-polygon-offsets', 'uint32', 3),
          featureOffsets: createTransientView(graph, 'raster-feature-offsets', 'uint32', 3)
        },
        status: {
          count: createTransientView(graph, 'raster-ring-count', 'uint32', 1),
          overflow: createTransientView(graph, 'raster-ring-overflow', 'uint32', 1),
          edgeOverflow: createTransientView(graph, 'raster-edge-overflow', 'uint32', 1),
          candidateOverflow,
          complete,
          polygonCount
        }
      }
    };
    const contributor = new GPUNetworkIsochrones(props);
    const status = contributor.props.raster!.rings!.output.status;
    expect(status.candidateOverflow === candidateOverflow).toBe(true);
    expect(status.complete === complete).toBe(true);
    expect(status.polygonCount === polygonCount).toBe(true);
    const bound = contributor
      .getCommandNodes(graph)
      .flatMap(node => (node.resources ?? []).map(getResourceBuffer));
    for (const destination of [candidateOverflow, complete, polygonCount]) {
      expect(bound.includes(destination)).toBe(true);
    }
  } finally {
    device.destroy();
  }
});
