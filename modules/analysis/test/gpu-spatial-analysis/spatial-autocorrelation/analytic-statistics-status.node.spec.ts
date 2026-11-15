// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPULocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function setupStatusValidation(chunked: boolean) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `status-validation-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 5),
    neighbors: view('uint32', 8),
    weights: view('float32', 8)
  };
  return {
    device,
    graph,
    view,
    props: {
      id: 'analytic-status-validation',
      values: view('float32', 4),
      mask: view('uint32', 4),
      parameters: view('float32', 8),
      weights: chunked ? {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights}]} : weights,
      output: {zScores: view('float32', 4), statistic: view('float32', 4)}
    }
  };
}

for (const contributorClass of [GPULocalMoran, GPUHotSpotAnalysis, GPULocalGeary]) {
  for (const chunked of [false, true]) {
    it(`${contributorClass.name} rejects nonscalar invalidCount at construction and permits ${chunked ? 'chunked' : 'whole'} retry`, () => {
      const {device, graph, view, props} = setupStatusValidation(chunked);
      try {
        const invalidCount = view('uint32', 2);
        const correctedCount = graph.createDataView(invalidCount.buffer, {
          format: 'uint32',
          length: 1
        });
        const overflow = view('uint32', 1);
        const allocate = vi.spyOn(graph, 'createTransientBuffer');
        expect(
          () =>
            new contributorClass({
              ...props,
              output: {...props.output, status: {invalidCount, overflow}}
            })
        ).toThrow(/^analytic-status-validation .*invalidCount.*1/);
        expect(allocate).not.toHaveBeenCalled();
        expect(() =>
          graph.add(
            new contributorClass({
              ...props,
              output: {...props.output, status: {invalidCount: correctedCount, overflow}}
            })
          )
        ).not.toThrow();
        expect(allocate).toHaveBeenCalled();
      } finally {
        device.destroy();
      }
    });

    for (const binding of ['invalidCount', 'overflow', 'inputOverflow'] as const) {
      it(`${contributorClass.name} preflights ${binding} before allocation and permits ${chunked ? 'chunked' : 'whole'} retry`, () => {
        const {device, graph, view, props} = setupStatusValidation(chunked);
        try {
          const storage = view('uint32', 17);
          const oversized = graph.createDataView(storage.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: 64
          });
          const corrected = graph.createDataView(storage.buffer, {
            format: 'uint32',
            length: 1,
            // An aligned binding of exactly 64 bytes remains accepted.
            byteOffset: 60
          });
          const invalidCount = view('uint32', 1);
          const overflow = view('uint32', 1);
          const inputOverflow = view('uint32', 1);
          const createContributor = (statusBinding: GraphDataView<'uint32'>) =>
            new contributorClass({
              ...props,
              weights: {
                ...props.weights,
                status: {overflow: binding === 'inputOverflow' ? statusBinding : inputOverflow}
              },
              output: {
                ...props.output,
                status: {
                  invalidCount: binding === 'invalidCount' ? statusBinding : invalidCount,
                  overflow: binding === 'overflow' ? statusBinding : overflow
                }
              }
            });
          Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
          const rejected = createContributor(oversized);
          const retry = createContributor(corrected);
          const allocate = vi.spyOn(graph, 'createTransientBuffer');
          expect(() => graph.add(rejected)).toThrow(/68.*64/);
          expect(allocate).not.toHaveBeenCalled();
          expect(() => graph.add(retry)).not.toThrow();
          expect(allocate).toHaveBeenCalled();
        } finally {
          device.destroy();
        }
      });
    }
  }

  it(`${contributorClass.name} exposes optional named status within core binding limits`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `analytic-status-${serial++}`, format, length);
    const props = {
      values: view('float32', 5),
      mask: view('uint32', 5),
      parameters: view('float32', 8),
      weights: {
        offsets: view('uint32', 6),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      output: {zScores: view('float32', 5), statistic: view('float32', 5)}
    };
    try {
      const plain = new contributorClass(props).getCommandNodes(graph);
      expect(plain.some(node => node.id.endsWith('-invalid-rows'))).toBe(false);
      for (const chunked of [false, true]) {
        const nodes = new contributorClass({
          ...props,
          id: `status-${chunked}`,
          weights: chunked
            ? {rowCount: 5, chunks: [{rowStart: 0, rowCount: 5, weights: props.weights}]}
            : props.weights,
          multipleTesting: 'benjamini-hochberg',
          output: {
            ...props.output,
            status: {invalidCount: view('uint32', 1), overflow: view('uint32', 1)}
          }
        }).getCommandNodes(graph);
        expect(nodes.some(node => node.id.endsWith('-invalid-rows'))).toBe(true);
        expect(nodes.some(node => node.id.endsWith('-overflow-status'))).toBe(true);
        const statusNodes = nodes.filter(
          node => node.id.endsWith('-invalid-rows') || node.id.endsWith('-overflow-status')
        );
        expect(getParsedKernelSources(statusNodes).size).toBe(2);
        for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
          expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
        }
      }
      const invalid = view('float32', 1) as unknown as GraphDataView<'uint32'>;
      expect(
        () =>
          new contributorClass({
            ...props,
            output: {...props.output, status: {invalidCount: invalid}}
          })
      ).toThrow(/invalidCount/);
      expect(
        () =>
          new contributorClass({
            ...props,
            output: {...props.output, status: {overflow: view('uint32', 0)}}
          })
      ).toThrow(/overflow/);
      const alias = graph.createDataView(
        (contributorClass === GPULocalGeary ? props.output.statistic : props.output.zScores).buffer,
        {
          format: 'uint32',
          length: 1
        }
      );
      expect(
        () =>
          new contributorClass({...props, output: {...props.output, status: {invalidCount: alias}}})
      ).toThrow(/share buffers/);
      const shared = view('uint32', 1);
      expect(
        () =>
          new contributorClass({
            ...props,
            output: {...props.output, status: {invalidCount: shared, overflow: shared}}
          })
      ).toThrow(/share buffers/);
      const inputAlias = graph.createDataView(props.values.buffer, {format: 'uint32', length: 1});
      expect(
        () =>
          new contributorClass({
            ...props,
            output: {...props.output, status: {invalidCount: inputAlias}}
          })
      ).toThrow(/share buffers/);
      const foreignGraph = new GPUCommandGraph(device);
      const foreign = createTransientView(foreignGraph, 'foreign-status', 'uint32', 1);
      expect(() =>
        new contributorClass({
          ...props,
          output: {...props.output, status: {invalidCount: foreign}}
        }).getCommandNodes(graph)
      ).toThrow(/target graph/);
    } finally {
      device.destroy();
    }
  });
}
