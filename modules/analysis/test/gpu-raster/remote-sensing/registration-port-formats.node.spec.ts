// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPURasterRegistration,
  type GPURasterRegistrationProps
} from '../../../src/raster-analysis/remote-sensing/gpu-raster-registration';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';

it('S2 gives named and packed registration failures the same structured identity', () => {
  const {device, graph, view} = createKernelAuditContext('registration-identity');
  try {
    const props: GPURasterRegistrationProps = {
      id: 'study',
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      reference: view('float32', 4),
      moving: view('float32', 4),
      output: {shift: view('float32x2', 1), peak: view('float32', 1), status: view('uint32', 1)}
    };
    expect(
      () =>
        new GPURasterRegistration({
          ...props,
          output: {...props.output, shift: view('uint32', 1)}
        } as unknown as GPURasterRegistrationProps)
    ).toThrowError(
      expect.objectContaining({
        id: 'study',
        contributor: 'GPURasterRegistration',
        path: 'output.shift',
        code: 'VIEW_FORMAT'
      })
    );
    const reference = graph.createDataView(view('float32', 8).buffer, {
      format: 'float32',
      length: 4,
      byteStride: 8
    });
    expect(() => new GPURasterRegistration({...props, reference})).toThrowError(
      expect.objectContaining({
        id: 'study',
        contributor: 'GPURasterRegistration',
        path: 'reference',
        code: 'VIEW_FORMAT'
      })
    );
  } finally {
    device.destroy();
  }
});

it.each([
  'periodic',
  'masked-linear'
] as const)('N1 captures registration output bindings in %s mode', mode => {
  const {device, graph, view} = createKernelAuditContext(`registration-snapshot-${mode}`);
  try {
    const props: GPURasterRegistrationProps = {
      mode,
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      reference: view('float32', 4),
      moving: view('float32', 4),
      output: {shift: view('float32x2', 1), peak: view('float32', 1), status: view('uint32', 1)}
    };
    const peak = props.output.peak;
    const contributor = new GPURasterRegistration(props);
    props.output.peak = props.reference;
    expect(contributor.props.output.peak).toBe(peak);
    const nodes = contributor.getCommandNodes(graph);
    expect(
      nodes.some(node =>
        node.resources?.some(
          resource =>
            resource.usage === 'storage-read-write' &&
            'buffer' in resource &&
            resource.buffer === props.reference
        )
      )
    ).toBe(false);
  } finally {
    device.destroy();
  }
});

it('CCK-05/06 masked registration preserves named input and output formats, lengths and aliases', () => {
  const {device, graph, view} = createKernelAuditContext('masked-formats');
  try {
    const props: GPURasterRegistrationProps = {
      id: 'masked-formats',
      mode: 'masked-linear',
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      reference: view('float32', 4),
      moving: view('float32', 4),
      referenceMask: view('uint32', 4),
      movingMask: view('uint32', 4),
      output: {
        shift: view('float32x2', 1),
        peak: view('float32', 1),
        status: view('uint32', 1),
        secondPeakRatio: view('float32', 1),
        overlap: view('uint32', 1)
      }
    };
    expect(() => new GPURasterRegistration(props)).not.toThrow();
    expect(
      () => new GPURasterRegistration({...props, referenceMask: undefined, movingMask: undefined})
    ).not.toThrow();
    for (const name of ['reference', 'moving', 'referenceMask', 'movingMask'] as const) {
      const malformed = {...props, [name]: view(name.endsWith('Mask') ? 'float32' : 'uint32', 4)};
      expect(
        () => new GPURasterRegistration(malformed as GPURasterRegistrationProps),
        name
      ).toThrow(new RegExp(`masked-formats ${name}.*format`));
      expect(
        () => new GPURasterRegistration({...props, [name]: view(props[name]!.format, 3)})
      ).toThrow(/grid.width times grid.height/);
    }
    for (const name of ['shift', 'peak', 'status', 'secondPeakRatio', 'overlap'] as const) {
      const wrong = view(props.output[name]!.format === 'uint32' ? 'float32' : 'uint32', 1);
      expect(
        () =>
          new GPURasterRegistration({
            ...props,
            output: {...props.output, [name]: wrong}
          } as GPURasterRegistrationProps),
        name
      ).toThrow(new RegExp(`masked-formats output.${name}.*format`));
      expect(
        () =>
          new GPURasterRegistration({
            ...props,
            output: {...props.output, [name]: view(props.output[name]!.format, 0)}
          })
      ).toThrow(/must not be empty/);
    }
    const strided = graph.createDataView(props.reference.buffer, {
      format: 'float32',
      length: 2,
      byteStride: 8
    });
    expect(() => new GPURasterRegistration({...props, reference: strided})).toThrow(/packed/);
    expect(
      () => new GPURasterRegistration({...props, output: {...props.output, peak: props.reference}})
    ).toThrow(/overlap|alias/);
  } finally {
    device.destroy();
  }
});
