// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import {
  createTransientView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import type {GPUKeplerTimeUnit} from '@luma-contributors/kepler';

/**
 * Normalizes one original timestamp chunk to exact integer milliseconds for GPUCalendarBuckets.
 * Millisecond and empty views are borrowed; other units use graph-owned transient storage.
 *
 * Keep the ORIGINAL source validity beside this view. A null row is not identified by its time
 * words. Converted out-of-Date-range instants map to Int64 maximum, never to null or the epoch.
 * Borrowed milliseconds retain their words; the calendar consumer still checks Date range.
 * Micro/nanoseconds truncate toward zero, including negative fractions, without Arrow's intermediate
 * Number rounding. The caller must separately qualify native chart/filter compatibility for those
 * units. This helper does not admit chart modes or change source timestamps.
 */
export function makeResidentCalendarTimeView<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    timestamps: GraphDataView<'uint32x2'>;
    unit: GPUKeplerTimeUnit;
    sourceType: string;
  }
): GraphDataView<'uint32x2'> {
  // Declared Int64 time has bigint native values; new Date(bigint) is not a native chart contract.
  assert(props.sourceType.startsWith('Timestamp') || props.sourceType.startsWith('Date'));
  validatePackedView(props.timestamps, ['uint32x2'], 'Calendar timestamps');
  if (props.unit === 'millisecond' || !props.timestamps.length) return props.timestamps;

  const source = props.timestamps;
  const output = createTransientView(
    graph,
    `${props.id}-milliseconds`,
    'uint32x2',
    source.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const dispatch = getBoundedDispatchLayout(
    'CalendarTimeNormalization',
    source.length,
    256,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const divisor = props.unit === 'microsecond' ? 1000 : 1000000;
  const sourceCode = `
@group(0) @binding(0) var<storage, read> timestamps: array<u32>;
@group(0) @binding(1) var<storage, read_write> milliseconds: array<vec2<u32>>;

fn negateWords(value: vec2<u32>) -> vec2<u32> {
  let low = ~value.x + 1u;
  return vec2<u32>(low, ~value.y + select(0u, 1u, low == 0u));
}

fn exceedsMagnitude(value: vec2<u32>, maximum: vec2<u32>) -> bool {
  return value.y > maximum.y || (value.y == maximum.y && value.x > maximum.x);
}

// Same nibble-wise division as GPUCalendarBuckets: remainder * 16 fits u32 for both divisors.
fn divideMagnitude(value: vec2<u32>, divisor: u32) -> vec2<u32> {
  var remainder = value.y % divisor;
  var quotientLow = 0u;
  for (var nibble = 7i; nibble >= 0i; nibble = nibble - 1i) {
    let shift = u32(nibble) * 4u;
    let current = (remainder << 4u) | ((value.x >> shift) & 0xfu);
    quotientLow = quotientLow | ((current / divisor) << shift);
    remainder = current % divisor;
  }
  return vec2<u32>(quotientLow, value.y / divisor);
}

fn normalizeTime(value: vec2<u32>) -> vec2<u32> {
  let negative = (value.y & 0x80000000u) != 0u;
  let magnitude = select(value, negateWords(value), negative);
  var normalized: vec2<u32>;
  ${
    props.unit === 'second'
      ? `// Date's seconds bound is stricter than Int64 overflow, so the product below is safe.
  if (exceedsMagnitude(magnitude, vec2<u32>(${Number(8640000000000n & 0xffffffffn)}u, ${Number(8640000000000n >> 32n)}u))) {
    return vec2<u32>(0xffffffffu, 0x7fffffffu);
  }
  let lowerProduct = (magnitude.x & 0xffffu) * 1000u;
  let upperProduct = (magnitude.x >> 16u) * 1000u + (lowerProduct >> 16u);
  normalized = vec2<u32>((upperProduct << 16u) | (lowerProduct & 0xffffu), magnitude.y * 1000u + (upperProduct >> 16u));`
      : `${
          props.unit === 'microsecond'
            ? `// Reject fractional instants outside Date's range before discarding the remainder.
  if (exceedsMagnitude(magnitude, vec2<u32>(${Number(8640000000000000000n & 0xffffffffn)}u, ${Number(8640000000000000000n >> 32n)}u))) {
    return vec2<u32>(0xffffffffu, 0x7fffffffu);
  }
  `
            : ''
        }normalized = divideMagnitude(magnitude, ${divisor}u);`
  }
  // TimeClip range is inclusive +/- 8,640,000,000,000,000 integer milliseconds.
  if (exceedsMagnitude(normalized, vec2<u32>(${Number(8640000000000000n & 0xffffffffn)}u, ${Number(8640000000000000n >> 32n)}u))) {
    return vec2<u32>(0xffffffffu, 0x7fffffffu);
  }
  return select(normalized, negateWords(normalized), negative);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${source.length}u) {return;}
  let sourceIndex = ${source.byteOffset / 4}u + index * 2u;
  milliseconds[index] = normalizeTime(vec2<u32>(timestamps[sourceIndex], timestamps[sourceIndex + 1u]));
}`;
  graph.addComputePass({
    id: `${props.id}-normalize`,
    resources: [
      {buffer: source, usage: 'storage-read'},
      {buffer: output, usage: 'storage-write'}
    ],
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: `${props.id}-normalize`,
        source: sourceCode,
        shaderLayout: {
          bindings: [
            {name: 'timestamps', location: 0, group: 0, type: 'storage'},
            {name: 'milliseconds', location: 1, group: 0, type: 'storage'}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {
              timestamps: getBuffer(source.buffer),
              milliseconds: getBuffer(output.buffer)
            },
            ...dispatch
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
  return output;
}
