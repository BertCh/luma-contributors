// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  createTransientView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {
  GPUCalendarBuckets,
  getGPUCalendarBucketsParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';

import type {GPUKeplerTimeUnit} from '@luma-contributors/kepler';
import type {ResidentCalendarInterval} from './resident-chart-plan';
import {makeResidentCalendarTimeView} from './resident-calendar-time';

/** Calendar contributor per original chunk; ordered calendar keys never pass through Float32. */
export class ResidentCalendarKeys {
  readonly ordered: GPUVector<'uint32'>;
  readonly validity: GPUVector<'uint32'>;
  readonly groups: GPUVector<'uint32'>;
  readonly hourOrdered?: GPUVector<'uint32'>;
  readonly hourValidity?: GPUVector<'uint32'>;
  readonly byteLength: number;
  private readonly prepare: CompiledGPUCommandGraph;
  private readonly remap: CompiledGPUCommandGraph;
  private readonly domain: Buffer;
  private readonly parameters: Buffer;
  private readonly hourRemap?: CompiledGPUCommandGraph;
  private readonly dayDomain?: Buffer;
  private readonly ownedBuffers: Buffer[] = [];
  private readonly ownedVectors: GPUVector<'uint32'>[] = [];
  private readonly compiledGraphs: CompiledGPUCommandGraph[] = [];
  private readonly capacity: number;
  constructor(props: {
    device: Device;
    timestamps: GPUVector<'uint32x2'>;
    validity: GPUVector<'uint32'>;
    interval: ResidentCalendarInterval;
    unit: GPUKeplerTimeUnit;
    sourceType: string;
    capacity: number;
  }) {
    const {device} = props;
    const buffers = this.ownedBuffers;
    const vectors = this.ownedVectors;
    this.capacity = props.capacity;
    try {
      const allocate = (name: string) => {
        const chunks: GPUData<'uint32'>[] = [];
        for (const chunk of props.timestamps.data) {
          const buffer = device.createBuffer({
            id: name,
            byteLength: Math.max(4, chunk.length * 4),
            usage: Buffer.STORAGE
          });
          buffers.push(buffer);
          chunks.push(
            // This class owns all raw buffers; vectors own only their chunk wrappers.
            new GPUData({buffer, format: 'uint32', length: chunk.length, ownsBuffer: false})
          );
        }
        const vector = new GPUVector({
          name,
          type: 'data',
          format: 'uint32',
          data: chunks,
          ownsData: true
        });
        vectors.push(vector);
        return vector;
      };
      this.ordered = allocate('calendar-ordered');
      this.validity = allocate('calendar-validity');
      this.groups = allocate('calendar-groups');
      const sanitizedHours = props.interval === 'hour' ? allocate('calendar-hours') : undefined;
      if (sanitizedHours) {
        this.hourOrdered = allocate('calendar-hour-ordered');
        this.hourValidity = allocate('calendar-hour-validity');
        this.dayDomain = device.createBuffer({
          id: 'calendar-day-domain',
          byteLength: (props.capacity + 1) * 4,
          usage: Buffer.STORAGE | Buffer.COPY_DST
        });
        buffers.push(this.dayDomain);
      }
      this.parameters = device.createBuffer({
        id: 'calendar-parameters',
        data: getGPUCalendarBucketsParameterValues(0),
        usage: Buffer.STORAGE
      });
      buffers.push(this.parameters);
      this.domain = device.createBuffer({
        id: 'calendar-domain',
        byteLength: (props.capacity + 1) * 4,
        usage: Buffer.STORAGE | Buffer.COPY_DST
      });
      buffers.push(this.domain);
      const graph = new GPUCommandGraph(device, {id: 'chart-calendar-keys'});
      const timestamps = graph.importGPUVector('times', props.timestamps);
      const validity = graph.importGPUVector('validity', props.validity);
      const ordered = graph.importGPUVector('ordered', this.ordered);
      const validOutput = graph.importGPUVector('valid-output', this.validity);
      const hourOutput = sanitizedHours
        ? graph.importGPUVector('hour-output', sanitizedHours)
        : undefined;
      const parameters = graph.createDataView(
        graph.importBuffer(
          {id: 'parameters', byteLength: 8, usage: this.parameters.usage},
          this.parameters
        ),
        {format: 'sint32', length: 2}
      );
      for (const [chunkIndex, timestampChunk] of timestamps.data.entries()) {
        if (!timestampChunk.length) continue;
        const source = makeResidentCalendarTimeView(graph, {
          id: `calendar-time-${chunkIndex}`,
          timestamps: timestampChunk,
          unit: props.unit,
          sourceType: props.sourceType
        });
        const valid = validity.data[chunkIndex];
        const key = ordered.data[chunkIndex];
        const outputValidity = validOutput.data[chunkIndex];
        const year = createTransientView(graph, `year-${chunkIndex}`, 'sint32', source.length);
        const month =
          props.interval === 'month'
            ? createTransientView(graph, `month-${chunkIndex}`, 'uint32', source.length)
            : undefined;
        const dayOfYear =
          props.interval === 'hour' || props.interval === 'day' || props.interval === 'week'
            ? createTransientView(graph, `day-of-year-${chunkIndex}`, 'uint32', source.length)
            : undefined;
        const hour = hourOutput
          ? createTransientView(graph, `hour-${chunkIndex}`, 'uint32', source.length)
          : undefined;
        const sanitizedHour = hourOutput?.data[chunkIndex];
        // Month and day/week each need only one additional calendar output binding.
        const periodPart = month ?? dayOfYear;
        graph.add(
          new GPUCalendarBuckets({
            id: `calendar-${chunkIndex}`,
            timestamps: source,
            mask: valid,
            parameters,
            output: {
              year,
              ...(month ? {month} : {}),
              ...(dayOfYear ? {dayOfYear} : {}),
              ...(hour ? {hour} : {})
            }
          })
        );
        const dispatch = getBoundedDispatchLayout(
          'CalendarChartKeys',
          source.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `calendar-key-${chunkIndex}`,
          resources: [
            {buffer: source, usage: 'storage-read'},
            {buffer: valid, usage: 'storage-read'},
            {buffer: year, usage: 'storage-read'},
            ...(periodPart ? [{buffer: periodPart, usage: 'storage-read' as const}] : []),
            {buffer: key, usage: 'storage-write'},
            {buffer: outputValidity, usage: 'storage-write'},
            ...(hour ? [{buffer: hour, usage: 'storage-read' as const}] : []),
            ...(sanitizedHour ? [{buffer: sanitizedHour, usage: 'storage-write' as const}] : [])
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
@group(0) @binding(0) var<storage,read> times: array<u32>;
@group(0) @binding(1) var<storage,read> valid: array<u32>;
@group(0) @binding(2) var<storage,read> year: array<i32>;
@group(0) @binding(3) var<storage,read_write> keys: array<u32>;
@group(0) @binding(4) var<storage,read_write> validity: array<u32>;
${periodPart ? '@group(0) @binding(5) var<storage,read> periodPart: array<u32>;' : ''}
${hour ? '@group(0) @binding(6) var<storage,read> rawHours: array<u32>;' : ''}
${sanitizedHour ? '@group(0) @binding(7) var<storage,read_write> hours: array<u32>;' : ''}
${
  dayOfYear
    ? `
// WGSL truncates signed division; Gregorian years before zero need floor division.
fn floorDivide(value: i32, divisor: i32) -> i32 {
  return value / divisor - select(0i, 1i, value % divisor < 0i);
}
fn positiveModulo(value: i32, divisor: i32) -> i32 {
  return ((value % divisor) + divisor) % divisor;
}`
    : ''
}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${source.length}u) {return;}
  let present = valid[${valid.byteOffset / 4}u + index] != 0u;
  let timeIndex = ${source.byteOffset / 4}u + index * 2u;
  let instant = vec2<u32>(times[timeIndex], times[timeIndex + 1u]);
  // Date TimeClip is +/- 8,640,000,000,000,000 ms; compare the Int64 words exactly.
  let high = instant.y ^ 0x80000000u;
  let minimumHigh = ${Number((BigInt.asUintN(64, -8640000000000000n) >> 32n) ^ 0x80000000n)}u;
  let maximumHigh = ${Number((8640000000000000n >> 32n) ^ 0x80000000n)}u;
  let inRange = (high > minimumHigh || (high == minimumHigh && instant.x >= ${Number(BigInt.asUintN(64, -8640000000000000n) & 0xffffffffn)}u)) && (high < maximumHigh || (high == maximumHigh && instant.x <= ${Number(8640000000000000n & 0xffffffffn)}u));
  // Native new Date(null) groups source nulls at the Unix epoch; invalid valid instants stay invalid.
  let calendarYear = select(1970i, year[index], present);
  ${
    dayOfYear
      ? `
  let previousYear = calendarYear - 1i;
  // dayOfYear is one-based; nulls substitute 1970-01-01 before forming the ordinal.
  let epochDay = 365i * (calendarYear - 1970i)
    + floorDivide(previousYear, 4i) - 492i
    - floorDivide(previousYear, 100i) + 19i
    + floorDivide(previousYear, 400i) - 4i
    + i32(select(1u, periodPart[index], present)) - 1i;
  let ordinal = ${props.interval === 'week' ? 'epochDay - positiveModulo(epochDay + 4i, 7i)' : 'epochDay'};`
      : `let ordinal = ${month ? 'calendarYear * 12i + i32(select(1u, periodPart[index], present)) - 1i' : 'calendarYear'};`
  }
  keys[index] = bitcast<u32>(ordinal) ^ 0x80000000u;
  let validInstant = (!present || inRange) && calendarYear != (-2147483647i - 1i);
  validity[index] = select(0u, 1u, validInstant);
  ${hour ? 'hours[index] = select(0xffffffffu, select(0u, rawHours[index], present), validInstant);' : ''}
}`,
              shaderLayout: {
                bindings: ['times', 'valid', 'year', 'keys', 'validity']
                  .map((name, location) => ({name, location, group: 0, type: 'storage' as const}))
                  .concat(
                    periodPart ? [{name: 'periodPart', location: 5, group: 0, type: 'storage'}] : []
                  )
                  .concat(
                    hour
                      ? [
                          {name: 'rawHours', location: 6, group: 0, type: 'storage'},
                          {name: 'hours', location: 7, group: 0, type: 'storage'}
                        ]
                      : []
                  )
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    times: getBuffer(source.buffer),
                    valid: getBuffer(valid.buffer),
                    year: getBuffer(year.buffer),
                    keys: getBuffer(key.buffer),
                    validity: getBuffer(outputValidity.buffer),
                    ...(periodPart ? {periodPart: getBuffer(periodPart.buffer)} : {}),
                    ...(hour ? {rawHours: getBuffer(hour.buffer)} : {}),
                    ...(sanitizedHour ? {hours: getBuffer(sanitizedHour.buffer)} : {})
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
      this.prepare = graph.compile();
      this.compiledGraphs.push(this.prepare);
      if (sanitizedHours && this.hourOrdered && this.hourValidity && this.dayDomain) {
        this.hourRemap = this.compileHours(
          device,
          sanitizedHours,
          this.hourOrdered,
          this.hourValidity,
          this.dayDomain
        );
        this.compiledGraphs.push(this.hourRemap);
      }
      const remapGraph = new GPUCommandGraph(device, {id: 'chart-calendar-groups'});
      const keys = remapGraph.importGPUVector('ordered', this.hourOrdered ?? this.ordered);
      const valid = remapGraph.importGPUVector('valid', this.hourValidity ?? this.validity);
      const groups = remapGraph.importGPUVector('groups', this.groups);
      const domain = remapGraph.createDataView(
        remapGraph.importBuffer(
          {id: 'domain', byteLength: this.domain.byteLength, usage: this.domain.usage},
          this.domain
        ),
        {format: 'uint32', length: props.capacity + 1}
      );
      for (const [chunkIndex, key] of keys.data.entries()) {
        if (!key.length) continue;
        const validChunk = valid.data[chunkIndex];
        const target = groups.data[chunkIndex];
        const dispatch = getBoundedDispatchLayout(
          'CalendarChartGroups',
          key.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        remapGraph.addComputePass({
          id: `calendar-group-${chunkIndex}`,
          resources: [
            {buffer: key, usage: 'storage-read'},
            {buffer: validChunk, usage: 'storage-read'},
            {buffer: domain, usage: 'storage-read'},
            {buffer: target, usage: 'storage-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
@group(0) @binding(0) var<storage,read> keys: array<u32>;
@group(0) @binding(1) var<storage,read> valid: array<u32>;
@group(0) @binding(2) var<storage,read> domain: array<u32>;
@group(0) @binding(3) var<storage,read_write> output: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
 ${getBoundedInvocationIndexSource(dispatch, 256)}
 if(index >= ${key.length}u) {return;}
 var lower = 0u; var upper = domain[0]; let key = keys[index];
 while(lower < upper) {let middle = lower + (upper - lower) / 2u; if(domain[middle + 1u] < key) {lower = middle + 1u;} else {upper = middle;}}
 output[index] = select(0xffffffffu, lower, valid[index] != 0u && lower < domain[0] && domain[lower + 1u] == key);
}`,
              shaderLayout: {
                bindings: ['keys', 'valid', 'domain', 'output'].map((name, location) => ({
                  name,
                  location,
                  group: 0,
                  type: 'storage' as const
                }))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    keys: getBuffer(key.buffer),
                    valid: getBuffer(validChunk.buffer),
                    domain: getBuffer(domain.buffer),
                    output: getBuffer(target.buffer)
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
      this.remap = remapGraph.compile();
      this.compiledGraphs.push(this.remap);
      this.byteLength =
        buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
        this.compiledGraphs.reduce(
          (total, compiled) => total + compiled.stats.physicalTransientResourceBytes,
          0
        );
    } catch (error) {
      for (const compiled of this.compiledGraphs) compiled.destroy();
      for (const vector of vectors) vector.destroy();
      for (const buffer of buffers) buffer.destroy();
      throw error;
    }
  }
  private compileHours(
    device: Device,
    sanitizedHours: GPUVector<'uint32'>,
    hourOrdered: GPUVector<'uint32'>,
    hourValidity: GPUVector<'uint32'>,
    dayDomain: Buffer
  ): CompiledGPUCommandGraph {
    const graph = new GPUCommandGraph(device, {id: 'chart-calendar-dense-hours'});
    const days = graph.importGPUVector('days', this.ordered);
    const validity = graph.importGPUVector('validity', this.validity);
    const hours = graph.importGPUVector('hours', sanitizedHours);
    const output = graph.importGPUVector('output', hourOrdered);
    const outputValidity = graph.importGPUVector('output-validity', hourValidity);
    const domain = graph.createDataView(
      graph.importBuffer(
        {id: 'day-domain', byteLength: dayDomain.byteLength, usage: dayDomain.usage},
        dayDomain
      ),
      {format: 'uint32', length: this.capacity + 1}
    );
    for (const [chunkIndex, day] of days.data.entries()) {
      if (!day.length) continue;
      const valid = validity.data[chunkIndex];
      const hour = hours.data[chunkIndex];
      const target = output.data[chunkIndex];
      const targetValidity = outputValidity.data[chunkIndex];
      const dispatch = getBoundedDispatchLayout(
        'CalendarDenseHours',
        day.length,
        256,
        device.limits.maxComputeWorkgroupsPerDimension
      );
      graph.addComputePass({
        id: `calendar-dense-hours-${chunkIndex}`,
        resources: [
          {buffer: day, usage: 'storage-read'},
          {buffer: valid, usage: 'storage-read'},
          {buffer: hour, usage: 'storage-read'},
          {buffer: domain, usage: 'storage-read'},
          {buffer: target, usage: 'storage-write'},
          {buffer: targetValidity, usage: 'storage-write'}
        ],
        compile: ({device: compileDevice}) => {
          const kernel = new Kernel(compileDevice, {
            source: `
@group(0) @binding(0) var<storage,read> days: array<u32>;
@group(0) @binding(1) var<storage,read> valid: array<u32>;
@group(0) @binding(2) var<storage,read> hours: array<u32>;
@group(0) @binding(3) var<storage,read> domain: array<u32>;
@group(0) @binding(4) var<storage,read_write> output: array<u32>;
@group(0) @binding(5) var<storage,read_write> outputValidity: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${day.length}u) {return;}
  let day = days[index];
  let hour = hours[index];
  var lower = 0u;
  var upper = domain[0];
  while (lower < upper) {
    let middle = lower + (upper - lower) / 2u;
    if (domain[middle + 1u] < day) {lower = middle + 1u;} else {upper = middle;}
  }
  let found = valid[index] != 0u && hour < 24u && lower < domain[0] && domain[lower + 1u] == day;
  // A missing day has false validity, never a valid sentinel key in the final domain.
  output[index] = select(0u, lower * 24u + hour, found);
  outputValidity[index] = select(0u, 1u, found);
}`,
            shaderLayout: {
              bindings: ['days', 'valid', 'hours', 'domain', 'output', 'outputValidity'].map(
                (name, location) => ({
                  name,
                  location,
                  group: 0,
                  type: 'storage' as const
                })
              )
            }
          });
          return {
            encode: ({computePass, getBuffer}) =>
              kernel.dispatch(computePass, {
                bindings: {
                  days: getBuffer(day.buffer),
                  valid: getBuffer(valid.buffer),
                  hours: getBuffer(hour.buffer),
                  domain: getBuffer(domain.buffer),
                  output: getBuffer(target.buffer),
                  outputValidity: getBuffer(targetValidity.buffer)
                },
                ...dispatch
              }),
            destroy: () => kernel.destroy()
          };
        }
      });
    }
    return graph.compile();
  }
  encodePrepare(encoder: CommandEncoder): void {
    this.prepare.encode(encoder, {parameters: undefined});
  }
  /** Remap observed epoch days to bounded dense hours after the first domain discovery. */
  encodeHours(encoder: CommandEncoder, dayKeys: readonly number[]): void {
    // Hour mode requires its own day-domain buffer and a bounded, sorted day-key list.
    assert(this.hourRemap && this.dayDomain && dayKeys.length <= this.capacity);
    this.dayDomain.write(new Uint32Array([dayKeys.length, ...dayKeys]));
    this.hourRemap.encode(encoder, {parameters: undefined});
  }
  encodeGroups(encoder: CommandEncoder, keys: readonly number[]): void {
    // The discovered final domain must fit the admitted calendar capacity.
    assert(keys.length <= this.capacity);
    this.domain.write(new Uint32Array([keys.length, ...keys]));
    this.remap.encode(encoder, {parameters: undefined});
  }
  destroy(): void {
    for (const compiled of this.compiledGraphs) compiled.destroy();
    for (const vector of this.ownedVectors) vector.destroy();
    for (const buffer of this.ownedBuffers) buffer.destroy();
  }
}
