// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Binding, Buffer} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import {
  GPUCommandGraph,
  createGPUComputeCommandNode,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUPointCloudChange,
  GPUPointCloudFeatures,
  GPUPointCloudICP,
  GPUPointCloudNeighborhoods
} from '@luma-contributors/graph/gpu-pointcloud';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import type {BarChartData} from '../chart-types';
import type {PointCloudChangeOptions} from './point-cloud-change.types';
import {clamp} from '@math.gl/core';

type ChangeSettings = Pick<
  PointCloudChangeOptions,
  | 'metric'
  | 'alignment'
  | 'radius'
  | 'halfDepth'
  | 'registrationError'
  | 'uncertaintyCombination'
  | 'robust'
  | 'maximumDistance'
>;
type ActiveGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  settings: ChangeSettings;
  changeValueBuffer: Buffer;
  roughnessValueBuffer: Buffer;
  countsValueBuffer: Buffer;
  statusValueBuffer: Buffer;
  icpResources: {
    transform: Buffer;
    history: Buffer;
    inlierMask: Buffer;
    status: Buffer;
    targetFeatureStatus: Buffer | null;
    targetNeighborhoodCounts: Buffer | null;
  } | null;
  ownedBuffers: Buffer[];
};
const ICP_ROWS = 1024;
const ICP_ITERATIONS = 12;
const TARGET_NORMAL_RADIUS = 3;
const TARGET_NORMAL_CAPACITY = 64;
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export async function createPointCloudChange(
  ctx: SceneContext<PointCloudChangeOptions>
): Promise<SceneInstance<PointCloudChangeOptions>> {
  const dataset = ctx.datasets.get('ahk-rock-glacier');
  const reference = dataset.column<Float32Array>('referenceXYZ');
  const comparison = dataset.column<Float32Array>('comparisonXYZ');
  const cores = dataset.column<Float32Array>('coresXYZ');
  const normals = dataset.column<Float32Array>('normal');
  const referenceRows = reference.length / 3;
  const referenceDisplayPositions = dataset.projectColumn(
    'referencePosition',
    dataset.defaultOrigin
  );
  const comparisonDisplayPositions = dataset.projectColumn(
    'comparisonPosition',
    dataset.defaultOrigin
  );
  const comparisonRows = comparison.length / 3;
  const coreCount = cores.length / 3;
  const originLngLat = dataset.defaultOrigin;
  const displayPositions = dataset.projectColumn('position', originLngLat);
  const coordinatesBuffer = new SpatialAnalysisResources(ctx.device, 'point-cloud-change');
  const corePositionBuffer = coordinatesBuffer.createBuffer('core-positions', displayPositions);
  const referenceDisplayBuffer = coordinatesBuffer.createBuffer(
    'reference-display-positions',
    referenceDisplayPositions
  );
  const comparisonDisplayBuffer = coordinatesBuffer.createBuffer(
    'comparison-display-positions',
    comparisonDisplayPositions
  );
  const referenceBuffer = coordinatesBuffer.createBuffer('reference-xyz', reference);
  const comparisonBuffer = coordinatesBuffer.createBuffer('comparison-xyz', comparison);
  const coresBuffer = coordinatesBuffer.createBuffer('cores-xyz', cores);
  const normalsBuffer = coordinatesBuffer.createBuffer('normals', normals);
  const valueBuffer = coordinatesBuffer.createBuffer(
    'core-result-values',
    new Float32Array(coreCount)
  );
  const validityBuffer = coordinatesBuffer.createBuffer(
    'core-result-validity',
    new Uint32Array(coreCount)
  );
  const {device} = ctx;
  const graphSwap = coordinatesBuffer.track(new GraphSwap());
  const scalarOrigin = [originLngLat[0], originLngLat[1], 0] as [number, number, number];
  let graphDirty = true;
  let currentGraph: ActiveGraph;
  let currentAlignmentValid = true;
  let latestChanges = new Float32Array(coreCount * 3);
  let latestCounts = new Uint32Array(coreCount * 2);
  let latestStatuses = new Uint32Array(coreCount);
  let latestIcpHistory = new Float32Array(ICP_ITERATIONS * 3);
  let latestIcpStatus = 0;
  let latestIcpInliers = new Uint32Array(ICP_ROWS);
  let latestTargetFeatureStatus = new Uint32Array(ICP_ROWS);
  let latestTargetNeighborhoodCounts = new Uint32Array(ICP_ROWS);
  const updateDisplayValues = () => {
    const values = new Float32Array(coreCount);
    const validity = new Uint32Array(coreCount);
    for (let coreIndex = 0; coreIndex < coreCount; coreIndex++) {
      if (!currentAlignmentValid || latestStatuses[coreIndex] !== 0) {
        validity[coreIndex] = ctx.options.view === 'detection' ? 3 : 0;
        continue;
      }
      const difference = latestChanges[coreIndex * 3];
      const lower = latestChanges[coreIndex * 3 + 1];
      const upper = latestChanges[coreIndex * 3 + 2];
      values[coreIndex] =
        ctx.options.view === 'difference' ? difference : lower > 0 ? 1 : upper < 0 ? 2 : 0;
      validity[coreIndex] =
        ctx.options.view === 'difference' ? 1 : lower > 0 ? 1 : upper < 0 ? 2 : 0;
    }
    valueBuffer.write(values);
    validityBuffer.write(validity);
  };

  const compileGraph = async (settings: ChangeSettings): Promise<ActiveGraph> => {
    const graph = new GPUCommandGraph<void>(device, {
      id: `point-cloud-change-${settings.alignment}-${settings.metric}-${settings.radius}-${settings.registrationError}-${settings.robust}-${settings.maximumDistance}`
    });
    const referenceView = importGraphBuffer(
      graph,
      'reference-epoch',
      referenceBuffer,
      'float32x3',
      referenceRows
    );
    const comparisonView = importGraphBuffer(
      graph,
      'comparison-epoch',
      comparisonBuffer,
      'float32x3',
      comparisonRows
    );
    const coresView = importGraphBuffer(
      graph,
      'provided-cores',
      coresBuffer,
      'float32x3',
      coreCount
    );
    const normalsView = importGraphBuffer(
      graph,
      'provided-normals',
      normalsBuffer,
      'float32x3',
      coreCount
    );
    const resourceSuffix = `${settings.alignment}-${settings.metric}-${settings.radius}-${settings.halfDepth}-${settings.registrationError}-${settings.robust}-${settings.maximumDistance}`;
    const ownedBuffers: Buffer[] = [];
    const own = (buffer: Buffer) => {
      ownedBuffers.push(buffer);
      return buffer;
    };
    const changeValues = own(
      coordinatesBuffer.createBuffer(`${resourceSuffix}-signed-changes`, coreCount * 12)
    );
    const roughnessValues = own(
      coordinatesBuffer.createBuffer(`${resourceSuffix}-roughness-and-limit`, coreCount * 12)
    );
    const countsValues = own(
      coordinatesBuffer.createBuffer(`${resourceSuffix}-epoch-support-counts`, coreCount * 8)
    );
    const statusValues = own(
      coordinatesBuffer.createBuffer(`${resourceSuffix}-measurement-status`, coreCount * 4)
    );
    const changesView = importGraphBuffer(
      graph,
      'signed-changes',
      changeValues,
      'float32x3',
      coreCount
    );
    const roughnessView = importGraphBuffer(
      graph,
      'roughness-and-limit',
      roughnessValues,
      'float32x3',
      coreCount
    );
    const countsView = importGraphBuffer(
      graph,
      'epoch-support-counts',
      countsValues,
      'uint32',
      coreCount * 2
    );
    const statusView = importGraphBuffer(
      graph,
      'measurement-status',
      statusValues,
      'uint32',
      coreCount
    );
    let alignment: 'already-aligned' | {transform: GraphDataView<'float32'>} = 'already-aligned';
    let icpResources: {
      transform: Buffer;
      history: Buffer;
      inlierMask: Buffer;
      status: Buffer;
      targetFeatureStatus: Buffer | null;
      targetNeighborhoodCounts: Buffer | null;
    } | null = null;

    if (settings.alignment === 'local-icp') {
      const sampledReference = new Float32Array(ICP_ROWS * 3);
      const sampledComparison = new Float32Array(ICP_ROWS * 3);
      for (let row = 0; row < ICP_ROWS; row++) {
        const referenceRow = Math.floor((row * referenceRows) / ICP_ROWS);
        const comparisonRow = Math.floor((row * comparisonRows) / ICP_ROWS);
        sampledReference.set(reference.subarray(referenceRow * 3, referenceRow * 3 + 3), row * 3);
        sampledComparison.set(
          comparison.subarray(comparisonRow * 3, comparisonRow * 3 + 3),
          row * 3
        );
      }
      const sourceBuffer = own(
        coordinatesBuffer.createBuffer(`${resourceSuffix}-icp-source-sample`, sampledComparison)
      );
      const targetBuffer = own(
        coordinatesBuffer.createBuffer(`${resourceSuffix}-icp-target-sample`, sampledReference)
      );
      const transformBuffer = own(
        coordinatesBuffer.createBuffer(
          `${resourceSuffix}-icp-transform`,
          new Float32Array(IDENTITY)
        )
      );
      const historyBuffer = own(
        coordinatesBuffer.createBuffer(`${resourceSuffix}-icp-history`, ICP_ITERATIONS * 12)
      );
      const inlierBuffer = own(
        coordinatesBuffer.createBuffer(`${resourceSuffix}-icp-inliers`, ICP_ROWS * 4)
      );
      const icpStatusBuffer = own(
        coordinatesBuffer.createBuffer(`${resourceSuffix}-icp-status`, 4)
      );
      const sourceView = importGraphBuffer(
        graph,
        'icp-source',
        sourceBuffer,
        'float32x3',
        ICP_ROWS
      );
      const targetView = importGraphBuffer(
        graph,
        'icp-target',
        targetBuffer,
        'float32x3',
        ICP_ROWS
      );
      let targetNormalsView: GraphDataView<'float32x3'> | undefined;
      let targetFeatureStatusBuffer: Buffer | null = null;
      let targetNeighborhoodCountsBuffer: Buffer | null = null;
      if (settings.metric === 'point-to-plane') {
        const neighborhoodCapacity = TARGET_NORMAL_CAPACITY;
        const neighborhoodIndicesBuffer = own(
          coordinatesBuffer.createBuffer(
            `${resourceSuffix}-target-neighborhood-indices`,
            ICP_ROWS * neighborhoodCapacity * 4
          )
        );
        const neighborhoodDistancesBuffer = own(
          coordinatesBuffer.createBuffer(
            `${resourceSuffix}-target-neighborhood-distances`,
            ICP_ROWS * neighborhoodCapacity * 4
          )
        );
        targetNeighborhoodCountsBuffer = own(
          coordinatesBuffer.createBuffer(
            `${resourceSuffix}-target-neighborhood-counts`,
            ICP_ROWS * 4
          )
        );
        const featureNormalsBuffer = own(
          coordinatesBuffer.createBuffer(`${resourceSuffix}-target-feature-normals`, ICP_ROWS * 12)
        );
        const targetNormalsBuffer = own(
          coordinatesBuffer.createBuffer(`${resourceSuffix}-target-valid-normals`, ICP_ROWS * 12)
        );
        const targetEigenvaluesBuffer = own(
          coordinatesBuffer.createBuffer(`${resourceSuffix}-target-eigenvalues`, ICP_ROWS * 12)
        );
        const targetShapesBuffer = own(
          coordinatesBuffer.createBuffer(`${resourceSuffix}-target-shapes`, ICP_ROWS * 12)
        );
        targetFeatureStatusBuffer = own(
          coordinatesBuffer.createBuffer(`${resourceSuffix}-target-feature-status`, ICP_ROWS * 4)
        );
        const neighborhoodIndicesView = importGraphBuffer(
          graph,
          'target-neighborhood-indices',
          neighborhoodIndicesBuffer,
          'uint32',
          ICP_ROWS * neighborhoodCapacity
        );
        const neighborhoodDistancesView = importGraphBuffer(
          graph,
          'target-neighborhood-distances',
          neighborhoodDistancesBuffer,
          'float32',
          ICP_ROWS * neighborhoodCapacity
        );
        const neighborhoodCountsView = importGraphBuffer(
          graph,
          'target-neighborhood-counts',
          targetNeighborhoodCountsBuffer,
          'uint32',
          ICP_ROWS
        );
        const featureNormalsView = importGraphBuffer(
          graph,
          'target-feature-normals',
          featureNormalsBuffer,
          'float32x3',
          ICP_ROWS
        );
        const targetValidNormalsView = importGraphBuffer(
          graph,
          'target-valid-normals',
          targetNormalsBuffer,
          'float32x3',
          ICP_ROWS
        );
        targetNormalsView = targetValidNormalsView;
        const targetEigenvaluesView = importGraphBuffer(
          graph,
          'target-eigenvalues',
          targetEigenvaluesBuffer,
          'float32x3',
          ICP_ROWS
        );
        const targetShapesView = importGraphBuffer(
          graph,
          'target-shapes',
          targetShapesBuffer,
          'float32x3',
          ICP_ROWS
        );
        const targetFeatureStatusView = importGraphBuffer(
          graph,
          'target-feature-status',
          targetFeatureStatusBuffer,
          'uint32',
          ICP_ROWS
        );
        graph.add(
          new GPUPointCloudNeighborhoods({
            id: 'icp-target-neighborhoods',
            positions: targetView,
            radius: TARGET_NORMAL_RADIUS,
            capacity: neighborhoodCapacity,
            indices: neighborhoodIndicesView,
            squaredDistances: neighborhoodDistancesView,
            counts: neighborhoodCountsView
          })
        );
        graph.add(
          new GPUPointCloudFeatures({
            id: 'icp-target-features',
            positions: targetView,
            indices: neighborhoodIndicesView,
            counts: neighborhoodCountsView,
            capacity: neighborhoodCapacity,
            normals: featureNormalsView,
            eigenvalues: targetEigenvaluesView,
            shapes: targetShapesView,
            status: targetFeatureStatusView
          })
        );
        const maskInvalidNormalsNode = createGPUComputeCommandNode<void>({
          id: 'icp-mask-invalid-target-normals',
          resources: [
            {buffer: targetFeatureStatusView, usage: 'storage-read'},
            {buffer: featureNormalsView, usage: 'storage-read'},
            {buffer: targetValidNormalsView, usage: 'storage-read-write'}
          ],
          compile: ({device: graphDevice}) => {
            const kernel = new Kernel(graphDevice, {
              id: 'icp-mask-invalid-target-normals',
              source: `@group(0) @binding(0) var<storage, read> featureStatus: array<u32>;
@group(0) @binding(1) var<storage, read> featureNormals: array<f32>;
@group(0) @binding(2) var<storage, read_write> targetNormals: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= ${ICP_ROWS}u) { return; }
  let base = index * 3u;
  if ((featureStatus[index] & 5u) != 0u) {
    targetNormals[base] = 0.0;
    targetNormals[base + 1u] = 0.0;
    targetNormals[base + 2u] = 0.0;
  } else {
    targetNormals[base] = featureNormals[base];
    targetNormals[base + 1u] = featureNormals[base + 1u];
    targetNormals[base + 2u] = featureNormals[base + 2u];
  }
}`,
              shaderLayout: {
                bindings: [
                  {name: 'featureStatus', type: 'read-only-storage', group: 0, location: 0},
                  {name: 'featureNormals', type: 'read-only-storage', group: 0, location: 1},
                  {name: 'targetNormals', type: 'storage', group: 0, location: 2}
                ]
              }
            });
            return {
              encode: ({computePass, getBuffer}) => {
                const bindings: Record<string, Binding> = {
                  featureStatus: getBuffer(targetFeatureStatusView),
                  featureNormals: getBuffer(featureNormalsView),
                  targetNormals: getBuffer(targetValidNormalsView)
                };
                kernel.dispatch(computePass, {
                  bindings,
                  x: Math.ceil(ICP_ROWS / 256),
                  y: 1,
                  z: 1
                });
              },
              destroy: () => kernel.destroy()
            };
          }
        });
        graph.add(maskInvalidNormalsNode);
      }
      icpResources = {
        transform: transformBuffer,
        history: historyBuffer,
        inlierMask: inlierBuffer,
        status: icpStatusBuffer,
        targetFeatureStatus: targetFeatureStatusBuffer,
        targetNeighborhoodCounts: targetNeighborhoodCountsBuffer
      };
      const transformView = importGraphBuffer(
        graph,
        'icp-transform',
        transformBuffer,
        'float32',
        16
      );
      const historyView = importGraphBuffer(
        graph,
        'icp-history',
        historyBuffer,
        'float32x3',
        ICP_ITERATIONS
      );
      const inlierView = importGraphBuffer(graph, 'icp-inliers', inlierBuffer, 'uint32', ICP_ROWS);
      const icpStatusView = importGraphBuffer(graph, 'icp-status', icpStatusBuffer, 'uint32', 1);
      const icpInputs = {
        source: sourceView,
        target: targetView,
        maximumDistance: settings.maximumDistance,
        iterations: ICP_ITERATIONS,
        ...(settings.robust === 'none' ? {} : {robust: {method: settings.robust, scale: 0.5}}),
        transform: transformView,
        history: historyView,
        inlierMask: inlierView,
        status: icpStatusView
      };
      if (settings.metric === 'point-to-plane') {
        if (!targetNormalsView)
          throw new Error('Point-to-plane ICP requires computed target normals.');
        graph.add(
          new GPUPointCloudICP({
            id: 'local-icp',
            ...icpInputs,
            metric: 'point-to-plane',
            targetNormals: targetNormalsView
          })
        );
      } else {
        graph.add(new GPUPointCloudICP({id: 'local-icp', ...icpInputs, metric: 'point-to-point'}));
      }
      alignment = {transform: transformView};
    }
    graph.add(
      new GPUPointCloudChange({
        id: 'core-change',
        reference: referenceView,
        comparison: comparisonView,
        cores: coresView,
        normals: normalsView,
        alignment,
        radius: settings.radius,
        halfDepth: settings.halfDepth,
        minimumSupport: 3,
        registrationError: settings.registrationError,
        uncertaintyCombination: settings.uncertaintyCombination,
        confidenceMultiplier: 1.96,
        changes: changesView,
        roughness: roughnessView,
        counts: countsView,
        status: statusView
      })
    );
    const compiled = coordinatesBuffer.track(await graph.compileAsync());
    return {
      compiled,
      settings,
      changeValueBuffer: changeValues,
      roughnessValueBuffer: roughnessValues,
      countsValueBuffer: countsValues,
      statusValueBuffer: statusValues,
      icpResources,
      ownedBuffers
    };
  };

  currentGraph = await compileGraph({
    alignment: ctx.options.alignment,
    metric: ctx.options.metric,
    radius: ctx.options.radius,
    halfDepth: ctx.options.halfDepth,
    registrationError: ctx.options.registrationError,
    uncertaintyCombination: ctx.options.uncertaintyCombination,
    robust: ctx.options.robust,
    maximumDistance: ctx.options.maximumDistance
  });
  type FullGraph = ActiveGraph;

  const sourceViews = (active: FullGraph) => [
    {buffer: active.changeValueBuffer, size: coreCount * 12},
    {buffer: active.roughnessValueBuffer, size: coreCount * 12},
    {buffer: active.countsValueBuffer, size: coreCount * 8},
    {buffer: active.statusValueBuffer, size: coreCount * 4},
    ...(active.icpResources
      ? [
          {buffer: active.icpResources.history, size: ICP_ITERATIONS * 12},
          {buffer: active.icpResources.status, size: 4},
          {buffer: active.icpResources.inlierMask, size: ICP_ROWS * 4},
          ...(active.icpResources.targetFeatureStatus
            ? [{buffer: active.icpResources.targetFeatureStatus, size: ICP_ROWS * 4}]
            : []),
          ...(active.icpResources.targetNeighborhoodCounts
            ? [{buffer: active.icpResources.targetNeighborhoodCounts, size: ICP_ROWS * 4}]
            : [])
        ]
      : [])
  ];
  let summaryReader: SummaryReader;
  const makeSummaryReader = (active: FullGraph): SummaryReader =>
    new SummaryReader(
      coordinatesBuffer,
      'point-cloud-change-summary',
      sourceViews(active),
      bytes => {
        let offset = 0;
        const readFloats = (length: number) => {
          const result = new Float32Array(bytes, offset, length);
          offset += length * 4;
          return result;
        };
        const readIntegers = (length: number) => {
          const result = new Uint32Array(bytes, offset, length);
          offset += length * 4;
          return result;
        };
        latestChanges = readFloats(coreCount * 3);
        // Roughness remains in the packed GPU summary for reference checks; skip its unused view.
        offset += coreCount * 3 * Float32Array.BYTES_PER_ELEMENT;
        latestCounts = readIntegers(coreCount * 2);
        latestStatuses = readIntegers(coreCount);
        if (active.icpResources) {
          latestIcpHistory = readFloats(ICP_ITERATIONS * 3);
          latestIcpStatus = readIntegers(1)[0];
          latestIcpInliers = readIntegers(ICP_ROWS);
          latestTargetFeatureStatus = active.icpResources.targetFeatureStatus
            ? readIntegers(ICP_ROWS)
            : new Uint32Array(ICP_ROWS);
          latestTargetNeighborhoodCounts = active.icpResources.targetNeighborhoodCounts
            ? readIntegers(ICP_ROWS)
            : new Uint32Array(ICP_ROWS);
        } else {
          latestIcpHistory = new Float32Array(ICP_ITERATIONS * 3);
          latestIcpStatus = 0;
          latestIcpInliers = new Uint32Array(ICP_ROWS);
          latestTargetFeatureStatus = new Uint32Array(ICP_ROWS);
          latestTargetNeighborhoodCounts = new Uint32Array(ICP_ROWS);
        }
        currentAlignmentValid = active.settings.alignment !== 'local-icp' || latestIcpStatus <= 1;
        const values = new Float32Array(coreCount);
        const validity = new Uint32Array(coreCount);
        let invalidNormals = 0;
        let insufficientSupport = 0;
        let validCores = 0;
        const referenceSupport: number[] = [];
        const comparisonSupport: number[] = [];
        let detectableCores = 0;
        const differences: number[] = [];
        const differenceBins = new Array<number>(20).fill(0);
        const limits: number[] = [];
        const lowerBounds: number[] = [];
        const upperBounds: number[] = [];
        for (let coreIndex = 0; coreIndex < coreCount; coreIndex++) {
          const status = latestStatuses[coreIndex];
          if (status === 1) {
            invalidNormals++;
            validity[coreIndex] = ctx.options.view === 'detection' ? 3 : 0;
            continue;
          }
          if (status === 2) {
            insufficientSupport++;
            validity[coreIndex] = ctx.options.view === 'detection' ? 3 : 0;
            continue;
          }
          if (status !== 0 || !currentAlignmentValid) {
            validity[coreIndex] = ctx.options.view === 'detection' ? 3 : 0;
            continue;
          }
          validCores++;
          referenceSupport.push(latestCounts[coreIndex * 2]);
          comparisonSupport.push(latestCounts[coreIndex * 2 + 1]);
          const difference = latestChanges[coreIndex * 3];
          const lower = latestChanges[coreIndex * 3 + 1];
          const upper = latestChanges[coreIndex * 3 + 2];
          const limit = Math.max(difference - lower, upper - difference);
          differences.push(difference);
          const binIndex = clamp(Math.floor(((difference + 1) / 2) * 20), 0, 19);
          differenceBins[binIndex]++;
          limits.push(limit);
          lowerBounds.push(lower);
          upperBounds.push(upper);
          if (lower > 0 || upper < 0) detectableCores++;
          values[coreIndex] =
            ctx.options.view === 'difference' ? difference : lower > 0 ? 1 : upper < 0 ? 2 : 0;
          validity[coreIndex] =
            ctx.options.view === 'difference' ? 1 : lower > 0 ? 1 : upper < 0 ? 2 : 0;
        }
        valueBuffer.write(values);
        validityBuffer.write(validity);
        if (active.icpResources) {
          let usableTargetNormals = 0;
          let weakTargetNormals = 0;
          let truncatedTargetNormals = 0;
          let maximumTargetSupport = 0;
          if (active.settings.metric === 'point-to-plane') {
            for (let row = 0; row < ICP_ROWS; row++) {
              const featureStatus = latestTargetFeatureStatus[row];
              if ((featureStatus & 5) === 0) usableTargetNormals++;
              else weakTargetNormals++;
              if ((featureStatus & 2) !== 0) truncatedTargetNormals++;
              maximumTargetSupport = Math.max(
                maximumTargetSupport,
                latestTargetNeighborhoodCounts[row]
              );
            }
          }
          ctx.setReadout(
            'targetNormalsReady',
            active.settings.metric === 'point-to-plane' ? usableTargetNormals : 'Not estimated'
          );
          ctx.setReadout(
            'targetNormalsWeak',
            active.settings.metric === 'point-to-plane' ? weakTargetNormals : 'Not estimated'
          );
          ctx.setReadout(
            'targetNormalsTruncated',
            active.settings.metric === 'point-to-plane' ? truncatedTargetNormals : 'Not estimated'
          );
          ctx.setReadout(
            'targetMaximumSupport',
            active.settings.metric === 'point-to-plane' ? maximumTargetSupport : 'Not estimated'
          );
          ctx.setChart('icpChart', {
            kind: 'line',
            title: `${active.settings.metric === 'point-to-plane' ? 'Point-to-plane residual' : 'Point-to-point RMSE'} and translation step`,
            xLabel: 'iteration',
            yLabel:
              active.settings.metric === 'point-to-plane'
                ? 'point-to-plane residual (m)'
                : 'point-to-point RMSE (m)',
            y2Label: 'Translation step (m)',
            description:
              'Point-to-plane residual and point-to-point RMSE are different objectives; compare each only within its own metric.',
            series: [
              {
                label:
                  active.settings.metric === 'point-to-plane' ? 'Plane residual' : 'Point RMSE',
                y: Array.from(
                  {length: ICP_ITERATIONS},
                  (_, index) => latestIcpHistory[index * 3 + 1]
                ),
                color: 1,
                points: true
              },
              {
                label: 'Translation step (m)',
                y: Array.from(
                  {length: ICP_ITERATIONS},
                  (_, index) => latestIcpHistory[index * 3 + 2]
                ),
                color: 3,
                points: true,
                axis: 'y2'
              }
            ]
          });
          ctx.setChart('icpInlierChart', {
            kind: 'bars',
            title: 'ICP inlier count',
            labels: Array.from({length: ICP_ITERATIONS}, (_, index) => `${index + 1}`),
            values: Array.from({length: ICP_ITERATIONS}, (_, index) => latestIcpHistory[index * 3]),
            color: 2,
            xLabel: 'iteration',
            yLabel: 'rows'
          } satisfies BarChartData);
          ctx.setReadout(
            'icpFinalInliers',
            latestIcpInliers.reduce((total, value) => total + value, 0)
          );
          ctx.setReadout('icpFinalStep', latestIcpHistory[(ICP_ITERATIONS - 1) * 3 + 2]);
        } else {
          ctx.setChart('icpChart', null);
          ctx.setChart('icpInlierChart', null);
          ctx.setReadout('icpFinalInliers', null);
          ctx.setReadout('icpFinalStep', null);
          ctx.setReadout('targetNormalsReady', null);
          ctx.setReadout('targetNormalsWeak', null);
          ctx.setReadout('targetNormalsTruncated', null);
          ctx.setReadout('targetMaximumSupport', null);
        }
        if (currentAlignmentValid) {
          const median = (numbers: number[]) => {
            const sorted = [...numbers].sort((a, b) => a - b);
            return sorted.length
              ? (sorted[Math.floor((sorted.length - 1) / 2)] +
                  sorted[Math.floor(sorted.length / 2)]) /
                  2
              : null;
          };
          ctx.setReadout('validCores', validCores);
          ctx.setReadout('invalidNormals', invalidNormals);
          ctx.setReadout('insufficientSupport', insufficientSupport);
          ctx.setReadout('detectableCores', detectableCores);
          ctx.setReadout('medianReferenceSupport', median(referenceSupport));
          ctx.setReadout('medianComparisonSupport', median(comparisonSupport));
          ctx.setReadout('medianDifference', median(differences));
          ctx.setReadout('medianLimit', median(limits));
          ctx.setReadout(
            'icpStatus',
            active.settings.alignment === 'surveyed'
              ? 'Supplied survey frame'
              : latestIcpStatus === 0
                ? `Converged · exploratory ${active.settings.metric}`
                : `Iteration budget exhausted · provisional exploratory ${active.settings.metric} fit`
          );
          ctx.setChart('differenceChart', {
            kind: 'histogram',
            title: 'Signed core differences',
            values: differenceBins,
            xDomain: [-1, 1],
            breaks: Array.from({length: 21}, (_, index) => -1 + index * 0.1),
            xLabel: '2018A − 2017 (m)',
            yLabel: 'cores',
            description:
              'Fixed bins from −1 m to +1 m; edge bins include values beyond the displayed interval.'
          } satisfies BarChartData);
          ctx.setChart('boundChart', {
            kind: 'line',
            title: 'Uncertainty interval by supported core',
            xLabel: 'supported core rank',
            yLabel: 'metres',
            yDomain: [-1, 1],
            series: [
              {label: 'Lower bound', y: lowerBounds, color: 0, points: false},
              {label: 'Upper bound', y: upperBounds, color: 3, points: false}
            ]
          });
          ctx.setChart('intervalChart', {
            kind: 'bars',
            title: 'Intervals relative to zero',
            labels: ['Includes zero', 'Excludes zero'],
            values: [validCores - detectableCores, detectableCores],
            color: 2,
            description: 'Counts derived from the GPU lower and upper bounds.'
          } satisfies BarChartData);
        } else {
          ctx.setReadout('validCores', 'ICP failed · measurements rejected');
          ctx.setReadout('invalidNormals', invalidNormals);
          ctx.setReadout('insufficientSupport', insufficientSupport);
          ctx.setReadout('detectableCores', 'ICP failed · result rejected');
          ctx.setReadout('medianDifference', null);
          ctx.setReadout('medianLimit', null);
          ctx.setReadout('medianReferenceSupport', null);
          ctx.setReadout('medianComparisonSupport', null);
          ctx.setReadout(
            'icpStatus',
            `Status ${latestIcpStatus} · ${active.settings.metric} transform rejected`
          );
          ctx.setReadout(
            'icpFinalInliers',
            latestIcpInliers.reduce((total, value) => total + value, 0)
          );
          ctx.setReadout('icpFinalStep', latestIcpHistory[(ICP_ITERATIONS - 1) * 3 + 2]);
          ctx.setChart('differenceChart', null);
          ctx.setChart('intervalChart', null);
          ctx.setChart('boundChart', null);
        }
        ctx.requestLayers();
      }
    );
  summaryReader = makeSummaryReader(currentGraph as FullGraph);
  ctx.setReadout('validCores', 'Waiting for GPU result');
  ctx.setReadout('invalidNormals', 'Waiting for GPU result');
  ctx.setReadout('insufficientSupport', 'Waiting for GPU result');
  ctx.setReadout('icpStatus', 'Supplied survey frame');
  const getSettings = (state: PointCloudChangeOptions): ChangeSettings => ({
    alignment: state.alignment,
    metric: state.metric,
    radius: state.radius,
    halfDepth: state.halfDepth,
    registrationError: state.registrationError,
    uncertaintyCombination: state.uncertaintyCombination,
    robust: state.robust,
    maximumDistance: state.maximumDistance
  });
  const releaseGraph = (graph: ActiveGraph) => {
    coordinatesBuffer.release(graph.compiled);
    for (const buffer of graph.ownedBuffers) coordinatesBuffer.release(buffer);
  };
  const requestGraph = (settings: ChangeSettings) => {
    void graphSwap
      .request({
        build: () => compileGraph(settings),
        commit: next => {
          const previous = currentGraph as FullGraph;
          summaryReader.release();
          currentGraph = next;
          summaryReader = makeSummaryReader(currentGraph as FullGraph);
          releaseGraph(previous);
          graphDirty = true;
          currentAlignmentValid = settings.alignment === 'surveyed';
          valueBuffer.write(new Float32Array(coreCount));
          validityBuffer.write(new Uint32Array(coreCount));
          ctx.setReadout('validCores', 'Recomputing');
          ctx.setReadout('detectableCores', 'Recomputing');
          ctx.setReadout(
            'icpStatus',
            settings.alignment === 'surveyed'
              ? 'Supplied survey frame'
              : `Fitting local ${settings.metric} ICP`
          );
          ctx.setReadout('medianDifference', null);
          ctx.setReadout('medianLimit', null);
          ctx.setReadout('medianReferenceSupport', null);
          ctx.setReadout('medianComparisonSupport', null);
          ctx.requestLayers();
        },
        discard: next => releaseGraph(next)
      })
      .catch(error => ctx.setStatus(`Graph rebuild failed: ${String(error)}`));
  };

  return {
    getCompiledGraphs: () => [currentGraph.compiled],
    setOption(id, _value, state) {
      if (id === 'view') updateDisplayValues();
      if (
        [
          'alignment',
          'metric',
          'radius',
          'halfDepth',
          'registrationError',
          'uncertaintyCombination',
          'robust',
          'maximumDistance'
        ].includes(id)
      )
        requestGraph(getSettings(state));
      else ctx.requestLayers();
    },
    encode(commandEncoder) {
      if (graphDirty) {
        currentGraph.compiled.encode(commandEncoder, {parameters: undefined});
        summaryReader.request(commandEncoder);
        graphDirty = false;
      } else summaryReader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const common = {
        coordinateOrigin: scalarOrigin,
        radiusPixels: ctx.options.pointSize,
        opacity: 0.94
      };
      if (ctx.options.view === 'epochs')
        return [
          new SpatialAnalysisPointLayer({
            id: 'rock-glacier-reference-returns',
            positions: referenceDisplayBuffer,
            color: [98, 164, 207, 230],
            instanceCount: referenceRows,
            radiusPixels: 1.5,
            coordinateOrigin: scalarOrigin,
            opacity: 0.9
          }),
          new SpatialAnalysisPointLayer({
            id: 'rock-glacier-comparison-returns',
            positions: comparisonDisplayBuffer,
            color: [225, 147, 112, 220],
            instanceCount: comparisonRows,
            radiusPixels: 1.5,
            coordinateOrigin: scalarOrigin,
            opacity: 0.9
          }),
          new SpatialAnalysisPointLayer({
            id: 'rock-glacier-core-locations',
            positions: corePositionBuffer,
            color: [244, 203, 122, 255],
            instanceCount: coreCount,
            shape: 'ring',
            radiusPixels: ctx.options.pointSize,
            coordinateOrigin: scalarOrigin,
            opacity: 1
          })
        ];
      if (ctx.options.view === 'detection')
        return [
          new SpatialAnalysisPointLayer({
            id: 'rock-glacier-detection-status',
            positions: corePositionBuffer,
            values: validityBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [75, 83, 91, 255],
              [113, 195, 157, 255],
              [225, 133, 105, 255],
              [174, 180, 185, 255]
            ],
            instanceCount: coreCount,
            ...common
          })
        ];
      return [
        new SpatialAnalysisPointLayer({
          id: 'rock-glacier-signed-change',
          positions: corePositionBuffer,
          values: valueBuffer,
          valueFormat: 'float32',
          colormap: 'vik',
          valueRange: [-1, 1],
          instanceCount: coreCount,
          ...common
        }),
        new SpatialAnalysisPointLayer({
          id: 'rock-glacier-invalid-neutral',
          positions: corePositionBuffer,
          values: validityBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: [
            [104, 112, 122, 255],
            [0, 0, 0, 0]
          ],
          instanceCount: coreCount,
          radiusPixels: ctx.options.pointSize + 1,
          coordinateOrigin: scalarOrigin,
          opacity: 1
        })
      ];
    },
    destroy() {
      summaryReader.release();
      coordinatesBuffer.destroy();
    }
  };
}
