# Lagrangian diagnostics (time-varying fields, FTLE, connectivity, residence time)

Import from `@luma-contributors/analysis/raster-analysis`.

The canonical contracts and generated parameter tables live in
[Flow-map and particle diagnostics](./flow-map-and-particle-diagnostics.md).
This historical route preserves its incoming fragments as links; recording capability IDs
remain on the canonical guide.

## Choose a flow diagnostic

Use the [task chooser](./flow-map-and-particle-diagnostics.md#choose-a-flow-diagnostic).

<a id="gpulagrangianflowmap"></a>

## Flow-map contract

Read [GPULagrangianFlowMap](./flow-map-and-particle-diagnostics.md#gpulagrangianflowmap),
including support, positions, regions and residence-time semantics.

<a id="gpulagrangianflowmap-parameters"></a>

### Flow-map parameters

Read the [generated parameter table](./flow-map-and-particle-diagnostics.md#gpulagrangianflowmap-parameters).

### Preserve local displacement and slice times

Use the [local integration and time-origin example](./flow-map-and-particle-diagnostics.md#preserve-local-displacement-and-slice-times).

<a id="gpufinitetimelyapunovexponent"></a>

## FTLE contract

Read [GPUFiniteTimeLyapunovExponent](./flow-map-and-particle-diagnostics.md#gpufinitetimelyapunovexponent).

<a id="gpufinitetimelyapunovexponent-parameters"></a>

### FTLE parameters

Read the [generated parameter table](./flow-map-and-particle-diagnostics.md#gpufinitetimelyapunovexponent-parameters).

<a id="gpuscalarfieldridges"></a>

## Scalar-ridge contract and recordings

Read [GPUScalarFieldRidges and its recordings](./flow-map-and-particle-diagnostics.md#gpuscalarfieldridges).

<a id="gpuscalarfieldridges-parameters"></a>

### Scalar-ridge parameters

Read the [generated parameter table](./flow-map-and-particle-diagnostics.md#gpuscalarfieldridges-parameters).

## Connectivity matrix

Import the connectivity consumer from its spatial entry:

```ts
import {GPUTransitionMatrix} from '@luma-contributors/analysis/gpu-spatial-analysis';
```

Basic counts and probabilities accept up to **65,535 classes**, subject to dense buffer/device
bounds. Optional diagnostics retain the **64-class** limit. See the
[canonical connectivity recipe](./flow-map-and-particle-diagnostics.md#connectivity-matrix)
for period-major inputs, output layout, ignored transitions and complete wiring.
