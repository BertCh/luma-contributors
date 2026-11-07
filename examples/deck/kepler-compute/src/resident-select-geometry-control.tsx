// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createContext, useContext, useEffect, useRef, useState, type ComponentProps} from 'react';
import {Button, MapPopoverFactory} from '@kepler.gl/components';
import {ResidentHoverIdentity} from './resident-hover-identity';
import type {ResidentSelectedGeometryResult} from './resident-select-geometry';

type GeometryIdentity = {
  datasetId: string;
  layerId: string;
  sourceRowIndex: number;
  featureId: string;
};
export type ResidentGeometryReader = (
  identity: GeometryIdentity
) => Promise<ResidentSelectedGeometryResult>;
type SelectionControl = {key: string; select(isActive: () => boolean): Promise<void>};
const SelectionContext = createContext<SelectionControl | null>(null);

/** The native frozen popover supplies editor actions; source materialization stays bounded. */
export function createResidentSelectGeometryFactory(options: {
  isResidentDataset(datasetId: string): boolean;
  readGeometry: ResidentGeometryReader;
  /** Index of the native map that owns the frozen popover (1 for the second split view). */
  getMapIndex(): number;
}) {
  function ResidentSelectGeometryFactory(...dependencies: Parameters<typeof MapPopoverFactory>) {
    const NativePopover = MapPopoverFactory(...dependencies);
    function ResidentSelectGeometry(props: ComponentProps<typeof NativePopover>) {
      const layer = props.layerHoverProp?.layer;
      const row = props.layerHoverProp?.data;
      const datasetId = layer?.config.dataId;
      const eligible =
        props.frozen &&
        datasetId &&
        options.isResidentDataset(datasetId) &&
        (layer.type === 'geojson' || layer.type === 'hexagonId') &&
        row instanceof ResidentHoverIdentity;
      const selection: SelectionControl | null = eligible
        ? {
            key: `${datasetId}:${layer.id}:${row.sourceRowIndex}`,
            async select(isActive) {
              const featureId = crypto.randomUUID();
              const result = await options.readGeometry({
                datasetId,
                layerId: layer.id,
                sourceRowIndex: row.sourceRowIndex,
                featureId
              });
              if (!isActive()) return;
              if (result.status !== 'ready') throw new Error(result.reason);
              if (!result.isCurrent())
                throw new Error(
                  'The selected source or layer changed. Click its current geometry again.'
                );
              const geometry = result.feature.geometry;
              if (geometry.type === 'GeometryCollection')
                throw new Error(
                  'The native editor requires one coordinate geometry, not a GeometryCollection.'
                );
              const feature = {...result.feature, geometry, id: featureId};
              props.setSelectedFeature(feature, {
                mapIndex: options.getMapIndex(),
                rightClick: true,
                position: {x: props.x, y: props.y}
              });
              props.onSetFeatures([...(props.featureCollection?.features ?? []), feature]);
              props.onClose();
            }
          }
        : null;
      return (
        <SelectionContext.Provider value={selection}>
          <NativePopover {...props} />
        </SelectionContext.Provider>
      );
    }
    ResidentSelectGeometry.WrappedComponent = NativePopover.WrappedComponent;
    return ResidentSelectGeometry;
  }
  ResidentSelectGeometryFactory.deps = MapPopoverFactory.deps;
  return ResidentSelectGeometryFactory;
}

export function ResidentSelectGeometryControl() {
  const selection = useContext(SelectionContext);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const generation = useRef(0);
  useEffect(() => {
    generation.current++;
    setPending(false);
    setError('');
    return () => {
      generation.current++;
    };
  }, [selection?.key]);
  if (!selection) return null;
  return (
    <div>
      <Button
        type="button"
        disabled={pending}
        onClick={() => {
          const request = ++generation.current;
          setPending(true);
          setError('');
          const isActive = () => request === generation.current;
          void selection
            .select(isActive)
            .catch(failure => {
              if (isActive())
                setError(failure instanceof Error ? failure.message : String(failure));
            })
            .finally(() => {
              if (isActive()) setPending(false);
            });
        }}
      >
        {pending ? 'Preparing geometry…' : 'Select Geometry'}
      </Button>
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
