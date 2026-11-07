// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ComponentProps} from 'react';
import {Icons, MapControlButton, MapControlFactory} from '@kepler.gl/components';

type NativeMapControlProps = ComponentProps<ReturnType<typeof MapControlFactory>>;

function ResidentChartControl({mapControls, onToggleMapControl}: NativeMapControlProps) {
  if (!mapControls.chart?.show) return null;
  return (
    <MapControlButton
      className="map-control-button toggle-chart-panel"
      aria-label="Toggle charts"
      title="Toggle charts"
      aria-pressed={Boolean(mapControls.chart.active)}
      active={Boolean(mapControls.chart.active)}
      onClick={() => onToggleMapControl('chart')}
    >
      <Icons.LineChart height="22px" />
    </MapControlButton>
  );
}

/** Register the chart action in Kepler's native map-control layout and theme. */
export function ResidentMapControlFactory(...dependencies: Parameters<typeof MapControlFactory>) {
  const MapControl = MapControlFactory(...dependencies);
  const defaultActionComponents = [
    ...(MapControl.defaultActionComponents ?? []),
    ResidentChartControl
  ];
  function ResidentMapControl(props: NativeMapControlProps) {
    return (
      <MapControl
        {...props}
        actionComponents={
          props.actionComponents
            ? [...props.actionComponents, ResidentChartControl]
            : defaultActionComponents
        }
      />
    );
  }
  ResidentMapControl.defaultActionComponents = defaultActionComponents;
  return ResidentMapControl;
}
ResidentMapControlFactory.deps = MapControlFactory.deps;
