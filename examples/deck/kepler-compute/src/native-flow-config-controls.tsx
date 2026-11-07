// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useSyncExternalStore, type ComponentProps} from 'react';
import {
  LayerConfiguratorFactory,
  PanelLabel,
  SidePanelSection,
  Switch
} from '@kepler.gl/components';
import {getFlowmapStats, subscribeFlowmapStats, type FlowmapStats} from './flowmap-gpu-layer';

type NativeConfiguratorProps = ComponentProps<ReturnType<typeof LayerConfiguratorFactory>>;

const FLOW_EXTENSION_SWITCHES = [
  {property: 'flowLocationLabelsEnabled', label: 'Location labels'},
  {property: 'flowScaleLockEnabled', label: 'Lock scales'}
] as const;

let statsSnapshot = '';
function readStatsSnapshot(): string {
  const next = JSON.stringify(getFlowmapStats());
  if (next !== statsSnapshot) statsSnapshot = next;
  return statsSnapshot;
}

function formatDomain(domain: [number, number] | undefined): string {
  return domain ? `${domain[0].toLocaleString()} to ${domain[1].toLocaleString()}` : 'none';
}

/** Scale-lock indicator: shows whether the flow and location-total domains are frozen. */
export function FlowScaleLockIndicator(props: {layerId: string; enabled: boolean}) {
  const snapshot = useSyncExternalStore(subscribeFlowmapStats, readStatsSnapshot);
  if (!props.enabled) return null;
  const stats = (JSON.parse(snapshot) as Record<string, FlowmapStats>)[props.layerId];
  return (
    <SidePanelSection>
      <div data-testid="flow-scale-lock-indicator" style={{fontSize: 11, opacity: 0.85}}>
        {stats?.scaleLocked
          ? `Scales locked. Flow magnitude ${formatDomain(stats.flowMagnitudeDomain)}; location totals ${formatDomain(stats.locationTotalsDomain)}.`
          : 'Scales will lock to the next computed domains.'}
      </div>
    </SidePanelSection>
  );
}

/**
 * Wraps Kepler's layer configurator and appends the host-only flow map switches (location labels,
 * scale lock) for flow layers. Provided at LayerConfiguratorFactory so the native panel stays
 * otherwise untouched; the switches write ordinary visConfig keys through updateLayerVisConfig.
 */
export function createFlowConfiguratorFactory() {
  function FlowConfiguratorFactory(...dependencies: Parameters<typeof LayerConfiguratorFactory>) {
    const NativeConfigurator = LayerConfiguratorFactory(...dependencies);
    function FlowConfigurator(props: NativeConfiguratorProps) {
      const {layer, updateLayerVisConfig} = props as unknown as {
        layer: {id: string; type: string; config: {visConfig: Record<string, unknown>}};
        updateLayerVisConfig: (change: Record<string, boolean>) => void;
      };
      return (
        <>
          <NativeConfigurator {...props} />
          {layer?.type === 'flow' ? (
            <div data-testid="flow-extension-controls" style={{paddingLeft: 12}}>
              {FLOW_EXTENSION_SWITCHES.map(({property, label}) => (
                <SidePanelSection key={property}>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <PanelLabel>{label}</PanelLabel>
                    <Switch
                      checked={Boolean(layer.config.visConfig[property])}
                      id={`${layer.id}-${property}-switch`}
                      onChange={() =>
                        updateLayerVisConfig({[property]: !layer.config.visConfig[property]})
                      }
                    />
                  </div>
                </SidePanelSection>
              ))}
              <FlowScaleLockIndicator
                layerId={layer.id}
                enabled={Boolean(layer.config.visConfig['flowScaleLockEnabled'])}
              />
            </div>
          ) : null}
        </>
      );
    }
    return FlowConfigurator;
  }
  FlowConfiguratorFactory.deps = LayerConfiguratorFactory.deps;
  return FlowConfiguratorFactory;
}
