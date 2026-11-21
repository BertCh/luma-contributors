// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUAnalysisOperatorCatalog,
  getGPUAnalysisCapabilityCatalog
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogOperationBinding} from './catalog-binding-types';
import type {CatalogEntry} from './catalog-bindings';
import {getCatalogIdentityKey} from './catalog-identity';
import {groupCatalogLensScenarios, type CatalogLensScenario} from './catalog-lens';

type Identity = CatalogEntry['identity'];

// These bound contributors are imported from this published upstream entry in
// catalog-platform5-raster.ts; they are intentionally outside the analysis-owned catalog.
const UPSTREAM_CATALOG_IDENTITIES = new Map<string, Identity>(
  ['GPURasterNeighborhood', 'GPURasterConnectedComponents'].map(exportName => [
    exportName,
    {package: '@luma.gl/graph/gpu-raster', exportName}
  ])
);

const NOT_BOUND_REASON =
  'No executable host binding or lens scenario is registered for this package and export. Computation-pack operator and formula entries describe it but do not execute it.';

const LENS_NOT_LOADED_REASON =
  ' The lens scenario registry is empty here; loading it exposes embedded-dataset scenarios for contributors that have one.';

/** Qualified discovery never guesses a package for a colliding bare export name. */
export function listQualifiedCatalogEntries(
  bindings: readonly CatalogOperationBinding[],
  scenarios: readonly CatalogLensScenario[],
  nativePanels: readonly (Identity & {location: string})[]
): CatalogEntry[] {
  const operators = getGPUAnalysisOperatorCatalog();
  const capabilities = new Map(
    getGPUAnalysisCapabilityCatalog().map(capability => [capability.id, capability])
  );
  const identities = new Map<
    string,
    {identity: Identity; capabilityId: string; title: string; domain: string}
  >(
    operators.map(operator => [
      getCatalogIdentityKey(operator),
      {
        identity: {package: operator.package, exportName: operator.exportName},
        capabilityId: operator.primaryCapability,
        title: capabilities.get(operator.primaryCapability)?.title ?? operator.primaryCapability,
        domain: capabilities.get(operator.primaryCapability)?.domain ?? 'analysis'
      }
    ])
  );
  const bindingDeclarations = bindings.map(binding => ({
    binding,
    explicitIdentities: [
      ...(binding.contributorIdentities ?? []),
      ...(binding.stages?.map(stage => stage.identity) ?? [])
    ]
  }));

  // Deduplicate canonical and explicit identities before checking whether a name is unique.
  const candidates = new Map([...identities].map(([key, entry]) => [key, entry.identity]));
  for (const {explicitIdentities} of bindingDeclarations)
    for (const identity of explicitIdentities)
      candidates.set(getCatalogIdentityKey(identity), identity);
  const uniqueIdentitiesByName = new Map<string, Identity | null>();
  for (const identity of candidates.values()) {
    const {exportName} = identity;
    uniqueIdentitiesByName.set(
      exportName,
      uniqueIdentitiesByName.has(exportName) ? null : identity
    );
  }

  const bindingsByIdentity = new Map<string, CatalogOperationBinding[]>();
  for (const {binding, explicitIdentities} of bindingDeclarations) {
    const explicitNames = new Set(explicitIdentities.map(identity => identity.exportName));
    const resolved = new Map(
      explicitIdentities.map(identity => [getCatalogIdentityKey(identity), identity])
    );
    const names = new Set([...binding.contributorNames, ...binding.gpuContributorNames]);
    for (const name of names) {
      if (explicitNames.has(name)) continue;
      const identity = UPSTREAM_CATALOG_IDENTITIES.get(name) ?? uniqueIdentitiesByName.get(name);
      if (identity) resolved.set(getCatalogIdentityKey(identity), identity);
    }
    for (const [key, identity] of resolved) {
      if (!identities.has(key))
        identities.set(key, {
          identity: {...identity},
          capabilityId: binding.capabilityIds[0] ?? binding.domain,
          title: binding.title,
          domain: binding.domain
        });
      const matches = bindingsByIdentity.get(key) ?? [];
      matches.push(binding);
      bindingsByIdentity.set(key, matches);
    }
  }
  const scenariosByIdentity = groupCatalogLensScenarios(scenarios);
  const locations = new Map(
    nativePanels.map(panel => [getCatalogIdentityKey(panel), panel.location])
  );

  return [...identities].map<CatalogEntry>(([key, {identity, capabilityId, title, domain}]) => {
    const matches = bindingsByIdentity.get(key) ?? [];
    const lensScenarioIds = scenariosByIdentity.get(key)?.map(scenario => scenario.id);
    const base = {
      identity,
      contributorName: identity.exportName,
      capabilityId,
      capabilityTitle: title,
      domain,
      bindingIds: matches.map(binding => binding.id),
      ...(lensScenarioIds ? {lensScenarioIds} : {})
    };
    if (matches.length) {
      let gpuAvailable = false;
      let cpuAvailable = false;
      for (const binding of matches) {
        const stages = binding.stages?.filter(
          stage => stage.identity.exportName === identity.exportName
        );
        const usesGpu =
          binding.engine !== 'cpu-reference' &&
          (stages?.length
            ? stages.some(
                stage => getCatalogIdentityKey(stage.identity) === key && stage.execution === 'gpu'
              )
            : binding.gpuContributorNames.includes(identity.exportName));
        gpuAvailable ||= usesGpu;
        cpuAvailable ||= !usesGpu || binding.engine === 'gpu-or-cpu';
      }
      return {
        ...base,
        runnable: 'catalog-binding',
        engine: gpuAvailable ? (cpuAvailable ? 'gpu-or-cpu' : 'gpu') : 'cpu-reference'
      };
    }
    const location = locations.get(key);
    if (location) return {...base, runnable: 'native-panel', location};
    if (lensScenarioIds) return {...base, runnable: 'lens-scenario', engine: 'lens-scenario'};
    return {
      ...base,
      runnable: 'no',
      reason: scenarios.length ? NOT_BOUND_REASON : NOT_BOUND_REASON + LENS_NOT_LOADED_REASON
    };
  });
}
