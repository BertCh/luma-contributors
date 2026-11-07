# Hypothetical Chicago turn-sequence restriction

The Chicago route workbench compares the unexpanded graph with an expanded state graph from `prepareGPUNetworkRestrictionStates`. The synthetic rule forbids the consecutive directed edge identity sequence `66 → 38`; it does not remove either road edge. This mirrors the sequence shape used by map turn restrictions (`from`/`via`/`to`) while remaining explicit that the bundled `chicago-walk` extract has no turn-restriction relations. See the [OpenStreetMap restriction relation documentation](https://wiki.openstreetmap.org/wiki/Restriction_type).

A deterministic CPU reference was computed from the same 61-node, 128-edge extract, affine duration costs and 12:00 departure relative to 08:00. The baseline route arrives at 246.8983 minutes after 08:00; the hypothetical sequence ban arrives at 247.7714 minutes, a 0.8731-minute increase. The baseline edge sequence begins `[66, 38, 19, ...]`; the expanded-state detour begins `[66, 36, 70, 38, ...]`. The original directed links are retained, so this is a turn/sequence constraint rather than a corridor closure.

The browser scene uses GPU outputs for both arrival times and the expanded route. The CPU computation in `chicago-restriction-reference.mjs` is an independent numerical oracle only. The restriction is a teaching scenario, not a claim about an observed sign or current road rule.
