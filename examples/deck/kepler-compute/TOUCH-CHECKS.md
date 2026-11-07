# Touch and swipe checks

Automated coverage: `scripts/views-touch.mjs` (emulated phone, Chromium via CDP touch input or
WebKit via synthetic Pointer Events; see its header for `KEPLER_TOUCH_BROWSER`,
`KEPLER_TOUCH_HEADED`, `KEPLER_TOUCH_DEVICE`) and the node unit tests in
`test/examples/kepler-touch-gestures.node.spec.ts`. Emulation does not reproduce real browser
chrome, momentum scrolling or GPU behavior, so run the checklist below on physical devices before
declaring touch support verified.

## Manual device checklist

Devices: iOS Safari (record iOS version) and Android Chrome (record Chrome version). Use a build
served over HTTPS or the local network. Load a sample with resident points.

| # | Check | Steps | Expected |
|---|-------|-------|----------|
| 1 | One-finger pan | Drag one finger across the map in Single, Dual (each view) and Swipe (right side). | The map under the finger follows it; no page scroll or rubber-banding. |
| 2 | Pinch zoom | Pinch in and out on each view. | Zoom changes smoothly; geography under the pinch center stays under the fingers. |
| 3 | Two-finger rotate | Twist two fingers. | Bearing changes in the twist direction; no browser zoom or page rotation. |
| 4 | Swipe divider drag | In Swipe, drag the divider handle with one finger across the screen, including to both edges. | Divider tracks the finger; the left side clips live; it clamps at 0 and 100 percent; the position persists after release and in a saved config. |
| 5 | Divider vs map | Touch just beside the divider, then drag. | Only the divider handle area (44 px wide) moves the divider; outside it the map pans. |
| 6 | Second finger during divider drag | Hold the divider with one finger and touch the map with another. | The divider keeps following the first finger; the map does not jump. |
| 7 | Interrupted drag | Drag the divider, then trigger a system gesture (edge swipe, notification pull, incoming alert). | The divider stays where it was last; it does not jump to an edge. |
| 8 | Scroll vs map | Scroll the surrounding page or side panel by dragging on it, then drag on the map. | Panel scrolls on panel drags; the map pans on map drags; neither steals the other's gesture. |
| 9 | Globe view | Switch to globe in Single, Dual and Swipe; pan, pinch, rotate. | The globe turns and zooms; Swipe shows one seamless globe; no blank or stretched view. |
| 10 | Orientation change | Rotate the device portrait to landscape and back in Single, Dual and Swipe. | Layout resizes; the divider keeps its percentage; the canvas is not blurry or stretched; gestures still work. |
| 11 | Pen (optional) | Use Apple Pencil or an Android stylus on the divider and map. | Behaves like touch. |

## Results

Copy this table per test run.

| Date | Tester | Device and OS | Browser version | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | Notes |
|------|--------|---------------|-----------------|---|---|---|---|---|---|---|---|---|----|----|-------|
|      |        | iOS Safari    |                 |   |   |   |   |   |   |   |   |   |    |    |       |
|      |        | Android Chrome|                 |   |   |   |   |   |   |   |   |   |    |    |       |

Mark each cell Pass, Fail or N/A, and describe failures in Notes with a screen recording.
