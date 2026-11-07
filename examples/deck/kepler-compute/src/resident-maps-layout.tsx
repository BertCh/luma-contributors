// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Children,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ComponentProps,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent
} from 'react';
import {MapViewStateContextProvider, MapsLayoutFactory} from '@kepler.gl/components';
import {PointerDragTracker, getSwipePercentage} from './touch-gestures';

export type ResidentMapsLayoutOptions = {
  /** Live divider position in percent while swipe compare is active, otherwise null. */
  onSwipePercentageChange(percentage: number | null): void;
};

const SWIPE_STYLE_ID = 'luma-swipe-style';
/** Hides the left basemap right of the divider; the shared Deck canvas clips its own views. */
const SWIPE_STYLE = `.luma-swipe-left .maplibregl-map {
  clip-path: polygon(0 0, 0 100%, var(--luma-swipe-percentage) 100%, var(--luma-swipe-percentage) 0);
}`;

const SWIPE_WRAPPER_STYLE: CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%'
};

type SwipeDividerProps = {
  referenceElement: HTMLElement;
  percentage: number;
  onMove(percentage: number): void;
  onRelease(percentage: number): void;
};

/**
 * Vertical swipe divider driven by Pointer Events for mouse, touch and pen. The handle sets
 * `touch-action: none` so a finger drag moves the divider instead of scrolling the page, captures
 * the pointer that started the drag, ignores other pointers, and commits the last position on
 * `pointercancel` instead of jumping.
 */
function SwipeDivider(props: SwipeDividerProps) {
  const {referenceElement, percentage, onMove, onRelease} = props;
  const trackerReference = useRef(new PointerDragTracker());
  const lastPercentageReference = useRef(percentage);
  const getPercentage = (clientX: number) => {
    const bounds = referenceElement.getBoundingClientRect();
    return getSwipePercentage(clientX, bounds.left, bounds.right);
  };
  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (trackerReference.current.handlePointerDown(event.pointerId, event.button) !== 'start') {
      return;
    }
    event.stopPropagation();
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (trackerReference.current.handlePointerMove(event.pointerId) !== 'move') return;
    event.stopPropagation();
    event.preventDefault();
    const nextPercentage = getPercentage(event.clientX);
    lastPercentageReference.current = nextPercentage;
    if (nextPercentage !== percentage) onMove(nextPercentage);
  };
  const handlePointerEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (trackerReference.current.handlePointerEnd(event.pointerId) !== 'end') return;
    event.stopPropagation();
    event.preventDefault();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const nextPercentage =
      event.type === 'pointercancel'
        ? lastPercentageReference.current
        : getPercentage(event.clientX);
    if (nextPercentage !== percentage) onMove(nextPercentage);
    onRelease(nextPercentage);
  };
  return (
    <div
      className="layout-splitter"
      style={{
        position: 'absolute',
        top: 0,
        left: `${percentage}%`,
        width: 44,
        marginLeft: -22,
        height: '100%',
        zIndex: 3,
        touchAction: 'none',
        cursor: 'ew-resize'
      }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerEnd}
      onPointerCancel={handlePointerEnd}
      onClick={event => event.stopPropagation()}
    >
      <div
        className="splitter"
        style={{
          position: 'absolute',
          top: 0,
          left: '50%',
          width: 2,
          height: '100%',
          transform: 'translateX(-50%)',
          background: '#FFFFFF',
          boxShadow: '0 0 2px rgba(0, 0, 0, 0.3)'
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          width: 36,
          height: 36,
          transform: 'translate(-50%, -50%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#29323C',
          color: '#FFFFFF',
          borderRadius: 4
        }}
      >
        <svg width="20" height="20" viewBox="-15 -15 30 30">
          <path d="M-15,0 L-5,-6 L-5,6Z" fill="currentColor" />
          <path d="M15,0 L5,-6 L5,6Z" fill="currentColor" />
        </svg>
      </div>
    </div>
  );
}

/**
 * Native `MapsLayout` clips the whole left map (and therefore the shared Deck canvas) at the
 * divider. This layout keeps every child mounted under the same wrappers in every mode, so
 * switching Single, Dual and Swipe never remounts (and finalizes) the shared Deck, and in Swipe
 * leaves the canvas unclipped: the Deck clips each view itself and only the left basemap is
 * clipped here.
 */
export function createResidentMapsLayoutFactory(options: ResidentMapsLayoutOptions) {
  function ResidentMapsLayoutFactory(...dependencies: Parameters<typeof MapsLayoutFactory>) {
    const NativeMapsLayout = MapsLayoutFactory(...dependencies);
    return function ResidentMapsLayout(props: ComponentProps<typeof NativeMapsLayout>) {
      const {mapState, className, children, onSetSwipeComparePercentage} = props;
      const containerReference = useRef<HTMLDivElement>(null);
      const [containerMounted, setContainerMounted] = useState(false);
      const childArray = Children.toArray(children);
      const isSwipe = childArray.length > 1 && mapState.mapSplitMode === 'SWIPE_COMPARE';
      const storedPercentage = mapState.swipeComparePercentage ?? 50;
      const [percentage, setPercentage] = useState(storedPercentage);
      const [isSwiping, setSwiping] = useState(false);
      const lastReleased = useRef<number | null>(null);
      useEffect(() => {
        setContainerMounted(true);
        if (!document.getElementById(SWIPE_STYLE_ID)) {
          const style = document.createElement('style');
          style.id = SWIPE_STYLE_ID;
          style.textContent = SWIPE_STYLE;
          document.head.append(style);
        }
      }, []);
      useEffect(() => {
        // Follow the store unless it has not yet caught up with the value just released.
        if (
          !isSwiping &&
          (lastReleased.current === null || storedPercentage === lastReleased.current)
        ) {
          setPercentage(storedPercentage);
          lastReleased.current = null;
        }
      }, [storedPercentage, isSwiping]);
      useEffect(() => {
        options.onSwipePercentageChange(isSwipe ? percentage : null);
      }, [isSwipe, percentage]);
      useEffect(() => () => options.onSwipePercentageChange(null), []);
      const handleMove = useCallback((value: number) => {
        setSwiping(true);
        setPercentage(value);
      }, []);
      const handleRelease = useCallback(
        (value: number) => {
          lastReleased.current = value;
          setPercentage(value);
          setSwiping(false);
          if (value !== storedPercentage) onSetSwipeComparePercentage?.({percentage: value});
        },
        [storedPercentage, onSetSwipeComparePercentage]
      );
      return (
        <div
          className={className}
          ref={containerReference}
          style={{position: 'relative', display: 'flex', width: '100%', height: '100%'}}
        >
          <MapViewStateContextProvider mapState={mapState}>
            {childArray.map((child, index) => (
              <div
                key={index}
                className={isSwipe && index === 0 ? 'luma-swipe-left' : undefined}
                style={
                  isSwipe
                    ? ({
                        ...SWIPE_WRAPPER_STYLE,
                        '--luma-swipe-percentage': `${percentage}%`
                      } as CSSProperties)
                    : {display: 'contents'}
                }
              >
                {child}
              </div>
            ))}
            {isSwipe && containerMounted && containerReference.current && (
              <>
                {isSwiping && (
                  <div
                    style={{
                      ...SWIPE_WRAPPER_STYLE,
                      background: 'transparent',
                      zIndex: 2
                    }}
                  />
                )}
                <SwipeDivider
                  referenceElement={containerReference.current}
                  percentage={percentage}
                  onMove={handleMove}
                  onRelease={handleRelease}
                />
              </>
            )}
          </MapViewStateContextProvider>
        </div>
      );
    };
  }
  ResidentMapsLayoutFactory.deps = MapsLayoutFactory.deps;
  return ResidentMapsLayoutFactory;
}
