// Conversion from plain JS to React JS done by Cursor Composer 2
import { useEffect, useRef, useState, useCallback } from 'react';
import { flushSync } from 'react-dom';
import { useStreamViewer, emitScenePrompt } from './useStreamViewer.js';
import { ManualMaskDialog } from './ManualMaskDialog.jsx';
import { snapshotMainCanvasForMaskDialog, paintMaskOnOverlay, clearMaskOverlay, getMaskAlphaBoundingBox } from './maskUtils.js';
import { buildSubmitImages, saveSubmitDebugBundle } from './debugExportClient.js';
import { normalizeMaskBlobToObjectUrl } from './maskNormalize.js';
import './App.css';

import Moveable from 'react-moveable';

const DEFAULT_SOCKET = 'http://localhost:7777/';
const socketUrl = import.meta.env.VITE_SOCKET_URL || DEFAULT_SOCKET;

const CANVAS_SIZE = 512;
/** Placeholder box (no target image), centered in the canvas. */
const FOREGROUND_TRANSFORM_INITIAL = `translate(${Math.round((CANVAS_SIZE - 140) / 2)}px, ${Math.round((CANVAS_SIZE - 88) / 2)}px) rotate(0deg) scale(1, 1)`;

/** Preserve aspect ratio; scale down so both sides are ≤ max; never upscale past natural size. */
function fitImageToMaxBox(nw, nh, max) {
  if (!nw || !nh) return { w: 1, h: 1 };
  const scale = Math.min(1, max / nw, max / nh);
  return {
    w: Math.max(1, Math.round(nw * scale)),
    h: Math.max(1, Math.round(nh * scale)),
  };
}

function centerForegroundTransform(w, h) {
  const tx = Math.round((CANVAS_SIZE - w) / 2);
  const ty = Math.round((CANVAS_SIZE - h) / 2);
  return `translate(${tx}px, ${ty}px) rotate(0deg) scale(1, 1)`;
}

/** Kept inside the 512x512 main stack (Moveable `bounds`). */
const MAIN_CANVAS_MOVEABLE_BOUNDS = {
  left: 0,
  top: 0,
  right: CANVAS_SIZE,
  bottom: CANVAS_SIZE,
};

const EDIT_MODES = [
  { value: 'manipulation', label: 'Manipulation' },
  { value: 'addition', label: 'Addition' },
  { value: 'copy', label: 'Copy' },
  { value: 'replacement', label: 'Replacement' },
];

const SHOW_DRAW_MASK = new Set(['manipulation', 'copy', 'replacement']);
const SHOW_TARGET_IMAGE = new Set(['addition', 'replacement']);
/** Main-canvas mask drives a tight moveable patch (no target image). */
const SOURCE_MASK_MODES = new Set(['manipulation', 'copy']);
const SOURCE_MASK_TRANSFORM_INITIAL = 'translate(0px, 0px) rotate(0deg) scale(1, 1)';

export default function App() {
  const keyboardRootRef = useRef(null);
  const mainCanvasRef = useRef(null);
  const vizCanvasRef = useRef(null);
  const foregroundRef = useRef(null);
  const moveableRef = useRef(null);
  const foregroundTransformRef = useRef(FOREGROUND_TRANSFORM_INITIAL);
  const mainMaskOverlayCanvasRef = useRef(null);
  const targetFileInputRef = useRef(null);
  const maskUploadInputRef = useRef(null);
  const maskDialogKindRef = useRef('target');

  const [moveContainer, setMoveContainer] = useState(null);
  const [moveableSyncToken, setMoveableSyncToken] = useState(0);
  const bumpMoveableSync = useCallback(() => {
    setMoveableSyncToken((n) => n + 1);
  }, []);

  const [prompt, setPrompt] = useState('');
  const [serverConnect, setServerConnect] = useState('');
  const [fpsText, setFpsText] = useState('');
  const [iterNumber, setIterNumber] = useState('');
  const [camMeta, setCamMeta] = useState({
    camIndex: 0,
    fx: 1000,
    fy: 1000,
    innerW: typeof window !== 'undefined' ? window.innerWidth : 0,
    innerH: typeof window !== 'undefined' ? window.innerHeight : 0,
  });
  const [edit, setEdit] = useState(false);

  const [editOperationMode, setEditOperationMode] = useState('manipulation');
  const [targetImageObjectUrl, setTargetImageObjectUrl] = useState(null);
  const [targetDisplaySize, setTargetDisplaySize] = useState(null);
  const [targetHighResWarning, setTargetHighResWarning] = useState(false);
  const [maskDialogOpen, setMaskDialogOpen] = useState(false);
  const [maskDialogImage, setMaskDialogImage] = useState('');
  const [maskDialogTitle, setMaskDialogTitle] = useState('Manual mask');
  const [uploadedMaskObjectUrl, setUploadedMaskObjectUrl] = useState(null);
  const [manualMaskObjectUrl, setManualMaskObjectUrl] = useState(null);
  const [mainCanvasMaskCommitted, setMainCanvasMaskCommitted] = useState(false);
  const [sourceMaskBounds, setSourceMaskBounds] = useState(null);
  const [submitBusy, setSubmitBusy] = useState(false);

  const editBlockCameraKeysRef = useRef(false);
  useEffect(() => {
    editBlockCameraKeysRef.current = edit;
  }, [edit]);

  const sourceMaskFgCanvasRef = useRef(null);

  const showDrawMask = SHOW_DRAW_MASK.has(editOperationMode);
  const showTargetImage = SHOW_TARGET_IMAGE.has(editOperationMode);
  const showMaskMethodRow = showTargetImage && Boolean(targetImageObjectUrl);

  const isSourceMaskForeground =
    SOURCE_MASK_MODES.has(editOperationMode) && mainCanvasMaskCommitted && sourceMaskBounds != null;

  const showForegroundMoveable =
    (showTargetImage && Boolean(targetImageObjectUrl)) || isSourceMaskForeground;

  const fgW = isSourceMaskForeground && sourceMaskBounds ? sourceMaskBounds.w : (targetDisplaySize?.w ?? 140);
  const fgH = isSourceMaskForeground && sourceMaskBounds ? sourceMaskBounds.h : (targetDisplaySize?.h ?? 88);

  const effectiveTargetMaskUrl = manualMaskObjectUrl || uploadedMaskObjectUrl;

  const { socketRef } = useStreamViewer({
    socketUrl,
    keyboardRootRef,
    mainCanvasRef,
    vizCanvasRef,
    editBlockCameraKeysRef,
    setPrompt,
    setServerConnect,
    setFpsText,
    setIterNumber,
    setCamMeta,
  });

  const closeMaskDialog = useCallback(() => {
    setMaskDialogOpen(false);
    setMaskDialogImage('');
    setMaskDialogTitle('Manual mask');
  }, []);

  useEffect(() => {
    if (edit) return;
    setManualMaskObjectUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setUploadedMaskObjectUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setTargetImageObjectUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    closeMaskDialog();
    setEditOperationMode('manipulation');
    setTargetHighResWarning(false);
    setTargetDisplaySize(null);
    setMainCanvasMaskCommitted(false);
    setSourceMaskBounds(null);
    const oc = mainMaskOverlayCanvasRef.current;
    if (oc) clearMaskOverlay(oc);
  }, [edit, closeMaskDialog]);

  useEffect(() => {
    const el = keyboardRootRef.current;
    if (!el) return;
    const t = requestAnimationFrame(() => el.focus());
    return () => cancelAnimationFrame(t);
  }, []);

  useEffect(() => {
    if (!edit) return;
    const id = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (foregroundRef.current) {
          foregroundRef.current.style.transform = foregroundTransformRef.current;
        }
        moveableRef.current?.updateRect();
      });
    });
    return () => cancelAnimationFrame(id);
  }, [edit, moveableSyncToken]);

  useEffect(() => {
    if (edit && showForegroundMoveable) bumpMoveableSync();
  }, [edit, showForegroundMoveable, bumpMoveableSync, sourceMaskBounds]);

  const applyForegroundTransform = useCallback((target, transform) => {
    target.style.transform = transform;
    // console.log(transform);
    foregroundTransformRef.current = transform;
  }, []);

  const applyCenteredForegroundTransform = useCallback(
    (w, h) => {
      const t = centerForegroundTransform(w, h);
      foregroundTransformRef.current = t;
      const fg = foregroundRef.current;
      if (fg) fg.style.transform = t;
      bumpMoveableSync();
    },
    [bumpMoveableSync]
  );

  const onEditModeSelect = useCallback((e) => {
    setEditOperationMode(e.target.value);
  }, []);

  const openMainCanvasMaskDialog = useCallback(() => {
    maskDialogKindRef.current = 'mainCanvas';
    setMaskDialogTitle('Draw mask');
    setMaskDialogImage(snapshotMainCanvasForMaskDialog(mainCanvasRef.current, CANVAS_SIZE));
    setMaskDialogOpen(true);
  }, []);

  const openTargetManualMaskDialog = useCallback(() => {
    if (!targetImageObjectUrl) return;
    maskDialogKindRef.current = 'target';
    setMaskDialogTitle('Manual mask');
    setMaskDialogImage(targetImageObjectUrl);
    setMaskDialogOpen(true);
  }, [targetImageObjectUrl]);

  const onMaskDialogApply = useCallback(
    async (blob) => {
      const kind = maskDialogKindRef.current;
      if (kind === 'mainCanvas') {
        const c = mainMaskOverlayCanvasRef.current;
        if (c) {
          try {
            await paintMaskOnOverlay(c, blob, CANVAS_SIZE);
            if (SOURCE_MASK_MODES.has(editOperationMode)) {
              const bbox = getMaskAlphaBoundingBox(c);
              if (bbox) {
                setMainCanvasMaskCommitted(true);
                setSourceMaskBounds(bbox);
                foregroundTransformRef.current = SOURCE_MASK_TRANSFORM_INITIAL;
                const fg = foregroundRef.current;
                if (fg) fg.style.transform = SOURCE_MASK_TRANSFORM_INITIAL;
              } else {
                clearMaskOverlay(c);
                setMainCanvasMaskCommitted(false);
                setSourceMaskBounds(null);
              }
            } else {
              setMainCanvasMaskCommitted(true);
              setSourceMaskBounds(null);
              const w = targetDisplaySize?.w ?? 140;
              const h = targetDisplaySize?.h ?? 88;
              applyCenteredForegroundTransform(w, h);
            }
          } catch {
            /* ignore paint errors */
          }
        }
      } else {
        try {
          const newUrl = await normalizeMaskBlobToObjectUrl(blob);
          setUploadedMaskObjectUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return null;
          });
          setManualMaskObjectUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return newUrl;
          });
        } catch {
          /* ignore mask normalize errors */
        }
      }
      bumpMoveableSync();
    },
    [applyCenteredForegroundTransform, bumpMoveableSync, editOperationMode, targetDisplaySize]
  );

  const onTargetFileChange = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      e.target.value = '';
      if (!file) return;
      setTargetImageObjectUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      setTargetDisplaySize(null);
      const oc = mainMaskOverlayCanvasRef.current;
      if (oc) clearMaskOverlay(oc);
      setMainCanvasMaskCommitted(false);
      setSourceMaskBounds(null);
      foregroundTransformRef.current = FOREGROUND_TRANSFORM_INITIAL;
      const fg = foregroundRef.current;
      if (fg) fg.style.transform = FOREGROUND_TRANSFORM_INITIAL;
      bumpMoveableSync();
    },
    [bumpMoveableSync]
  );

  const onTargetImageLoad = useCallback(
    (e) => {
      const img = e.currentTarget;
      const nw = img.naturalWidth;
      const nh = img.naturalHeight;
      const { w, h } = fitImageToMaxBox(nw, nh, CANVAS_SIZE);
      setTargetHighResWarning(nw > CANVAS_SIZE || nh > CANVAS_SIZE);
      setTargetDisplaySize({ w, h });
      const t = centerForegroundTransform(w, h);
      foregroundTransformRef.current = t;
      const fg = foregroundRef.current;
      if (fg) {
        fg.style.transform = t;
      }
      bumpMoveableSync();
    },
    [bumpMoveableSync]
  );

  const onMaskUploadChange = useCallback(
    async (e) => {
      const file = e.target.files?.[0];
      e.target.value = '';
      if (!file) return;
      try {
        const newUrl = await normalizeMaskBlobToObjectUrl(file);
        setManualMaskObjectUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return null;
        });
        setUploadedMaskObjectUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return newUrl;
        });
      } catch (err) {
        console.error(err);
      }
      bumpMoveableSync();
    },
    [bumpMoveableSync]
  );

  /** Manipulation / Copy: live masked source only (no red); red overlay stays on stack under this layer. */
  useEffect(() => {
    if (!edit || !isSourceMaskForeground || !sourceMaskBounds) return;
    const base = mainCanvasRef.current;
    const ov = mainMaskOverlayCanvasRef.current;
    const out = sourceMaskFgCanvasRef.current;
    if (!base || !ov || !out) return;
    const ctx = out.getContext('2d');
    if (!ctx) return;
    const { x, y, w, h } = sourceMaskBounds;
    let id = 0;
    const frame = () => {
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'source-over';
      ctx.drawImage(base, x, y, w, h, 0, 0, w, h);
      ctx.globalCompositeOperation = 'destination-in';
      ctx.drawImage(ov, x, y, w, h, 0, 0, w, h);
      id = requestAnimationFrame(frame);
    };
    id = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(id);
  }, [edit, isSourceMaskForeground, sourceMaskBounds]);

  const onRootMouseDown = useCallback((e) => {
    if (!e.target.closest('input, button, select, [class*="moveable"], .manual-mask-dialog')) {
      keyboardRootRef.current?.focus();
    }
  }, []);

  const onSendClick = useCallback(() => {
    emitScenePrompt(socketRef, prompt);
  }, [socketRef, prompt]);

  const handleSubmit = useCallback(async () => {
    const isSourceMode = editOperationMode === 'manipulation' || editOperationMode === 'copy';
    const isTargetMode = editOperationMode === 'addition' || editOperationMode === 'replacement';

    if (isSourceMode && (!sourceMaskBounds || !foregroundRef.current)) {
      alert('Please draw a mask on the source canvas before submitting.');
      return;
    }
    if (isTargetMode && (!targetImageObjectUrl || !foregroundRef.current)) {
      alert('Please choose a target image before submitting.');
      return;
    }

    setSubmitBusy(true);
    try {
      const commonArgs = {
        mainCanvasEl: mainCanvasRef.current,
        mainMaskOverlayEl: mainMaskOverlayCanvasRef.current,
        foregroundEl: foregroundRef.current,
        fgW,
        fgH,
        targetImageObjectUrl,
        targetMaskObjectUrl: effectiveTargetMaskUrl,
        sourceMaskBounds: isSourceMaskForeground ? sourceMaskBounds : null,
        editType: editOperationMode,
      };

      const images = await buildSubmitImages(commonArgs);

      const payload = {
        edit_type: editOperationMode,
        ...images,
      };

      socketRef.current?.emit('edit_submit', payload);
      console.log('Submitted edit payload:', editOperationMode);

      if (import.meta.env.VITE_DEBUG_EXPORT_SUBMIT === 'true') {
        const prefix = `debug-${Date.now()}`;
        await saveSubmitDebugBundle({ prefix, ...commonArgs });
        console.log('Debug PNGs written to stream-viewer/debug-export/', `${prefix}-*.png`);
      }
    } catch (err) {
      console.error(err);
      alert(`Submit failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSubmitBusy(false);
    }
  }, [
    editOperationMode,
    sourceMaskBounds,
    targetImageObjectUrl,
    fgW,
    fgH,
    effectiveTargetMaskUrl,
    isSourceMaskForeground,
    socketRef,
  ]);

  const foregroundMaskStyle = effectiveTargetMaskUrl
    ? {
        WebkitMaskImage: `url(${effectiveTargetMaskUrl})`,
        maskImage: `url(${effectiveTargetMaskUrl})`,
        WebkitMaskSize: '100% 100%',
        maskSize: '100% 100%',
        WebkitMaskRepeat: 'no-repeat',
        maskRepeat: 'no-repeat',
        WebkitMaskPosition: 'center',
        maskPosition: 'center',
      }
    : undefined;

  return (
    <div
      ref={keyboardRootRef}
      className="app-root"
      tabIndex={-1}
      role="application"
      aria-label="Gaussian splat viewer"
      onMouseDown={onRootMouseDown}
    >
      <input
        type="text"
        id="prompt-box"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <button type="button" id="send-button" onClick={onSendClick}>
        Next scene is ..
      </button>

      <div id="progress" />

      <div id="message" />

      <div id="main-canvas-container">
        <div className="main-canvas-toolbar">
          <button type="button" id="edit-view-button" disabled={edit} onClick={() => setEdit(true)}>
            Edit view
          </button>
        </div>

        <div className="main-canvas-stack" ref={setMoveContainer}>
          <canvas ref={mainCanvasRef} id="canvas" width={CANVAS_SIZE} height={CANVAS_SIZE} />

          {edit && showDrawMask && (
            <canvas
              ref={mainMaskOverlayCanvasRef}
              className="main-mask-overlay-committed"
              width={CANVAS_SIZE}
              height={CANVAS_SIZE}
              aria-hidden={!mainCanvasMaskCommitted}
            />
          )}

          {edit && showForegroundMoveable && (
            <>
              <div className="canvas-overlay" aria-hidden="true">
                <div
                  ref={foregroundRef}
                  className={`foreground-object${effectiveTargetMaskUrl ? ' foreground-object-masked' : ''}`}
                  style={{
                    width: fgW,
                    height: fgH,
                    ...(isSourceMaskForeground && sourceMaskBounds
                      ? { left: sourceMaskBounds.x, top: sourceMaskBounds.y }
                      : { left: 0, top: 0 }),
                    ...foregroundMaskStyle,
                  }}
                >
                  {isSourceMaskForeground ? (
                    <canvas
                      ref={sourceMaskFgCanvasRef}
                      width={fgW}
                      height={fgH}
                      style={{ width: fgW, height: fgH, display: 'block', pointerEvents: 'none' }}
                    />
                  ) : targetImageObjectUrl ? (
                    <img
                      src={targetImageObjectUrl}
                      alt=""
                      draggable={false}
                      onLoad={onTargetImageLoad}
                      style={{ width: fgW, height: fgH, objectFit: 'contain', display: 'block' }}
                    />
                  ) : null}
                </div>
              </div>

              {moveContainer ? (
                <Moveable
                  ref={moveableRef}
                  flushSync={flushSync}
                  className="foreground-moveable"
                  target={foregroundRef}
                  container={moveContainer}
                  origin
                  draggable
                  rotatable
                  scalable
                  pinchable
                  snappable
                  bounds={MAIN_CANVAS_MOVEABLE_BOUNDS}
                  keepRatio={false}
                  edge={false}
                  useResizeObserver
                  throttleDrag={0}
                  throttleRotate={0}
                  throttleScale={0}
                  onDrag={({ target, transform }) => applyForegroundTransform(target, transform)}
                  onRotate={({ target, transform }) => applyForegroundTransform(target, transform)}
                  onScale={({ target, transform }) => applyForegroundTransform(target, transform)}
                />
              ) : null}
            </>
          )}
        </div>
      </div>

      <div className="viz-edit-column">
        <canvas ref={vizCanvasRef} id="canvas-viz" width={768} height={256} />

        {edit && (
          <aside className="edit-panel" aria-label="Edit mode panel">
            <header className="edit-panel-header">
              <span className="edit-panel-title">Edit mode</span>
              <button type="button" className="edit-panel-close" onClick={() => setEdit(false)}>
                Close
              </button>
            </header>
            <div className="edit-panel-body">
              <label className="edit-panel-label" htmlFor="edit-operation-mode">
                Choose edit mode
              </label>
              <select
                id="edit-operation-mode"
                className="edit-panel-select"
                value={editOperationMode}
                onChange={onEditModeSelect}
              >
                {EDIT_MODES.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>

              {showDrawMask && (
                <div className="edit-panel-section">
                  <button type="button" className="edit-panel-btn-primary" onClick={openMainCanvasMaskDialog}>
                    Draw mask
                  </button>
                  <p className="edit-panel-hint">Paint over a snapshot of the main canvas (white if empty).</p>
                  {mainCanvasMaskCommitted && (
                    <p className="edit-panel-hint">Selection is shown on the canvas at full opacity.</p>
                  )}
                </div>
              )}

              {showTargetImage && (
                <div className="edit-panel-section">
                  <button
                    type="button"
                    className="edit-panel-btn-primary"
                    onClick={() => targetFileInputRef.current?.click()}
                  >
                    Choose target image file
                  </button>
                  <input
                    ref={targetFileInputRef}
                    type="file"
                    hidden
                    accept="image/jpeg,image/jpg,image/png,.jpg,.jpeg,.png"
                    onChange={onTargetFileChange}
                  />
                  {targetHighResWarning && (
                    <p className="edit-panel-warning">Warning: image exceeds 512x512 on at least one side.</p>
                  )}
                </div>
              )}

              {showMaskMethodRow && (
                <div className="edit-panel-section">
                  <div className="edit-panel-subheading">Mask for target</div>
                  <div className="edit-panel-mask-actions">
                    <button type="button" className="edit-panel-dummy-btn" disabled title="Coming soon">
                      Use SAM
                    </button>
                    <button type="button" className="edit-panel-btn-secondary" onClick={openTargetManualMaskDialog}>
                      Manual
                    </button>
                    <button type="button" className="edit-panel-dummy-btn" disabled title="Coming soon">
                      Use Alpha Channel
                    </button>
                    <button
                      type="button"
                      className="edit-panel-btn-secondary"
                      onClick={() => maskUploadInputRef.current?.click()}
                    >
                      Upload mask image
                    </button>
                    <input
                      ref={maskUploadInputRef}
                      type="file"
                      hidden
                      accept="image/png,image/jpeg,.jpg,.jpeg,.png"
                      onChange={onMaskUploadChange}
                    />
                  </div>
                  {effectiveTargetMaskUrl && (
                    <p className="edit-panel-hint">
                      Target mask active. Use black and white (white = selection). A new upload or manual mask replaces
                      the previous one.
                    </p>
                  )}
                </div>
              )}

              <div className="edit-panel-footer-actions">
                <button
                  type="button"
                  className="edit-panel-btn-submit edit-panel-btn-submit-active"
                  disabled={submitBusy}
                  onClick={handleSubmit}
                >
                  {submitBusy ? 'Saving...' : 'Submit'}
                </button>
              </div>
            </div>
          </aside>
        )}
      </div>

      <ManualMaskDialog
        open={maskDialogOpen && Boolean(maskDialogImage)}
        imageUrl={maskDialogImage}
        title={maskDialogTitle}
        onClose={closeMaskDialog}
        onApply={onMaskDialogApply}
      />

      <div id="quality">
        <span id="fps">{fpsText}</span>
      </div>
      <div id="server-state">
        <span id="server-connect">{serverConnect}</span>
      </div>
      <div id="caminfo">
        <span id="iter-number">{iterNumber}</span>
        <br />
        <span id="camid">cam {camMeta.camIndex}</span>
        <br />
        <span id="focal-x">focal_x {camMeta.fx}</span>
        <br />
        <span id="focal-y">focal_y {camMeta.fy}</span>
        <br />
        <span id="inner-width">inner_width {camMeta.innerW}</span>
        <br />
        <span id="inner-height">inner_height {camMeta.innerH}</span>
      </div>
    </div>
  );
}
