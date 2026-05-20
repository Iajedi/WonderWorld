/** Dev-only debug saves + production submit payload builder. */

import { applyMaskMatteDestinationIn, buildOpaqueBWMaskFgCanvas } from './maskNormalize.js';

const SAVE_URL = '/__debug/save-image';
const SIZE = 512;

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

export async function saveDebugPng(filename, dataUrl) {
  const res = await fetch(SAVE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, dataUrl }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json();
}

export function canvasToPngDataUrl(canvas) {
  return canvas.toDataURL('image/png');
}

export function loadImageUrl(url) {
  return new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = () => reject(new Error('failed to load image'));
    i.src = url;
  });
}

/** Solid black 512x512 PNG data URL — used as fallback mask when none is set. */
export function blackMask512DataUrl() {
  const c = document.createElement('canvas');
  c.width = SIZE;
  c.height = SIZE;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, SIZE, SIZE);
  return canvasToPngDataUrl(c);
}

export function emptyTransparent512DataUrl() {
  const c = document.createElement('canvas');
  c.width = SIZE;
  c.height = SIZE;
  return canvasToPngDataUrl(c);
}

export function safeCanvasPngDataUrl(canvas) {
  try {
    return canvas.toDataURL('image/png');
  } catch {
    return emptyTransparent512DataUrl();
  }
}

// ---------------------------------------------------------------------------
// Transform helpers
// ---------------------------------------------------------------------------

/**
 * Full DOM-equivalent transform for manipulation/copy foreground (center origin, absolute coords).
 */
export function fullStackMatrixFromForeground(foregroundEl, fgW, fgH) {
  const ox = foregroundEl?.offsetLeft ?? 0;
  const oy = foregroundEl?.offsetTop ?? 0;
  const oxo = fgW / 2;
  const oyo = fgH / 2;
  const tf = foregroundEl ? getComputedStyle(foregroundEl).transform : 'none';
  const M = tf && tf !== 'none' ? new DOMMatrix(tf) : new DOMMatrix();
  const T_o = new DOMMatrix().translate(oxo, oyo);
  const T_no = new DOMMatrix().translate(-oxo, -oyo);
  const L = new DOMMatrix().translate(ox, oy);
  return L.multiply(T_o).multiply(M).multiply(T_no);
}

/**
 * Target image export (addition/replacement): offsetLeft/offsetTop + computed transform,
 * with fg centroid centered on the 512 PNG.
 */
export function setTargetForegroundExportTransform(ctx, foregroundEl, fgW, fgH) {
  const ox = foregroundEl?.offsetLeft ?? 0;
  const oy = foregroundEl?.offsetTop ?? 0;
  const tf = foregroundEl ? getComputedStyle(foregroundEl).transform : 'none';
  const local = tf && tf !== 'none' ? new DOMMatrix(tf) : new DOMMatrix();
  const full = new DOMMatrix().translate(ox, oy).multiply(local);

  const cx = fgW / 2;
  const cy = fgH / 2;
  const lcx = full.a * cx + full.c * cy;
  const lcy = full.b * cx + full.d * cy;
  const tx = full.e + SIZE / 2 - lcx;
  const ty = full.f + SIZE / 2 - lcy;
  ctx.setTransform(full.a, full.b, full.c, full.d, tx, ty);
}

/**
 * Source (manipulation/copy) export: absolute stack placement matching the DOM.
 */
export function setSourceForegroundExportTransform(ctx, foregroundEl, fgW, fgH) {
  const F = fullStackMatrixFromForeground(foregroundEl, fgW, fgH);
  ctx.setTransform(F.a, F.b, F.c, F.d, F.e, F.f);
}

// ---------------------------------------------------------------------------
// Image-rendering helpers (all return Canvas elements)
// ---------------------------------------------------------------------------

export async function urlToContained512DataUrl(url, fillStyle = '#ffffff') {
  if (!url) return null;
  const img = await loadImageUrl(url);
  const c = document.createElement('canvas');
  c.width = SIZE;
  c.height = SIZE;
  const ctx = c.getContext('2d');
  ctx.fillStyle = fillStyle;
  ctx.fillRect(0, 0, SIZE, SIZE);
  const s = Math.min(SIZE / img.naturalWidth, SIZE / img.naturalHeight);
  const w = Math.round(img.naturalWidth * s);
  const h = Math.round(img.naturalHeight * s);
  const x = Math.round((SIZE - w) / 2);
  const y = Math.round((SIZE - h) / 2);
  ctx.drawImage(img, x, y, w, h);
  return canvasToPngDataUrl(c);
}

/** Main overlay (colored) → strict B&W PNG, white = painted region, black = background. */
export function mainMaskOverlayToBwOpaquePngDataUrl(overlayCanvas) {
  const c = document.createElement('canvas');
  c.width = SIZE;
  c.height = SIZE;
  const ctx = c.getContext('2d');
  if (!ctx) return blackMask512DataUrl();
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, SIZE, SIZE);
  try {
    ctx.drawImage(overlayCanvas, 0, 0);
  } catch {
    return canvasToPngDataUrl(c);
  }
  const d = ctx.getImageData(0, 0, SIZE, SIZE);
  for (let i = 0; i < d.data.length; i += 4) {
    const a = d.data[i + 3];
    const r = d.data[i];
    const g = d.data[i + 1];
    const b = d.data[i + 2];
    const sel = a > 8 && (r + g + b) / 3 > 32;
    const v = sel ? 255 : 0;
    d.data[i] = v;
    d.data[i + 1] = v;
    d.data[i + 2] = v;
    d.data[i + 3] = 255;
  }
  ctx.putImageData(d, 0, 0);
  return canvasToPngDataUrl(c);
}

/** Foreground (target) with mask matte + transform, on 512×512 transparent background. */
export async function renderForegroundOverlay512(foregroundEl, fgW, fgH, targetImgUrl, maskUrl) {
  const out = document.createElement('canvas');
  out.width = SIZE;
  out.height = SIZE;
  const o = out.getContext('2d');
  if (!o || !targetImgUrl || !foregroundEl) return out;

  const img = await loadImageUrl(targetImgUrl);

  const masked = document.createElement('canvas');
  masked.width = fgW;
  masked.height = fgH;
  const mc = masked.getContext('2d');
  mc.drawImage(img, 0, 0, fgW, fgH);
  if (maskUrl) {
    await applyMaskMatteDestinationIn(mc, fgW, fgH, maskUrl);
  }

  setTargetForegroundExportTransform(o, foregroundEl, fgW, fgH);
  o.drawImage(masked, 0, 0);
  o.setTransform(1, 0, 0, 1, 0, 0);

  return out;
}

/** Manipulation/Copy: masked source crop at absolute stack coords. */
export function renderSourceMaskedForegroundOverlay512(
  foregroundEl,
  fgW,
  fgH,
  mainCanvas,
  overlay512,
  bounds
) {
  const out = document.createElement('canvas');
  out.width = SIZE;
  out.height = SIZE;
  const o = out.getContext('2d');
  if (!o || !foregroundEl || !mainCanvas || !overlay512 || !bounds) return out;
  const { x, y, w, h } = bounds;
  if (w <= 0 || h <= 0) return out;

  const masked = document.createElement('canvas');
  masked.width = fgW;
  masked.height = fgH;
  const mc = masked.getContext('2d');
  mc.drawImage(mainCanvas, x, y, w, h, 0, 0, fgW, fgH);
  mc.globalCompositeOperation = 'destination-in';
  mc.drawImage(overlay512, x, y, w, h, 0, 0, fgW, fgH);
  mc.globalCompositeOperation = 'source-over';

  setSourceForegroundExportTransform(o, foregroundEl, fgW, fgH);
  o.drawImage(masked, 0, 0);
  o.setTransform(1, 0, 0, 1, 0, 0);

  return out;
}

/** Opaque B&W source mask (white = selection) at absolute stack coords. */
export function renderSourceMaskTransformedBw512(foregroundEl, fgW, fgH, overlay512, bounds) {
  const out = document.createElement('canvas');
  out.width = SIZE;
  out.height = SIZE;
  const o = out.getContext('2d');
  if (!o || !foregroundEl || !overlay512 || !bounds) {
    if (o) { o.fillStyle = '#000'; o.fillRect(0, 0, SIZE, SIZE); }
    return out;
  }
  const { x, y, w, h } = bounds;
  if (w <= 0 || h <= 0) {
    o.fillStyle = '#000';
    o.fillRect(0, 0, SIZE, SIZE);
    return out;
  }

  const bw = document.createElement('canvas');
  bw.width = fgW;
  bw.height = fgH;
  const bctx = bw.getContext('2d');
  bctx.drawImage(overlay512, x, y, w, h, 0, 0, fgW, fgH);
  const d = bctx.getImageData(0, 0, fgW, fgH);
  for (let i = 0; i < d.data.length; i += 4) {
    const sel = d.data[i + 3] > 8;
    const v = sel ? 255 : 0;
    d.data[i] = v; d.data[i + 1] = v; d.data[i + 2] = v; d.data[i + 3] = 255;
  }
  bctx.putImageData(d, 0, 0);

  o.fillStyle = '#000';
  o.fillRect(0, 0, SIZE, SIZE);
  setSourceForegroundExportTransform(o, foregroundEl, fgW, fgH);
  o.drawImage(bw, 0, 0);
  o.setTransform(1, 0, 0, 1, 0, 0);

  return out;
}

/** Target mask only: opaque B&W at same transform as target foreground. */
export async function renderTransformedTargetMaskBw512(foregroundEl, fgW, fgH, maskUrl) {
  const out = document.createElement('canvas');
  out.width = SIZE;
  out.height = SIZE;
  const o = out.getContext('2d');
  if (!o) return out;
  o.fillStyle = '#000';
  o.fillRect(0, 0, SIZE, SIZE);
  if (!maskUrl || !foregroundEl) return out;

  const mcv = await buildOpaqueBWMaskFgCanvas(maskUrl, fgW, fgH);
  setTargetForegroundExportTransform(o, foregroundEl, fgW, fgH);
  o.drawImage(mcv, 0, 0);
  o.setTransform(1, 0, 0, 1, 0, 0);

  return out;
}

// ---------------------------------------------------------------------------
// Canonical image assembly — returns {source_image, source_mask, target_image, target_mask}
// all as PNG data URL strings, suitable for direct emit over Socket.IO.
// ---------------------------------------------------------------------------

/**
 * Builds the four 512x512 submit images for any edit mode.
 *
 * @param {object} params
 * @param {HTMLCanvasElement|null}  params.mainCanvasEl
 * @param {HTMLCanvasElement|null}  params.mainMaskOverlayEl
 * @param {HTMLElement|null}        params.foregroundEl
 * @param {number}                  params.fgW
 * @param {number}                  params.fgH
 * @param {string|null}             params.targetImageObjectUrl   addition/replacement
 * @param {string|null}             params.targetMaskObjectUrl    addition/replacement (optional)
 * @param {{x,y,w,h}|null}         params.sourceMaskBounds       manipulation/copy
 * @param {string}                  params.editType               editOperationMode
 * @returns {Promise<{source_image: string, source_mask: string, target_image: string, target_mask: string}>}
 */
export async function buildSubmitImages({
  mainCanvasEl,
  mainMaskOverlayEl,
  foregroundEl,
  fgW,
  fgH,
  targetImageObjectUrl,
  targetMaskObjectUrl,
  sourceMaskBounds,
  editType,
}) {
  const source_image = mainCanvasEl
    ? safeCanvasPngDataUrl(mainCanvasEl)
    : emptyTransparent512DataUrl();

  const source_mask = mainMaskOverlayEl
    ? mainMaskOverlayToBwOpaquePngDataUrl(mainMaskOverlayEl)
    : blackMask512DataUrl();

  const isSourceMode = editType === 'manipulation' || editType === 'copy';

  let target_image;
  let target_mask;

  if (isSourceMode) {
    const canExport =
      sourceMaskBounds &&
      sourceMaskBounds.w > 0 &&
      sourceMaskBounds.h > 0 &&
      mainCanvasEl &&
      mainMaskOverlayEl &&
      foregroundEl;

    if (canExport) {
      target_image = canvasToPngDataUrl(
        renderSourceMaskedForegroundOverlay512(
          foregroundEl, fgW, fgH, mainCanvasEl, mainMaskOverlayEl, sourceMaskBounds
        )
      );
      target_mask = canvasToPngDataUrl(
        renderSourceMaskTransformedBw512(
          foregroundEl, fgW, fgH, mainMaskOverlayEl, sourceMaskBounds
        )
      );
    } else {
      target_image = emptyTransparent512DataUrl();
      target_mask = blackMask512DataUrl();
    }
  } else {
    if (targetImageObjectUrl && foregroundEl) {
      target_image = canvasToPngDataUrl(
        await renderForegroundOverlay512(
          foregroundEl, fgW, fgH, targetImageObjectUrl, targetMaskObjectUrl || null
        )
      );
    } else {
      target_image = emptyTransparent512DataUrl();
    }

    if (targetMaskObjectUrl && foregroundEl) {
      target_mask = canvasToPngDataUrl(
        await renderTransformedTargetMaskBw512(foregroundEl, fgW, fgH, targetMaskObjectUrl)
      );
    } else {
      target_mask = blackMask512DataUrl();
    }
  }

  return { source_image, source_mask, target_image, target_mask };
}

// ---------------------------------------------------------------------------
// Debug file dump (thin wrapper around buildSubmitImages + file saves)
// ---------------------------------------------------------------------------

export async function saveSubmitDebugBundle({
  prefix,
  mainCanvasEl,
  mainMaskOverlayEl,
  foregroundEl,
  fgW,
  fgH,
  targetImageObjectUrl,
  targetMaskObjectUrl,
  sourceMaskBounds,
  editType,
}) {
  const writes = [];

  writes.push(saveDebugPng(`${prefix}-source.png`,
    mainCanvasEl ? safeCanvasPngDataUrl(mainCanvasEl) : emptyTransparent512DataUrl()));

  writes.push(saveDebugPng(`${prefix}-main-mask-bw.png`,
    mainMaskOverlayEl
      ? mainMaskOverlayToBwOpaquePngDataUrl(mainMaskOverlayEl)
      : blackMask512DataUrl()));

  const isSourceMode = editType === 'manipulation' || editType === 'copy';

  const canSourceExport =
    sourceMaskBounds &&
    sourceMaskBounds.w > 0 &&
    sourceMaskBounds.h > 0 &&
    mainCanvasEl &&
    mainMaskOverlayEl &&
    foregroundEl;

  if (isSourceMode && canSourceExport) {
    const fgT = renderSourceMaskedForegroundOverlay512(
      foregroundEl, fgW, fgH, mainCanvasEl, mainMaskOverlayEl, sourceMaskBounds
    );
    writes.push(saveDebugPng(`${prefix}-source-foreground-transformed.png`, canvasToPngDataUrl(fgT)));

    const maskT = renderSourceMaskTransformedBw512(
      foregroundEl, fgW, fgH, mainMaskOverlayEl, sourceMaskBounds
    );
    writes.push(saveDebugPng(`${prefix}-source-mask-transformed-bw.png`, canvasToPngDataUrl(maskT)));
  }

  if (!isSourceMode && targetImageObjectUrl) {
    const target512 = await urlToContained512DataUrl(targetImageObjectUrl, '#ffffff');
    if (target512) writes.push(saveDebugPng(`${prefix}-target-original.png`, target512));

    const transformed = await renderForegroundOverlay512(
      foregroundEl, fgW, fgH, targetImageObjectUrl, targetMaskObjectUrl || null
    );
    writes.push(saveDebugPng(`${prefix}-target-transformed.png`, canvasToPngDataUrl(transformed)));

    if (targetMaskObjectUrl) {
      const maskT = await renderTransformedTargetMaskBw512(
        foregroundEl, fgW, fgH, targetMaskObjectUrl
      );
      writes.push(saveDebugPng(`${prefix}-target-mask-transformed-bw.png`, canvasToPngDataUrl(maskT)));
    }
  }

  await Promise.all(writes);
}
