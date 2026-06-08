/** @param {HTMLCanvasElement | null} mainCanvas */
export function snapshotMainCanvasForMaskDialog(mainCanvas, size) {
  const white = createWhiteDataUrl(size, size);
  if (!mainCanvas || mainCanvas.width === 0 || mainCanvas.height === 0) return white;
  try {
    const off = document.createElement('canvas');
    off.width = size;
    off.height = size;
    const o = off.getContext('2d');
    if (!o) return white;
    o.fillStyle = '#cdcdcd';
    o.fillRect(0, 0, size, size);
    o.drawImage(mainCanvas, 0, 0, size, size);
    return off.toDataURL('image/png');
  } catch {
    return white;
  }
}

export function createWhiteDataUrl(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');
  if (!ctx) return '';
  ctx.fillStyle = '#cdcdcd';
  ctx.fillRect(0, 0, w, h);
  return c.toDataURL('image/png');
}

/**
 * Renders mask (PNG with alpha / white strokes) at full opacity on overlay canvas.
 * @param {HTMLCanvasElement} overlayCanvas Size must match `size`.
 * @param {Blob} maskBlob
 * @param {number} size
 * @param {{ r: number; g: number; b: number }} color Selection color, fully opaque.
 */
export function paintMaskOnOverlay(overlayCanvas, maskBlob, size, color = { r: 255, g: 60, b: 60 }) {
  const ctx = overlayCanvas.getContext('2d');
  if (!ctx) return Promise.resolve();

  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(maskBlob);
    const img = new Image();
    img.onload = () => {
      try {
        ctx.clearRect(0, 0, size, size);
        ctx.save();
        ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
        ctx.fillRect(0, 0, size, size);
        ctx.globalCompositeOperation = 'destination-in';
        ctx.drawImage(img, 0, 0, size, size);
        ctx.restore();
        resolve();
      } catch (e) {
        reject(e);
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('mask image load failed'));
    };
    img.src = url;
  });
}

export function clearMaskOverlay(overlayCanvas) {
  const ctx = overlayCanvas?.getContext('2d');
  if (!ctx || !overlayCanvas) return;
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

/** Axis-aligned bounds of pixels with alpha above threshold (e.g. mask overlay). */
export function getMaskAlphaBoundingBox(canvas, alphaThreshold = 8) {
  const ctx = canvas?.getContext('2d');
  if (!ctx || canvas.width <= 0 || canvas.height <= 0) return null;
  const { width, height } = canvas;
  const data = ctx.getImageData(0, 0, width, height).data;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = data[(y * width + x) * 4 + 3];
      if (a > alphaThreshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX) return null;
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}
