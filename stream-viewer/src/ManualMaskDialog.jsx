import { useRef, useEffect, useState, useCallback } from 'react';

const BRUSH_RADIUS = 14;

export function ManualMaskDialog({ open, imageUrl, title = 'Manual mask', onClose, onApply }) {
  const canvasRef = useRef(null);
  const lastRef = useRef({ x: 0, y: 0 });
  const [tool, setTool] = useState('brush');
  const [drawing, setDrawing] = useState(false);
  const [display, setDisplay] = useState({ w: 320, h: 320 });

  const getLocal = useCallback((canvas, clientX, clientY) => {
    const r = canvas.getBoundingClientRect();
    return {
      x: ((clientX - r.left) / r.width) * canvas.width,
      y: ((clientY - r.top) / r.height) * canvas.height,
    };
  }, []);

  const applyToolStyle = useCallback(
    (ctx) => {
      ctx.lineWidth = BRUSH_RADIUS * 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      if (tool === 'eraser') {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.strokeStyle = 'rgba(0,0,0,1)';
      } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
      }
    },
    [tool]
  );

  useEffect(() => {
    if (!open || !imageUrl) return;
    setTool('brush');
    const img = new Image();
    img.onload = () => {
      const maxW = 480;
      const maxH = 360;
      const sx = Math.min(1, maxW / img.naturalWidth, maxH / img.naturalHeight);
      const w = Math.max(1, Math.round(img.naturalWidth * sx));
      const h = Math.max(1, Math.round(img.naturalHeight * sx));
      setDisplay({ w, h });
      requestAnimationFrame(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = w;
        canvas.height = h;
        const c = canvas.getContext('2d');
        c.clearRect(0, 0, w, h);
      });
    };
    img.onerror = () => {
      setDisplay({ w: 320, h: 320 });
    };
    img.src = imageUrl;
  }, [open, imageUrl]);

  const onPointerDown = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    setDrawing(true);
    const { x, y } = getLocal(canvas, e.clientX, e.clientY);
    lastRef.current = { x, y };
    const ctx = canvas.getContext('2d');
    applyToolStyle(ctx);
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
    ctx.fill();
  };

  const onPointerMove = (e) => {
    if (!drawing) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getLocal(canvas, e.clientX, e.clientY);
    const ctx = canvas.getContext('2d');
    applyToolStyle(ctx);
    ctx.beginPath();
    ctx.moveTo(lastRef.current.x, lastRef.current.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastRef.current = { x, y };
  };

  const endStroke = () => {
    setDrawing(false);
  };

  const handleApply = () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      onClose();
      return;
    }
    canvas.toBlob(
      (blob) => {
        if (blob && onApply) onApply(blob);
        onClose();
      },
      'image/png',
      1
    );
  };

  if (!open || !imageUrl) return null;

  return (
    <div
      className="manual-mask-backdrop"
      role="presentation"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className="manual-mask-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="manual-mask-title"
      >
        <header className="manual-mask-header">
          <h2 id="manual-mask-title" className="manual-mask-title">
            {title}
          </h2>
          <button type="button" className="manual-mask-close-icon" onClick={onClose} aria-label="Close dialog">
            ×
          </button>
        </header>
        <div className="manual-mask-tools">
          <button type="button" className={tool === 'brush' ? 'active' : ''} onClick={() => setTool('brush')}>
            Brush
          </button>
          <button type="button" className={tool === 'eraser' ? 'active' : ''} onClick={() => setTool('eraser')}>
            Eraser
          </button>
        </div>
        <div className="manual-mask-canvas-wrap" style={{ width: display.w, height: display.h }}>
          <img src={imageUrl} alt="" className="manual-mask-underlay" width={display.w} height={display.h} draggable={false} />
          <canvas
            ref={canvasRef}
            className="manual-mask-canvas"
            width={display.w}
            height={display.h}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endStroke}
            onPointerCancel={endStroke}
          />
        </div>
        <footer className="manual-mask-footer">
          <button type="button" className="edit-panel-btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="edit-panel-btn-primary" onClick={handleApply}>
            Apply mask
          </button>
        </footer>
      </div>
    </div>
  );
}
