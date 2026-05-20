import { useEffect, useRef, useCallback } from 'react';
import { io } from 'socket.io-client';
import {
  defaultViewMatrix,
  invert4,
  rotate4,
  translate4,
  extractPositionFromViewMatrix,
  extractRotationFromViewMatrix,
} from './math.js';

const initialCameras = () => [
  {
    id: 0,
    position: [0, 0, 0],
    rotation: [
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, 1],
    ],
    fy: 1000,
    fx: 1000,
    yaw: 0,
    pitch: 0,
    movement: [0, 0, 0],
  },
];

function useExtrinsics(camera, yawRef, pitchRef, movementRef) {
  yawRef.current = camera.yaw;
  pitchRef.current = camera.pitch;
  movementRef.current[0] = camera.movement[0];
  movementRef.current[1] = camera.movement[1];
  movementRef.current[2] = camera.movement[2];
}

function useCamera(camera, yawRef, pitchRef, movementRef) {
  useExtrinsics(camera, yawRef, pitchRef, movementRef);
}

function storeCameraPose(camerasRef, matrix, yaw, pitch, movement) {
  const newPosition = extractPositionFromViewMatrix(matrix);
  const newRotation = extractRotationFromViewMatrix(matrix);
  const cameras = camerasRef.current;
  const cameraTmp = {
    id: cameras.length,
    position: newPosition,
    rotation: newRotation,
    fy: 1000,
    fx: 1000,
    yaw,
    pitch,
    movement: [...movement],
  };
  cameras.push(cameraTmp);
  if (cameras.length > 10) {
    cameras.splice(1, 1);
  }
}

/**
 * Socket URL must match the vanilla app: io.connect('http://localhost:7777/')
 * Override with import.meta.env.VITE_SOCKET_URL
 */
export function useStreamViewer({
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
}) {
  const socketRef = useRef(null);
  const yawRef = useRef(0);
  const pitchRef = useRef(0);
  const movementRef = useRef([0, 0, 0]);
  const viewMatrixRef = useRef([...defaultViewMatrix]);
  const camerasRef = useRef(initialCameras());
  const currentCameraIndexRef = useRef(0);
  const activeCameraRef = useRef(JSON.parse(JSON.stringify(camerasRef.current[0])));
  const activeKeysRef = useRef([]);
  const lastFrameRef = useRef(0);
  const avgFpsRef = useRef(0);
  const rafRef = useRef(0);
  const poseIntervalRef = useRef(0);

  const updateDisplayedInfo = useCallback(
    (camera) => {
      setCamMeta({
        camIndex: currentCameraIndexRef.current,
        fx: camera.fx,
        fy: camera.fy,
        innerW: window.innerWidth,
        innerH: window.innerHeight,
      });
    },
    [setCamMeta]
  );

  useEffect(() => {
    if (typeof window !== 'undefined' && window.location.host.includes('hf.space')) {
      document.body.classList.add('nohf');
    }
  }, []);

  useEffect(() => {
    const mainCanvas = mainCanvasRef.current;
    const vizCanvas = vizCanvasRef.current;
    if (!mainCanvas || !vizCanvas) return;

    const ctx = mainCanvas.getContext('2d');
    const ctxViz = vizCanvas.getContext('2d');

    const socket = io(socketUrl);
    socketRef.current = socket;

    socket.on('connect', () => {
      setServerConnect('Connected to server. Server initializing...');
    });

    socket.on('connect_error', () => {
      setServerConnect('Connection to server failed. Please retry.');
    });

    socket.on('frame', (data) => {
      const blob = new Blob([data], { type: 'image/jpeg' });
      const imageURL = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, mainCanvas.width, mainCanvas.height);
        URL.revokeObjectURL(imageURL);
      };
      img.src = imageURL;
    });

    socket.on('viz', (data) => {
      const blob = new Blob([data], { type: 'image/jpeg' });
      const imageURL = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        ctxViz.drawImage(img, 0, 0, vizCanvas.width, vizCanvas.height);
        URL.revokeObjectURL(imageURL);
      };
      img.src = imageURL;
    });

    socket.on('server-state', (msg) => {
      setServerConnect(msg);
    });

    socket.on('iter-number', (msg) => {
      setIterNumber(msg);
    });

    socket.on('scene-prompt', (msg) => {
      setPrompt(msg);
    });

    updateDisplayedInfo(activeCameraRef.current);

    const sendCameraPose = () => {
      const s = socketRef.current;
      if (s?.connected) {
        s.emit('render-pose', viewMatrixRef.current);
      }
    };

    poseIntervalRef.current = window.setInterval(sendCameraPose, 1000 / 60);

    const onKeyDown = (e) => {
      if (document.activeElement !== keyboardRootRef.current) return;
      if (editBlockCameraKeysRef?.current) return;
      const socketNow = socketRef.current;
      if (!socketNow) return;

      if (e.code === 'KeyI') {
        socketNow.emit('start', 'start signal');
      }
      if (e.code === 'KeyR') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        inv = translate4(inv, ...movementRef.current);
        inv = rotate4(inv, yawRef.current, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        const yawTmp = yawRef.current;
        const pitchTmp = pitchRef.current;
        const movementTmp = [...movementRef.current];
        storeCameraPose(camerasRef, viewMatrixRef.current, yawTmp, pitchTmp, movementTmp);
        socketNow.emit('gen', viewMatrixRef.current);
      }
      if (e.code === 'KeyQ') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        inv = translate4(inv, ...movementRef.current);
        inv = rotate4(inv, yawRef.current, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        console.log(`viewMatrix: [${viewMatrixRef.current}]`);
      }
      if (e.code === 'KeyT') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        const backwardMovement = [0, 0, -0.8];
        const combinedMovement = [
          movementRef.current[0] + backwardMovement[0],
          movementRef.current[1] + backwardMovement[1],
          movementRef.current[2] + backwardMovement[2],
        ];
        inv = translate4(inv, ...combinedMovement);
        inv = rotate4(inv, yawRef.current, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        const yawTmp = yawRef.current;
        const pitchTmp = pitchRef.current;
        storeCameraPose(camerasRef, viewMatrixRef.current, yawTmp, pitchTmp, combinedMovement);
        socketNow.emit('gen', viewMatrixRef.current);
      }
      if (e.code === 'KeyY') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        const leftTurnAngle = (20 * Math.PI) / 180;
        const yaw = yawRef.current;
        inv = translate4(inv, ...movementRef.current);
        inv = rotate4(inv, yaw - leftTurnAngle, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        const yawTmp = yaw - leftTurnAngle;
        const pitchTmp = pitchRef.current;
        const movementTmp = [...movementRef.current];
        storeCameraPose(camerasRef, viewMatrixRef.current, yawTmp, pitchTmp, movementTmp);
        socketNow.emit('gen', viewMatrixRef.current);
      }
      if (e.code === 'KeyU') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        const rightTurnAngle = (20 * Math.PI) / 180;
        const yaw = yawRef.current;
        inv = translate4(inv, ...movementRef.current);
        inv = rotate4(inv, yaw + rightTurnAngle, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        const yawTmp = yaw + rightTurnAngle;
        const pitchTmp = pitchRef.current;
        const movementTmp = [...movementRef.current];
        storeCameraPose(camerasRef, viewMatrixRef.current, yawTmp, pitchTmp, movementTmp);
        socketNow.emit('gen', viewMatrixRef.current);
      }
      if (e.code === 'KeyI') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        const leftTurnAngle = (15 * Math.PI) / 180;
        const backwardMovement = [0, 0, -0.5];
        const combinedMovement = [
          movementRef.current[0] + backwardMovement[0],
          movementRef.current[1] + backwardMovement[1],
          movementRef.current[2] + backwardMovement[2],
        ];
        const yaw = yawRef.current;
        inv = translate4(inv, ...combinedMovement);
        inv = rotate4(inv, yaw - leftTurnAngle, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        const yawTmp = yaw - leftTurnAngle;
        const pitchTmp = pitchRef.current;
        storeCameraPose(camerasRef, viewMatrixRef.current, yawTmp, pitchTmp, combinedMovement);
        socketNow.emit('gen', viewMatrixRef.current);
      }
      if (e.code === 'KeyO') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        const rightTurnAngle = (15 * Math.PI) / 180;
        const backwardMovement = [0, 0, -0.5];
        const combinedMovement = [
          movementRef.current[0] + backwardMovement[0],
          movementRef.current[1] + backwardMovement[1],
          movementRef.current[2] + backwardMovement[2],
        ];
        const yaw = yawRef.current;
        inv = translate4(inv, ...combinedMovement);
        inv = rotate4(inv, yaw + rightTurnAngle, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        const yawTmp = yaw + rightTurnAngle;
        const pitchTmp = pitchRef.current;
        storeCameraPose(camerasRef, viewMatrixRef.current, yawTmp, pitchTmp, combinedMovement);
        socketNow.emit('gen', viewMatrixRef.current);
      }
      if (e.code === 'KeyK') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        const leftTurnAngle = (15 * Math.PI) / 180;
        const backwardMovement = [0, 0, 0.5];
        const combinedMovement = [
          movementRef.current[0] + backwardMovement[0],
          movementRef.current[1] + backwardMovement[1],
          movementRef.current[2] + backwardMovement[2],
        ];
        const yaw = yawRef.current;
        inv = translate4(inv, ...combinedMovement);
        inv = rotate4(inv, yaw - leftTurnAngle, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        const yawTmp = yaw - leftTurnAngle;
        const pitchTmp = pitchRef.current;
        storeCameraPose(camerasRef, viewMatrixRef.current, yawTmp, pitchTmp, combinedMovement);
        socketNow.emit('gen', viewMatrixRef.current);
      }
      if (e.code === 'KeyL') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        const rightTurnAngle = (15 * Math.PI) / 180;
        const backwardMovement = [0, 0, 0.5];
        const combinedMovement = [
          movementRef.current[0] + backwardMovement[0],
          movementRef.current[1] + backwardMovement[1],
          movementRef.current[2] + backwardMovement[2],
        ];
        const yaw = yawRef.current;
        inv = translate4(inv, ...combinedMovement);
        inv = rotate4(inv, yaw + rightTurnAngle, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        const yawTmp = yaw + rightTurnAngle;
        const pitchTmp = pitchRef.current;
        storeCameraPose(camerasRef, viewMatrixRef.current, yawTmp, pitchTmp, combinedMovement);
        socketNow.emit('gen', viewMatrixRef.current);
      }
      if (e.code === 'KeyF') {
        activeCameraRef.current.fx += 10;
        activeCameraRef.current.fy += 10;
        updateDisplayedInfo(activeCameraRef.current);
      }
      if (e.code === 'KeyG') {
        activeCameraRef.current.fx -= 10;
        activeCameraRef.current.fy -= 10;
        updateDisplayedInfo(activeCameraRef.current);
      }
      if (e.code === 'KeyZ') {
        socketNow.emit('undo');
      }
      if (e.code === 'KeyX') {
        socketNow.emit('save');
      }
      if (e.code === 'KeyE') {
        socketNow.emit('fill_hole');
      }
      if (e.code === 'KeyC') {
        let inv = invert4(defaultViewMatrix);
        pitchRef.current = 0;
        inv = translate4(inv, ...movementRef.current);
        inv = rotate4(inv, yawRef.current, 0, 1, 0);
        inv = rotate4(inv, pitchRef.current, 1, 0, 0);
        viewMatrixRef.current = invert4(inv);
        socketNow.emit('delete', viewMatrixRef.current);
      }

      const ak = activeKeysRef.current;
      if (!ak.includes(e.code)) ak.push(e.code);
      if (/\d/.test(e.key)) {
        const idx = parseInt(e.key, 10);
        currentCameraIndexRef.current = idx;
        activeCameraRef.current = JSON.parse(JSON.stringify(camerasRef.current[idx]));
        useCamera(activeCameraRef.current, yawRef, pitchRef, movementRef);
        updateDisplayedInfo(activeCameraRef.current);
      }
    };

    const onKeyUp = (e) => {
      if (document.activeElement !== keyboardRootRef.current) return;
      if (editBlockCameraKeysRef?.current) return;
      activeKeysRef.current = activeKeysRef.current.filter((k) => k !== e.code);
    };

    const onBlur = () => {
      activeKeysRef.current = [];
    };

    const frame = (now) => {
      if (editBlockCameraKeysRef?.current) {
        activeKeysRef.current = [];
        const last = lastFrameRef.current;
        const currentFps = last ? 1000 / (now - last) : 0;
        avgFpsRef.current = avgFpsRef.current * 0.9 + currentFps * 0.1;
        setFpsText(`${Math.round(avgFpsRef.current)} fps`);
        lastFrameRef.current = now;
        rafRef.current = requestAnimationFrame(frame);
        return;
      }

      let inv = invert4(defaultViewMatrix);
      const speedFactor = 0.2;
      if (activeKeysRef.current.includes('KeyA')) yawRef.current -= 0.02 * speedFactor;
      if (activeKeysRef.current.includes('KeyD')) yawRef.current += 0.02 * speedFactor;
      if (activeKeysRef.current.includes('KeyW')) pitchRef.current += 0.005 * speedFactor;
      if (activeKeysRef.current.includes('KeyS')) pitchRef.current -= 0.005 * speedFactor;

      pitchRef.current = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitchRef.current));

      let dx = 0;
      let dz = 0;
      let dy = 0;
      if (activeKeysRef.current.includes('ArrowUp')) dz += 0.02 * speedFactor;
      if (activeKeysRef.current.includes('ArrowDown')) dz -= 0.02 * speedFactor;
      if (activeKeysRef.current.includes('ArrowLeft')) dx -= 0.02 * speedFactor;
      if (activeKeysRef.current.includes('ArrowRight')) dx += 0.02 * speedFactor;
      if (activeKeysRef.current.includes('KeyN')) dy -= 0.02 * speedFactor;
      if (activeKeysRef.current.includes('KeyM')) dy += 0.02 * speedFactor;

      const yaw = yawRef.current;
      const forward = [Math.sin(yaw) * dz, 0, Math.cos(yaw) * dz];
      const right = [Math.sin(yaw + Math.PI / 2) * dx, 0, Math.cos(yaw + Math.PI / 2) * dx];

      movementRef.current[0] += forward[0] + right[0];
      movementRef.current[1] += forward[1] + right[1] + dy;
      movementRef.current[2] += forward[2] + right[2];

      inv = translate4(inv, ...movementRef.current);
      inv = rotate4(inv, yawRef.current, 0, 1, 0);
      inv = rotate4(inv, pitchRef.current, 1, 0, 0);
      viewMatrixRef.current = invert4(inv);

      const last = lastFrameRef.current;
      const currentFps = last ? 1000 / (now - last) : 0;
      avgFpsRef.current = avgFpsRef.current * 0.9 + currentFps * 0.1;
      setFpsText(`${Math.round(avgFpsRef.current)} fps`);
      lastFrameRef.current = now;
      rafRef.current = requestAnimationFrame(frame);
    };

    rafRef.current = requestAnimationFrame(frame);

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
      cancelAnimationFrame(rafRef.current);
      clearInterval(poseIntervalRef.current);
      socket.disconnect();
      socketRef.current = null;
    };
  }, [
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
    updateDisplayedInfo,
  ]);

  return { socketRef };
}

export function emitScenePrompt(socketRef, text) {
  socketRef.current?.emit('scene-prompt', text);
}
