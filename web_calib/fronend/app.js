// ================================
//  SETTINGS
// ================================
const WS_URL = "ws://localhost:8000/ws";
const SEND_INTERVAL_MS = 180;


// ================================
// DOM ELEMENTS
// ================================
let stream = null;
let ws = null;

const video = document.getElementById("videoEl");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const captureBtn = document.getElementById("captureBtn");
const framesList = document.getElementById("frames");

let sendingInterval = null;


// ================================
//  START CAMERA
// ================================
startBtn.onclick = async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: 1280, height: 720 },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        captureBtn.disabled = false;

        startStreaming();
        setupWebSocket();

    } catch (err) {
        alert("Camera error: " + err);
    }
};


// ================================
// STOP CAMERA
// ================================
stopBtn.onclick = () => {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    if (ws) ws.close();
    if (sendingInterval) clearInterval(sendingInterval);

    startBtn.disabled = false;
    stopBtn.disabled = true;
};


// ================================
//  SEND FRAMES
// ================================
function startStreaming() {
    sendingInterval = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (!video || video.readyState < 2) return;

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const g = canvas.getContext("2d");
        g.drawImage(video, 0, 0);

        const dataURL = canvas.toDataURL("image/jpeg", 0.6);
        ws.send(dataURL);

    }, SEND_INTERVAL_MS);
}


// ================================
//  WEBSOCKET
// ================================
function setupWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => console.log("WS connected");

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        drawDetections(msg);
    };

    ws.onclose = () => {
        console.log("WS closed, retrying...");
        setTimeout(setupWebSocket, 1000);
    };
}


// ================================
// DRAW DETECTIONS
// ================================
function drawDetections(msg) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!msg.detections) return;

    msg.detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;

        // Box
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label
        ctx.fillStyle = "yellow";
        ctx.font = "15px Arial";
        ctx.fillText(`${det.class_id}  ${det.conf.toFixed(2)}`, x1, y1 - 5);

        // Center
        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.arc(det.cx, det.cy, 4, 0, Math.PI * 2);
        ctx.fill();
    });
}


// ================================
// CAPTURE SINGLE FRAME (FIXED)
// ================================
captureBtn.onclick = () => {
    if (!video || video.readyState < 2) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const g = canvas.getContext("2d");
    g.drawImage(video, 0, 0);

    const dataURL = canvas.toDataURL("image/jpeg", 0.95);

    // download file
    const a = document.createElement("a");
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = dataURL;
    a.download = `capture_${ts}.jpg`;
    a.click();
};
