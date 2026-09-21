/**
 * WhatsApp Alert Microservice for AI Pothole Detection System.
 * Built on whatsapp-web.js + Express.
 */

const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const QRCode = require('qrcode');
const express = require('express');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.env.WHATSAPP_PORT || '5005', 10);
const AUTH_DIR = path.resolve(__dirname, '.wwebjs_auth');
const QR_PNG_PATH = path.resolve(__dirname, '../outputs/whatsapp_qr.png');

const app = express();
app.use(express.json({ limit: '20mb' }));

let status = 'initializing';
let qrString = null;
let qrDataUrl = null;
let clientInfo = null;

// Locate Google Chrome executable
const CHROME_PATH = fs.existsSync('/usr/bin/google-chrome')
    ? '/usr/bin/google-chrome'
    : (fs.existsSync('/usr/bin/chromium-browser') ? '/usr/bin/chromium-browser' : undefined);

console.log(`[WHATSAPP BOT] Starting client using Chrome at: ${CHROME_PATH || 'default'}`);

const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: AUTH_DIR,
    }),
    puppeteer: {
        headless: true,
        executablePath: CHROME_PATH,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--disable-gpu',
        ],
    },
});

// Event: QR Received
client.on('qr', async (qr) => {
    status = 'qr_ready';
    qrString = qr;

    try {
        qrDataUrl = await QRCode.toDataURL(qr, { margin: 1, scale: 8 });
        // Also save to outputs/whatsapp_qr.png
        const qrDir = path.dirname(QR_PNG_PATH);
        if (!fs.existsSync(qrDir)) fs.mkdirSync(qrDir, { recursive: true });
        await QRCode.toFile(QR_PNG_PATH, qr);
    } catch (err) {
        console.error('[WHATSAPP BOT] Failed to generate QR image:', err.message);
    }

    console.log('\n============================================================');
    console.log('              WHATSAPP BOT QR AUTHENTICATION');
    console.log('============================================================');
    console.log('Scan the QR code below using your WhatsApp mobile app:');
    console.log('  1. Open WhatsApp on your phone');
    console.log('  2. Tap Menu / Settings -> Linked Devices -> Link a Device');
    console.log('  3. Scan this QR code:');
    console.log('------------------------------------------------------------');
    qrcode.generate(qr, { small: true });
    console.log('============================================================');
    console.log(`QR image also saved to: ${QR_PNG_PATH}`);
    console.log('Waiting for scan...\n');
});

// Event: Authenticated
client.on('authenticated', () => {
    status = 'authenticated';
    console.log('[WHATSAPP BOT] WhatsApp authentication successful!');
});

// Event: Auth Failure
client.on('auth_failure', (msg) => {
    status = 'auth_failure';
    console.error('[WHATSAPP BOT] Authentication failed:', msg);
});

// Event: Ready
client.on('ready', () => {
    status = 'ready';
    qrString = null;
    qrDataUrl = null;
    clientInfo = client.info;
    const phone = clientInfo && clientInfo.wid ? clientInfo.wid.user : 'Unknown';
    console.log('\n============================================================');
    console.log(`[WHATSAPP BOT] Ready! Connected as mobile user: +${phone}`);
    console.log(`[WHATSAPP BOT] Microservice listening on http://127.0.0.1:${PORT}`);
    console.log('============================================================\n');
});

// Event: Disconnected
client.on('disconnected', (reason) => {
    status = 'disconnected';
    console.log('[WHATSAPP BOT] Disconnected:', reason);
});

// ---------------------------------------------------------------------------
// HTTP API Endpoints
// ---------------------------------------------------------------------------

// Status endpoint
app.get('/status', (req, res) => {
    res.json({
        status,
        ready: status === 'ready',
        user: clientInfo && clientInfo.wid ? clientInfo.wid.user : null,
        qr_available: qrString !== null,
        qr_image_path: fs.existsSync(QR_PNG_PATH) ? QR_PNG_PATH : null,
    });
});

// QR Code endpoints
app.get('/qr', (req, res) => {
    if (!qrString) {
        return res.status(status === 'ready' ? 200 : 404).json({
            error: status === 'ready' ? 'Already connected' : 'QR code not generated yet',
            status,
        });
    }
    res.json({
        qr: qrString,
        data_url: qrDataUrl,
    });
});

app.get('/qr.png', (req, res) => {
    if (fs.existsSync(QR_PNG_PATH)) {
        res.sendFile(QR_PNG_PATH);
    } else {
        res.status(404).send('QR code image not found.');
    }
});

// Send Message & Media Endpoint
app.post('/send', async (req, res) => {
    if (status !== 'ready') {
        return res.status(503).json({
            success: false,
            error: `WhatsApp client is not ready (current state: ${status}). Please scan the QR code first.`,
        });
    }

    const { phone, message, image_path } = req.body;
    if (!phone || !message) {
        return res.status(400).json({
            success: false,
            error: 'Missing required fields: "phone" and "message".',
        });
    }

    // Clean and normalize phone number
    let cleanNumber = phone.replace(/\D/g, '');
    if (cleanNumber.length === 10) {
        cleanNumber = `91${cleanNumber}`;
    }
    if (!cleanNumber.endsWith('@c.us')) {
        cleanNumber = `${cleanNumber}@c.us`;
    }

    try {
        let sentMsg;
        if (image_path && fs.existsSync(image_path)) {
            console.log(`[WHATSAPP BOT] Sending alert with snapshot ${image_path} to ${cleanNumber}...`);
            const media = MessageMedia.fromFilePath(image_path);
            sentMsg = await client.sendMessage(cleanNumber, media, { caption: message });
        } else {
            console.log(`[WHATSAPP BOT] Sending text alert to ${cleanNumber}...`);
            sentMsg = await client.sendMessage(cleanNumber, message);
        }

        console.log(`[WHATSAPP BOT] Message delivered successfully! ID: ${sentMsg.id._serialized}`);
        return res.json({
            success: true,
            message_id: sentMsg.id._serialized,
            recipient: cleanNumber,
            timestamp: new Date().toISOString(),
        });
    } catch (err) {
        console.error('[WHATSAPP BOT] Failed to send message:', err);
        return res.status(500).json({
            success: false,
            error: err.message,
        });
    }
});

// Start Express Server
app.listen(PORT, '127.0.0.1', () => {
    console.log(`[WHATSAPP BOT] HTTP Server running on http://127.0.0.1:${PORT}`);
    console.log('[WHATSAPP BOT] Initializing WhatsApp Web client...');
    client.initialize().catch((err) => {
        console.error('[WHATSAPP BOT] Initialization error:', err);
    });
});
