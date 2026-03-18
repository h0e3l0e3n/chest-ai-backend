const express = require('express');
const multer = require('multer');
const http = require('http');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors()); // This allows your mobile app to talk to the server
const upload = multer({ dest: 'uploads/' });

// This is the "Door" the mobile app will knock on
app.post('/analyze', upload.single('xray'), (req, res) => {
    if (!req.file) return res.status(400).send('No file uploaded.');

    const imagePath = path.join(__dirname, req.file.path);
    const role = req.body.role || 'General User';
    const language = req.body.language || 'English';

    console.log(`🔍 Processing X-Ray... (Role: ${role}, Language: ${language})`);
    
    // Prepare the payload for our persistent AI Worker
    const postData = JSON.stringify({
        imagePath: imagePath,
        role: role,
        language: language
    });

    const options = {
        hostname: 'localhost',
        port: 5005,
        path: '/analyze',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': postData.length
        }
    };

    const reqToAi = http.request(options, (aiRes) => {
        let data = '';
        aiRes.on('data', (chunk) => { data += chunk; });
        aiRes.on('end', () => {
            try {
                const response = JSON.parse(data);
                res.json(response);
            } catch (e) {
                console.error("AI Error:", e);
                res.json({ disease: "Service Error", explanation: "The AI worker is offline." });
            }
        });
    });

    reqToAi.on('error', (e) => {
        console.warn("⚠️ AI Worker not found. Falling back to MOCK AI response.");
        res.json({
            disease: "Mock Diagnosis (AI Worker Offline)",
            explanation: `[MOCK EXPLANATION]\n\nPlease run 'py ai_worker.py' to enable high-performance results.\n\nRole: ${role}\nLanguage: ${language}`
        });
    });

    reqToAi.write(postData);
    reqToAi.end();
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`🚀 Messenger is ready at http://localhost:${PORT}`);
});