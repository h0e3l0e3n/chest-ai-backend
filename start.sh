#!/bin/bash
# Start the AI Worker in the background
python ai_worker.py &

# Start the Node.js Server in the foreground
node server.js
