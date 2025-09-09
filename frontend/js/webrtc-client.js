/**
 * WebRTC Client for Voxtral 3B Real-Time Streaming
 * Handles WebSocket connections, audio streaming, and UI interactions
 */

class VoxtralStreamingClient {
    constructor() {
        this.websocket = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.audioWorkletNode = null;
        this.isConnected = false;
        this.isRecording = false;
        this.sessionId = null;
        this.clientId = null;

        // Performance tracking
        this.messageCount = 0;
        this.latencyStats = [];
        this.connectionStartTime = null;

        // Audio processing
        this.audioProcessor = null;
        this.vadSensitivity = 2;
        this.audioQuality = 16000;
        this.noiseReduction = true;
        this.autoGain = true;

        // UI elements
        this.elements = {};

        this.init();
    }

    async init() {
        this.initUIElements();
        this.initEventListeners();
        this.initAudioVisualization();

        // Initialize audio processor
        this.audioProcessor = new AudioProcessor({
            sampleRate: this.audioQuality,
            vadSensitivity: this.vadSensitivity,
            noiseReduction: this.noiseReduction,
            autoGain: this.autoGain
        });

        this.updateUI();
        this.log('Client initialized');
    }

    initUIElements() {
        // Connection controls
        this.elements.connectBtn = document.getElementById('connectBtn');
        this.elements.disconnectBtn = document.getElementById('disconnectBtn');
        this.elements.startAudioBtn = document.getElementById('startAudioBtn');
        this.elements.stopAudioBtn = document.getElementById('stopAudioBtn');

        // Status indicators
        this.elements.connectionStatus = document.getElementById('connectionStatus');
        this.elements.connectionText = document.getElementById('connectionText');
        this.elements.audioStatus = document.getElementById('audioStatus');
        this.elements.audioText = document.getElementById('audioText');
        this.elements.processingStatus = document.getElementById('processingStatus');
        this.elements.processingText = document.getElementById('processingText');

        // Conversation
        this.elements.messages = document.getElementById('messages');
        this.elements.textInput = document.getElementById('textInput');
        this.elements.sendTextBtn = document.getElementById('sendTextBtn');

        // Settings
        this.elements.vadSensitivity = document.getElementById('vadSensitivity');
        this.elements.vadValue = document.getElementById('vadValue');
        this.elements.audioQuality = document.getElementById('audioQuality');
        this.elements.noiseReduction = document.getElementById('noiseReduction');
        this.elements.autoGain = document.getElementById('autoGain');

        // Metrics
        this.elements.latencyMetric = document.getElementById('latencyMetric');
        this.elements.messageCount = document.getElementById('messageCount');
        this.elements.audioLevel = document.getElementById('audioLevel');
        this.elements.connectionTime = document.getElementById('connectionTime');

        // Debug
        this.elements.sessionId = document.getElementById('sessionId');
        this.elements.clientId = document.getElementById('clientId');
        this.elements.webrtcStatus = document.getElementById('webrtcStatus');
        this.elements.audioContextStatus = document.getElementById('audioContextStatus');

        // Audio visualization
        this.elements.audioCanvas = document.getElementById('audioCanvas');
        this.canvasContext = this.elements.audioCanvas.getContext('2d');
    }

    initEventListeners() {
        // Connection controls
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.elements.startAudioBtn.addEventListener('click', () => this.startAudio());
        this.elements.stopAudioBtn.addEventListener('click', () => this.stopAudio());

        // Text input
        this.elements.sendTextBtn.addEventListener('click', () => this.sendTextMessage());
        this.elements.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendTextMessage();
            }
        });

        // Settings
        this.elements.vadSensitivity.addEventListener('input', (e) => {
            this.vadSensitivity = parseInt(e.target.value);
            this.elements.vadValue.textContent = this.vadSensitivity;
            if (this.audioProcessor) {
                this.audioProcessor.updateSettings({ vadSensitivity: this.vadSensitivity });
            }
        });

        this.elements.audioQuality.addEventListener('change', (e) => {
            this.audioQuality = parseInt(e.target.value);
            if (this.audioProcessor) {
                this.audioProcessor.updateSettings({ sampleRate: this.audioQuality });
            }
        });

        this.elements.noiseReduction.addEventListener('change', (e) => {
            this.noiseReduction = e.target.checked;
            if (this.audioProcessor) {
                this.audioProcessor.updateSettings({ noiseReduction: this.noiseReduction });
            }
        });

        this.elements.autoGain.addEventListener('change', (e) => {
            this.autoGain = e.target.checked;
            if (this.audioProcessor) {
                this.audioProcessor.updateSettings({ autoGain: this.autoGain });
            }
        });

        // Update connection time periodically
        setInterval(() => this.updateConnectionTime(), 1000);
    }

    async connect() {
        try {
            this.log('Connecting to server...');

            // Create session first
            const response = await fetch('/session/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`Failed to create session: ${response.statusText}`);
            }

            const sessionData = await response.json();
            this.sessionId = sessionData.session_id;
            this.elements.sessionId.textContent = this.sessionId;

            // Connect WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/audio/${this.sessionId}`;

            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = (event) => {
                this.isConnected = true;
                this.connectionStartTime = Date.now();
                this.log('WebSocket connected');
                this.updateConnectionStatus('connected', 'Connected');
                this.updateUI();
            };

            this.websocket.onmessage = (event) => {
                this.handleMessage(event.data);
            };

            this.websocket.onclose = (event) => {
                this.isConnected = false;
                this.log(`WebSocket closed: ${event.code} ${event.reason}`);
                this.updateConnectionStatus('disconnected', 'Disconnected');
                this.updateUI();
            };

            this.websocket.onerror = (error) => {
                this.log(`WebSocket error: ${error}`);
                this.updateConnectionStatus('error', 'Error');
            };

        } catch (error) {
            this.log(`Connection failed: ${error.message}`);
            this.updateConnectionStatus('error', 'Failed');
        }
    }

    async disconnect() {
        try {
            await this.stopAudio();

            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }

            this.isConnected = false;
            this.sessionId = null;
            this.clientId = null;

            this.updateConnectionStatus('disconnected', 'Disconnected');
            this.updateUI();

            this.log('Disconnected from server');

        } catch (error) {
            this.log(`Disconnect error: ${error.message}`);
        }
    }

    async startAudio() {
        try {
            this.log('Starting audio capture...');

            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.audioQuality,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: this.noiseReduction,
                    autoGainControl: this.autoGain
                }
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.audioQuality
            });

            // Load audio worklet processor
            try {
                await this.audioContext.audioWorklet.addModule('js/audio-processor-worklet.js');
            } catch (error) {
                this.log('Audio worklet not available, using fallback');
            }

            // Create audio source
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Create worklet node or script processor
            if (this.audioContext.audioWorklet) {
                this.audioWorkletNode = new AudioWorkletNode(this.audioContext, 'audio-processor');
                this.audioWorkletNode.port.onmessage = (event) => {
                    this.processAudioData(event.data);
                };
                source.connect(this.audioWorkletNode);
            } else {
                // Fallback to ScriptProcessorNode
                this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
                this.scriptProcessor.onaudioprocess = (event) => {
                    const inputBuffer = event.inputBuffer.getChannelData(0);
                    this.processAudioData(inputBuffer);
                };
                source.connect(this.scriptProcessor);
                this.scriptProcessor.connect(this.audioContext.destination);
            }

            this.isRecording = true;
            this.updateAudioStatus('recording', 'Recording');
            this.updateUI();

            this.log('Audio capture started');

        } catch (error) {
            this.log(`Audio start failed: ${error.message}`);
            this.updateAudioStatus('error', 'Error');
        }
    }

    async stopAudio() {
        try {
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
                this.mediaStream = null;
            }

            if (this.audioWorkletNode) {
                this.audioWorkletNode.disconnect();
                this.audioWorkletNode = null;
            }

            if (this.scriptProcessor) {
                this.scriptProcessor.disconnect();
                this.scriptProcessor = null;
            }

            if (this.audioContext) {
                await this.audioContext.close();
                this.audioContext = null;
            }

            this.isRecording = false;
            this.updateAudioStatus('stopped', 'Stopped');
            this.updateUI();

            this.log('Audio capture stopped');

        } catch (error) {
            this.log(`Audio stop failed: ${error.message}`);
        }
    }

    processAudioData(audioData) {
        if (!this.isConnected || !this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            return;
        }

        try {
            // Process through audio processor
            const processedData = this.audioProcessor.process(audioData);

            if (processedData && processedData.shouldSend) {
                // Convert to Int16Array for transmission
                const int16Array = new Int16Array(processedData.audio.length);
                for (let i = 0; i < processedData.audio.length; i++) {
                    int16Array[i] = Math.max(-32768, Math.min(32767, processedData.audio[i] * 32767));
                }

                // Send binary audio data
                this.websocket.send(int16Array.buffer);

                // Update processing status
                this.updateProcessingStatus('processing', 'Processing');
                setTimeout(() => {
                    this.updateProcessingStatus('idle', 'Idle');
                }, 100);
            }

            // Update audio visualization
            this.updateAudioVisualization(processedData ? processedData.audio : audioData);

            // Update audio level metric
            const audioLevel = this.calculateAudioLevel(audioData);
            this.elements.audioLevel.textContent = `${Math.round(audioLevel * 100)}%`;

        } catch (error) {
            this.log(`Audio processing error: ${error.message}`);
        }
    }

    handleMessage(data) {
        try {
            const message = JSON.parse(data);
            this.messageCount++;
            this.elements.messageCount.textContent = this.messageCount;

            switch (message.type) {
                case 'status':
                    if (message.client_id) {
                        this.clientId = message.client_id;
                        this.elements.clientId.textContent = this.clientId;
                    }
                    break;

                case 'text_response':
                    this.addMessage('assistant', message.text, message.timestamp);
                    break;

                case 'error':
                    this.log(`Server error: ${message.error}`);
                    this.addMessage('system', `Error: ${message.error}`, message.timestamp);
                    break;

                case 'heartbeat':
                    // Calculate latency
                    const latency = Date.now() - (message.timestamp * 1000);
                    this.latencyStats.push(latency);
                    if (this.latencyStats.length > 10) {
                        this.latencyStats.shift();
                    }
                    const avgLatency = this.latencyStats.reduce((a, b) => a + b, 0) / this.latencyStats.length;
                    this.elements.latencyMetric.textContent = `${Math.round(avgLatency)}ms`;
                    break;
            }

        } catch (error) {
            this.log(`Message parsing error: ${error.message}`);
        }
    }

    sendTextMessage() {
        const text = this.elements.textInput.value.trim();
        if (!text || !this.isConnected) {
            return;
        }

        try {
            const message = {
                type: 'text_input',
                text: text,
                timestamp: Date.now() / 1000
            };

            this.websocket.send(JSON.stringify(message));
            this.addMessage('user', text, Date.now() / 1000);
            this.elements.textInput.value = '';

        } catch (error) {
            this.log(`Send text error: ${error.message}`);
        }
    }

    addMessage(sender, text, timestamp) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const timeDiv = document.createElement('div');
        timeDiv.className = 'timestamp';
        timeDiv.textContent = new Date(timestamp * 1000).toLocaleTimeString();

        const textDiv = document.createElement('div');
        textDiv.textContent = text;

        messageDiv.appendChild(timeDiv);
        messageDiv.appendChild(textDiv);

        this.elements.messages.appendChild(messageDiv);
        this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    }

    updateConnectionStatus(status, text) {
        this.elements.connectionStatus.className = `status-dot ${status}`;
        this.elements.connectionText.textContent = text;
    }

    updateAudioStatus(status, text) {
        this.elements.audioStatus.className = `status-dot ${status}`;
        this.elements.audioText.textContent = text;
    }

    updateProcessingStatus(status, text) {
        this.elements.processingStatus.className = `status-dot ${status}`;
        this.elements.processingText.textContent = text;
    }

    updateConnectionTime() {
        if (this.connectionStartTime) {
            const elapsed = Math.floor((Date.now() - this.connectionStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            this.elements.connectionTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    updateUI() {
        // Enable/disable controls based on state
        this.elements.connectBtn.disabled = this.isConnected;
        this.elements.disconnectBtn.disabled = !this.isConnected;
        this.elements.startAudioBtn.disabled = !this.isConnected || this.isRecording;
        this.elements.stopAudioBtn.disabled = !this.isRecording;
        this.elements.textInput.disabled = !this.isConnected;
        this.elements.sendTextBtn.disabled = !this.isConnected;

        // Update debug info
        this.elements.webrtcStatus.textContent = this.isConnected ? 'Connected' : 'Disconnected';
        this.elements.audioContextStatus.textContent = this.audioContext ? 
            this.audioContext.state : 'Not initialized';
    }

    initAudioVisualization() {
        this.visualizationData = new Float32Array(128);
        this.animateVisualization();
    }

    updateAudioVisualization(audioData) {
        if (audioData && audioData.length > 0) {
            // Downsample audio data for visualization
            const step = Math.floor(audioData.length / this.visualizationData.length);
            for (let i = 0; i < this.visualizationData.length; i++) {
                this.visualizationData[i] = Math.abs(audioData[i * step] || 0);
            }
        }
    }

    animateVisualization() {
        const canvas = this.elements.audioCanvas;
        const ctx = this.canvasContext;
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw waveform
        ctx.strokeStyle = '#007bff';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const barWidth = width / this.visualizationData.length;

        for (let i = 0; i < this.visualizationData.length; i++) {
            const barHeight = this.visualizationData[i] * height;
            const x = i * barWidth;
            const y = height - barHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        requestAnimationFrame(() => this.animateVisualization());
    }

    calculateAudioLevel(audioData) {
        if (!audioData || audioData.length === 0) return 0;

        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }

        return Math.sqrt(sum / audioData.length);
    }

    log(message) {
        console.log(`[VoxtralClient] ${new Date().toISOString()}: ${message}`);
    }
}

// Initialize client when page loads
window.addEventListener('load', () => {
    window.voxtralClient = new VoxtralStreamingClient();
});