/**
 * Client-side Audio Processor
 * Handles audio processing, VAD, and buffering on the client side
 */

class AudioProcessor {
    constructor(config = {}) {
        this.config = {
            sampleRate: config.sampleRate || 16000,
            vadSensitivity: config.vadSensitivity || 2,
            noiseReduction: config.noiseReduction !== false,
            autoGain: config.autoGain !== false,
            frameSize: config.frameSize || 160, // 10ms at 16kHz
            vadFrames: config.vadFrames || 5,
            silenceThreshold: config.silenceThreshold || 0.01,
            ...config
        };

        // Processing state
        this.audioBuffer = [];
        this.vadHistory = [];
        this.lastSpeechTime = 0;
        this.isSpeaking = false;

        // Audio processing
        this.noiseGate = new NoiseGate(this.config.silenceThreshold);
        this.autoGainControl = new AutoGainControl();
        this.vadProcessor = new SimpleVAD(this.config.vadSensitivity);

        this.log('Audio processor initialized');
    }

    process(audioData) {
        try {
            // Convert to Float32Array if needed
            let processedAudio = audioData;
            if (audioData.constructor !== Float32Array) {
                processedAudio = new Float32Array(audioData);
            }

            // Apply noise reduction
            if (this.config.noiseReduction) {
                processedAudio = this.noiseGate.process(processedAudio);
            }

            // Apply auto gain control
            if (this.config.autoGain) {
                processedAudio = this.autoGainControl.process(processedAudio);
            }

            // Voice Activity Detection
            const vadResult = this.vadProcessor.process(processedAudio);
            this.vadHistory.push(vadResult);

            // Keep VAD history window
            if (this.vadHistory.length > this.config.vadFrames) {
                this.vadHistory.shift();
            }

            // Determine if currently speaking
            const speechRatio = this.vadHistory.filter(v => v).length / this.vadHistory.length;
            const currentlySpeaking = speechRatio > 0.4;

            if (currentlySpeaking) {
                this.lastSpeechTime = Date.now();
                this.isSpeaking = true;
            } else if (Date.now() - this.lastSpeechTime > 1000) { // 1 second silence
                this.isSpeaking = false;
            }

            // Buffer audio if speaking or recently spoke
            if (this.isSpeaking || (Date.now() - this.lastSpeechTime < 500)) {
                this.audioBuffer.push(...processedAudio);

                // Send buffered audio if enough accumulated
                if (this.audioBuffer.length >= this.config.sampleRate * 0.5) { // 500ms
                    const bufferToSend = new Float32Array(this.audioBuffer);
                    this.audioBuffer = [];

                    return {
                        audio: bufferToSend,
                        shouldSend: true,
                        vadResult: vadResult,
                        isSpeaking: this.isSpeaking,
                        speechRatio: speechRatio
                    };
                }
            }

            return {
                audio: processedAudio,
                shouldSend: false,
                vadResult: vadResult,
                isSpeaking: this.isSpeaking,
                speechRatio: speechRatio
            };

        } catch (error) {
            this.log(`Processing error: ${error.message}`);
            return null;
        }
    }

    updateSettings(newConfig) {
        Object.assign(this.config, newConfig);

        // Update processors
        this.vadProcessor.setSensitivity(this.config.vadSensitivity);
        this.noiseGate.setThreshold(this.config.silenceThreshold);

        this.log(`Settings updated: ${JSON.stringify(newConfig)}`);
    }

    reset() {
        this.audioBuffer = [];
        this.vadHistory = [];
        this.lastSpeechTime = 0;
        this.isSpeaking = false;

        this.noiseGate.reset();
        this.autoGainControl.reset();
        this.vadProcessor.reset();

        this.log('Audio processor reset');
    }

    log(message) {
        console.log(`[AudioProcessor] ${message}`);
    }
}

class SimpleVAD {
    constructor(sensitivity = 2) {
        this.sensitivity = sensitivity;
        this.energyThreshold = this.getThresholdForSensitivity(sensitivity);
        this.frameHistory = [];
        this.maxHistory = 5;
    }

    getThresholdForSensitivity(sensitivity) {
        // Map sensitivity (0-3) to energy threshold
        const thresholds = [0.001, 0.005, 0.01, 0.02];
        return thresholds[Math.min(3, Math.max(0, sensitivity))];
    }

    process(audioData) {
        // Calculate RMS energy
        let energy = 0;
        for (let i = 0; i < audioData.length; i++) {
            energy += audioData[i] * audioData[i];
        }
        energy = Math.sqrt(energy / audioData.length);

        // Zero crossing rate (simple frequency analysis)
        let zeroCrossings = 0;
        for (let i = 1; i < audioData.length; i++) {
            if ((audioData[i] >= 0) !== (audioData[i - 1] >= 0)) {
                zeroCrossings++;
            }
        }
        const zcr = zeroCrossings / audioData.length;

        // Combine energy and ZCR for VAD decision
        const isVoice = energy > this.energyThreshold && zcr < 0.3 && zcr > 0.01;

        // Add to history and smooth decision
        this.frameHistory.push(isVoice);
        if (this.frameHistory.length > this.maxHistory) {
            this.frameHistory.shift();
        }

        // Return majority vote from recent history
        const voiceFrames = this.frameHistory.filter(f => f).length;
        return voiceFrames > this.frameHistory.length / 2;
    }

    setSensitivity(sensitivity) {
        this.sensitivity = sensitivity;
        this.energyThreshold = this.getThresholdForSensitivity(sensitivity);
    }

    reset() {
        this.frameHistory = [];
    }
}

class NoiseGate {
    constructor(threshold = 0.01) {
        this.threshold = threshold;
        this.hysteresis = 0.5; // Ratio for close threshold
        this.attackTime = 0.001; // 1ms
        this.releaseTime = 0.1;  // 100ms
        this.envelope = 0;
        this.isOpen = false;
    }

    process(audioData) {
        const output = new Float32Array(audioData.length);

        for (let i = 0; i < audioData.length; i++) {
            const input = Math.abs(audioData[i]);

            // Envelope follower
            if (input > this.envelope) {
                this.envelope = input;
            } else {
                this.envelope *= (1 - this.releaseTime);
            }

            // Gate logic with hysteresis
            if (!this.isOpen && this.envelope > this.threshold) {
                this.isOpen = true;
            } else if (this.isOpen && this.envelope < this.threshold * this.hysteresis) {
                this.isOpen = false;
            }

            // Apply gate
            output[i] = this.isOpen ? audioData[i] : 0;
        }

        return output;
    }

    setThreshold(threshold) {
        this.threshold = threshold;
    }

    reset() {
        this.envelope = 0;
        this.isOpen = false;
    }
}

class AutoGainControl {
    constructor() {
        this.targetLevel = 0.1; // Target RMS level
        this.maxGain = 10.0;     // Maximum gain (20dB)
        this.minGain = 0.1;      // Minimum gain (-20dB)
        this.attackTime = 0.001; // Fast attack (1ms)
        this.releaseTime = 0.1;  // Slow release (100ms)
        this.currentGain = 1.0;
        this.envelope = 0;
    }

    process(audioData) {
        // Calculate input RMS level
        let rms = 0;
        for (let i = 0; i < audioData.length; i++) {
            rms += audioData[i] * audioData[i];
        }
        rms = Math.sqrt(rms / audioData.length);

        // Envelope follower
        if (rms > this.envelope) {
            this.envelope += (rms - this.envelope) * this.attackTime;
        } else {
            this.envelope += (rms - this.envelope) * this.releaseTime;
        }

        // Calculate desired gain
        let desiredGain = 1.0;
        if (this.envelope > 0) {
            desiredGain = this.targetLevel / this.envelope;
            desiredGain = Math.max(this.minGain, Math.min(this.maxGain, desiredGain));
        }

        // Smooth gain changes
        this.currentGain += (desiredGain - this.currentGain) * 0.01;

        // Apply gain
        const output = new Float32Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            output[i] = audioData[i] * this.currentGain;
        }

        return output;
    }

    reset() {
        this.currentGain = 1.0;
        this.envelope = 0;
    }
}