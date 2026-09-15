class RealtimeProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.chunkSize = 256;
        this.inputBuffer = new Float32Array(this.chunkSize);
        this.inputPtr = 0;
        
        // Output playback queue
        this.outputQueue = [];
        this.currentOutputChunk = null;
        this.outputPtr = 0;

        this.port.onmessage = (e) => {
            if (e.data.type === 'play') {
                this.outputQueue.push(new Float32Array(e.data.buffer));
            }
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];

        if (input && input.length > 0 && input[0].length > 0) {
            const inputChannel = input[0];
            for (let i = 0; i < inputChannel.length; i++) {
                this.inputBuffer[this.inputPtr++] = inputChannel[i];
                if (this.inputPtr >= this.chunkSize) {
                    // Send to main thread
                    this.port.postMessage({
                        type: 'chunk',
                        buffer: this.inputBuffer.slice().buffer
                    });
                    this.inputPtr = 0;
                }
            }
        }

        if (output && output.length > 0) {
            const outputChannel = output[0];
            for (let i = 0; i < outputChannel.length; i++) {
                if (!this.currentOutputChunk || this.outputPtr >= this.currentOutputChunk.length) {
                    if (this.outputQueue.length > 0) {
                        this.currentOutputChunk = this.outputQueue.shift();
                        this.outputPtr = 0;
                    } else {
                        this.currentOutputChunk = null;
                    }
                }
                
                if (this.currentOutputChunk) {
                    outputChannel[i] = this.currentOutputChunk[this.outputPtr++];
                } else {
                    outputChannel[i] = 0; // silence if underrun
                }
            }
            
            // Copy to other channels if stereo output
            for (let c = 1; c < output.length; c++) {
                output[c].set(output[0]);
            }
        }

        return true;
    }
}

registerProcessor('realtime-processor', RealtimeProcessor);
