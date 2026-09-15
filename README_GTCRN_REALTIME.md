# Real-Time GTCRN Audio Streaming Architecture

This document outlines an architectural approach for implementing ultra-low latency, real-time noise suppression using the **GTCRN (Gated Temporal Convolutional Recurrent Network)** model. While the principles here apply to generalized real-time AI audio processing, this guide specifically details how to deploy GTCRN with a web browser frontend and a stateful Python backend.

Traditional REST API architectures process audio in "batches" (e.g., uploading an entire `.wav` file, processing it with GTCRN, and returning the result). This introduces seconds of latency, making it unsuitable for live communication. Achieving sub-50ms latency requires a continuous streaming architecture tailored to recurrent models.

## 🏗️ Core Architecture Overview

The streaming architecture consists of three primary layers:

1. **Low-Latency Frontend Capture (Web Audio API)**
2. **Real-time Transport Layer (WebSockets)**
3. **Stateful AI Inference (Recurrent Neural Networks / Overlap-Add)**

---

### 1. Low-Latency Frontend Capture

To achieve real-time performance in a web browser, standard HTML5 `<audio>` tags and `MediaRecorder` APIs cannot be used, as they buffer audio too heavily. 

**The Solution:** `AudioWorkletProcessor`
* **Microphone Access:** The application uses `navigator.mediaDevices.getUserMedia()` to capture the live microphone feed. Browser-level enhancements (echo cancellation, noise suppression, auto-gain) should generally be disabled to provide the AI model with raw, unadulterated PCM audio.
* **Audio Thread Processing:** An `AudioWorkletProcessor` is registered to run on the browser's dedicated audio thread. 
* **Micro-Chunking:** Instead of waiting for seconds of audio, the Worklet collects microscopic chunks of audio. For example, capturing exactly **256 samples** at a 16 kHz sample rate yields a buffer representing exactly **16 milliseconds** of audio.
* **Immediate Dispatch:** As soon as a chunk is full, the raw `Float32` array is posted to the main JavaScript thread for immediate transmission.

### 2. Real-Time Transport Layer

Sending HTTP requests (like `POST`) for every 16ms chunk incurs massive TCP handshake and HTTP header overhead, destroying latency budgets.

**The Solution:** WebSockets
* **Persistent Connection:** A bidirectional WebSocket connection is established between the browser and the backend server.
* **Binary Streaming:** The `Float32` audio chunks are transmitted as raw binary payloads (`ArrayBuffer`) rather than base64-encoded strings, minimizing bandwidth and serialization overhead.
* **Full-Duplex:** The WebSocket receives microphone chunks from the client and simultaneously pushes processed chunks back to the client.

### 3. Stateful AI Inference

AI models designed for streaming (like GTCRN) are typically Recurrent Neural Networks (RNNs). They cannot process isolated 16ms chunks independently because they need historical context to understand what constitutes "noise" versus "voice."

**The Solution:** Stateful Sessions and Overlap-Add (OLA)
* **Connection-Scoped State:** The backend must maintain a persistent session object in memory for the duration of the WebSocket connection. This session holds the recurrent cache tensors (e.g., convolution caches, transformer caches) that the model updates during each inference step.
* **Framing & Overlap-Add (OLA):** 
  * AI models often require a larger "window" of audio than the tiny chunks transmitted over the network (e.g., a 512-sample window for a 256-sample chunk).
  * **Concatenation:** When the backend receives a new 256-sample chunk, it prepends the *previous* 256-sample chunk to form a 512-sample frame.
  * **Inference:** This frame undergoes an analysis window, Fast Fourier Transform (FFT), and is passed into the ONNX runtime along with the historical caches. The model outputs the cleaned frame and the updated caches.
  * **Reconstruction:** The output undergoes an Inverse FFT and synthesis window. Because of the overlapping windows, the first half of the new output is mathematically added to the leftover tail of the *previous* output. 
  * **Yielding:** This Overlap-Add process finalizes exactly 256 samples of denoised audio, which are instantly sent back over the WebSocket.

---

## ⏱️ Latency Breakdown

When implemented correctly on a local network or edge server, this architecture can achieve nearly imperceptible latency:

1. **Frontend Buffering:** 16 ms (waiting to collect 256 samples at 16 kHz)
2. **Network Ping (Up):** ~2-5 ms 
3. **AI Inference (CPU/ONNX):** ~3-6 ms
4. **Network Ping (Down):** ~2-5 ms
5. **Frontend Playback Buffer:** ~2-5 ms

**Total Round-Trip Latency:** ~25 - 40 ms.

## ⚠️ Key Considerations

* **Sample Rate Enforcement:** AI models are strictly trained on specific sample rates (typically 16 kHz or 48 kHz). The Web Audio API's `AudioContext` must be explicitly initialized with the correct sample rate, or the browser will default to the hardware rate, causing the AI to severely distort the audio.
* **Feedback Loops:** Because latency is so low, playing the denoised audio through computer speakers will cause the microphone to instantly pick it up again, creating an endless echo loop. This architecture requires users to wear headphones, or requires the implementation of a robust Acoustic Echo Canceller (AEC) before the AI step.
