document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.querySelector("#record");
  const stopBtn = document.querySelector("#stop");
  const statusText = document.querySelector("#status");
  stopBtn.disabled = true;

  let recorder, audioChunks;

  async function sendAudio(blob) {
    const formData = new FormData();
    formData.append(
      "file",
      new File([blob], "audio.webm", { type: "audio/webm" })
    );
    statusText.textContent = "🔄 Processing...";

    try {
      const res = await fetch("http://127.0.0.1:5002/predict_emotion", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        statusText.textContent = `🎯 Emotion: ${data.emotion} (${(
          data.confidence * 100
        ).toFixed(1)}%)`;
      } else {
        throw new Error(data.detail || "Unknown error");
      }
    } catch (err) {
      statusText.textContent = err.message;
    }
  }

  startBtn.onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    audioChunks = [];

    recorder.ondataavailable = (e) => audioChunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      sendAudio(blob);
    };

    recorder.start();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusText.textContent = "🎤 Recording...";
  };

  stopBtn.onclick = async () => {
    statusText.textContent = "🛑 Stopped. Sending audio...";
    recorder.stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };
});
