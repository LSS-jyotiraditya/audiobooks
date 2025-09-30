const player = document.getElementById("player");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const stopBtn = document.getElementById("stopBtn");

playBtn.addEventListener("click", () => {
  if (player.paused) {
    player.play().catch(error => {
      console.error("Error playing audio:", error);
    });
  }
});

pauseBtn.addEventListener("click", () => {
  if (!player.paused) {
    player.pause();
  }
});

stopBtn.addEventListener("click", () => {
  player.pause();
  player.currentTime = 0; // Reset to the beginning
});