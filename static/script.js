const socket = io();

socket.on('score_update', (data) => {
    const scoreDisplay = document.getElementById('score-display');
    const newScore = document.createElement('p');
    newScore.textContent = data.message;
    scoreDisplay.appendChild(newScore);
});
