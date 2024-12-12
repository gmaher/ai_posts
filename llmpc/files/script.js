let bird = document.getElementById("bird");
let pole1 = document.getElementById("pole1");
let pole2 = document.getElementById("pole2");
let score = document.getElementById("score");
let pause = document.getElementById("pause");
let restart = document.getElementById("restart");
let easy = document.getElementById("easy");
let medium = document.getElementById("medium");
let hard = document.getElementById("hard");

let jump = 0;
let gravity = 0.5;
let gameStartTime = Date.now();
let obstaclesPassed = 0;
let isPaused = false;
let gameLoopInterval;

function startGame(difficulty) {
  switch (difficulty) {
    case 'easy':
      gravity = 0.3;
      break;
    case 'medium':
      gravity = 0.5;
      break;
    case 'hard':
      gravity = 0.7;
      break;
  }
  gameLoopInterval = setInterval(gameLoop, 20);
}

function gameLoop() {
  if (!isPaused) {
    jump -= gravity;
    bird.style.top = (bird.offsetTop - jump) + "px";
    
    if (bird.offsetTop < 0 || bird.offsetTop > 480-20) {
      endGame();
    }
  
    let pole1Right = parseInt(pole1.style.right) || 0;
    let pole2Right = parseInt(pole2.style.right) || 0;
  
    pole1.style.right = (pole1Right + 5) + "px";
    pole2.style.right = (pole2Right + 5) + "px";
    
    if (pole1Right > 320) {
      pole1.style.right = "0px";
      pole2.style.right = "0px";
      obstaclesPassed++;
      score.innerText = "Score: " + obstaclesPassed;
    }
  
    if (isColliding(bird, pole1) || isColliding(bird, pole2)) {
      endGame();
    }
  }
}

function isColliding(elem1, elem2) {
  let rect1 = elem1.getBoundingClientRect();
  let rect2 = elem2.getBoundingClientRect();

  return !(rect1.right < rect2.left || 
           rect1.left > rect2.right || 
           rect1.bottom < rect2.top || 
           rect1.top > rect2.bottom);
}

function endGame() {
  clearInterval(gameLoopInterval);
  let gameDuration = Date.now() - gameStartTime;
  alert("Game Over! You lasted " + gameDuration/1000 + " seconds and passed " + obstaclesPassed + " obstacles.");
}

document.body.addEventListener('keydown', function (e) {
  jump = 10;
});

pause.addEventListener('click', function () {
  isPaused = !isPaused;
  pause.innerText = isPaused ? "Resume" : "Pause";
});

restart.addEventListener('click', function () {
  location.reload();
});

easy.addEventListener('click', function () {
  startGame('easy');
});

medium.addEventListener('click', function () {
  startGame('medium');
});

hard.addEventListener('click', function () {
  startGame('hard');
});