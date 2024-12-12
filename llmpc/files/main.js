document.addEventListener("DOMContentLoaded", function() {
    const canvas = document.getElementById("gameCanvas");
    const context = canvas.getContext("2d");

    const bird = {
        x: 50,
        y: 150,
        width: 20,
        height: 20,
        gravity: 0.6,
        lift: -10,
        velocity: 0
    };

    function drawBird() {
        context.fillStyle = "#ff0";
        context.fillRect(bird.x, bird.y, bird.width, bird.height);
    }

    function update() {
        bird.velocity += bird.gravity;
        bird.y += bird.velocity;

        if (bird.y + bird.height > canvas.height) {
            bird.y = canvas.height - bird.height;
            bird.velocity = 0;
        }

        if (bird.y < 0) {
            bird.y = 0;
            bird.velocity = 0;
        }
    }

    function gameLoop() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        update();
        drawBird();
        requestAnimationFrame(gameLoop);
    }

    canvas.addEventListener("click", function() {
        bird.velocity += bird.lift;
    });

    gameLoop();
});