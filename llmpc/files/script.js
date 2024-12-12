const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// Set the canvas dimensions
canvas.width = 320;
canvas.height = 480;

// Initial setup code here
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);