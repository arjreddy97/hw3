import numpy as np
import cv2

rows, cols = 6, 9
square_size = 50
board = np.zeros(((rows + 1) * square_size, (cols + 1) * square_size), dtype=np.uint8)

for i in range(rows + 1):
    for j in range(cols + 1):
        if (i + j) % 2 == 0:
            y1, y2 = i * square_size, (i + 1) * square_size
            x1, x2 = j * square_size, (j + 1) * square_size
            board[y1:y2, x1:x2] = 255

cv2.imwrite("checkerboard.png", board)
print("✅ Saved checkerboard.png")
