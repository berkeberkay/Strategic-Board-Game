import pygame

WIDTH, HEIGHT = 600, 600
ROWS, COLS = 7, 7
SQUARE_SIZE = WIDTH // COLS
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
HIGHLIGHT = (0, 255, 0)

# Piece Classes
class TrianglePiece:
    PADDING = 15
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = 0
        self.y = 0
        self.calc_pos()

    def calc_pos(self):
        self.x = self.col * SQUARE_SIZE + SQUARE_SIZE // 2
        self.y = self.row * SQUARE_SIZE + SQUARE_SIZE // 2

    def draw(self, win):
        half = SQUARE_SIZE // 2
        points = [
            (self.x, self.y - half + self.PADDING),
            (self.x - half + self.PADDING, self.y + half - self.PADDING),
            (self.x + half - self.PADDING, self.y + half - self.PADDING),
        ]
        pygame.draw.polygon(win, RED, points)

    def move(self, row, col):
        self.row = row
        self.col = col
        self.calc_pos()


class CirclePiece:
    PADDING = 15
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = 0
        self.y = 0
        self.calc_pos()

    def calc_pos(self):
        self.x = self.col * SQUARE_SIZE + SQUARE_SIZE // 2
        self.y = self.row * SQUARE_SIZE + SQUARE_SIZE // 2

    def draw(self, win):
        radius = SQUARE_SIZE // 3
        pygame.draw.circle(win, BLUE, (self.x, self.y), radius)

    def move(self, row, col):
        self.row = row
        self.col = col
        self.calc_pos()
