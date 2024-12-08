import pygame
import math
from copy import deepcopy

# Constants
WIDTH, HEIGHT = 800, 800
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


# Board Class
class Board:
    def __init__(self):
        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.triangle_left = 4
        self.circle_left = 4
        self.create_board()

    def create_board(self):
        self.board[0][0] = TrianglePiece(0, 0)
        self.board[2][0] = TrianglePiece(2, 0)
        self.board[4][6] = TrianglePiece(4, 6)
        self.board[6][6] = TrianglePiece(6, 6)
        self.board[0][6] = CirclePiece(0, 6)
        self.board[2][6] = CirclePiece(2, 6)
        self.board[4][0] = CirclePiece(4, 0)
        self.board[6][0] = CirclePiece(6, 0)

    def draw(self, win):
        win.fill(WHITE)
        for row in range(ROWS):
            for col in range(COLS):
                pygame.draw.rect(
                    win,
                    (200, 200, 200),
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    1,
                )
                piece = self.board[row][col]
                if piece:
                    piece.draw(win)

    def draw_valid_moves(self, win, moves):
        for move in moves:
            row, col = move
            pygame.draw.circle(
                win,
                HIGHLIGHT,
                (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                15,
            )

    def move(self, piece, row, col):
        self.board[piece.row][piece.col], self.board[row][col] = (
            self.board[row][col],
            self.board[piece.row][piece.col],
        )
        piece.move(row, col)
        
    def remove(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
            if isinstance(piece, TrianglePiece):
                self.triangle_left -= 1
            elif isinstance(piece, CirclePiece):
                self.circle_left -= 1    

    def get_valid_moves(self, piece):
        moves = {}
        row, col = piece.row, piece.col
        for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < ROWS and 0 <= new_col < COLS and self.board[new_row][new_col] == 0:
                moves[(new_row, new_col)] = []
        return moves

    def check_and_capture(self):
        """
        Check and capture pieces based on specific rules:
        1. Pieces between a wall and an opponent are captured.
        2. Pieces between two opponent pieces are captured.
        3. Groups of pieces between two opponent pieces are captured.
        """
        def capture_line(line, is_row, index):
            """
            Captures pieces in a given line (row or column) and returns the indices to be cleared.
            """
            to_capture = set()
            n = len(line)
            i = 0
            while i < n:
                if not line[i]:  # Empty space, move to the next
                    i += 1
                    continue

                current_piece = line[i]
                opponent_type = CirclePiece if isinstance(current_piece, TrianglePiece) else TrianglePiece

                # Check if piece is at the start with a wall on the left
                if i == 0 and i + 4 < n and isinstance(line[i + 1], current_piece.__class__) and \
                isinstance(line[i + 2], current_piece.__class__) and \
                isinstance(line[i + 3], current_piece.__class__) and \
                isinstance(line[i + 4], opponent_type):
                    to_capture.update({(index, j) if is_row else (j, index) for j in range(i, i + 4)})
                    i += 4
                    continue

                if i == 0 and i + 3 < n and isinstance(line[i + 1], current_piece.__class__) and \
                isinstance(line[i + 2], current_piece.__class__) and \
                isinstance(line[i + 3], opponent_type):
                    to_capture.update({(index, j) if is_row else (j, index) for j in range(i, i + 3)})
                    i += 3
                    continue

                if i == 0 and i + 2 < n and isinstance(line[i + 1], current_piece.__class__) and \
                isinstance(line[i + 2], opponent_type):
                    to_capture.update({(index, j) if is_row else (j, index) for j in range(i, i + 2)})
                    i += 2
                    continue

                if i == 0 and i + 1 < n and isinstance(line[i + 1], opponent_type):
                    to_capture.add((index, i) if is_row else (i, index))
                    i += 1
                    continue

                # Check if piece is at the end with a wall on the right
                if i == n - 1 and i - 1 >= 0 and isinstance(line[i - 1], opponent_type):
                    to_capture.add((index, i) if is_row else (i, index))

                if i == n - 1 and i - 2 >= 0 and isinstance(line[i - 1], current_piece.__class__) and \
                isinstance(line[i - 2], opponent_type):
                    to_capture.update({(index, j) if is_row else (j, index) for j in range(i - 2, i + 1)})

                if i == n - 1 and i - 3 >= 0 and isinstance(line[i - 1], current_piece.__class__) and \
                isinstance(line[i - 2], current_piece.__class__) and \
                isinstance(line[i - 3], opponent_type):
                    to_capture.update({(index, j) if is_row else (j, index) for j in range(i - 3, i + 1)})

                # Check if piece is between two different pieces
                if 0 < i < n - 1 and isinstance(line[i - 1], opponent_type) and \
                isinstance(line[i + 1], opponent_type):
                    to_capture.add((index, i) if is_row else (i, index))
                    i += 1
                    continue

                if 0 < i < n - 2 and isinstance(line[i - 1], opponent_type) and \
                isinstance(line[i + 1], current_piece.__class__) and \
                isinstance(line[i + 2], opponent_type):
                    to_capture.update({(index, j) if is_row else (j, index) for j in range(i, i + 2)})
                    i += 2
                    continue

                if 0 < i < n - 3 and isinstance(line[i - 1], opponent_type) and \
                isinstance(line[i + 1], current_piece.__class__) and \
                isinstance(line[i + 2], current_piece.__class__) and \
                isinstance(line[i + 3], opponent_type):
                    to_capture.update({(index, j) if is_row else (j, index) for j in range(i, i + 3)})
                    i += 3
                    continue

                if 0 < i < n - 4 and isinstance(line[i - 1], opponent_type) and \
                isinstance(line[i + 1], current_piece.__class__) and \
                isinstance(line[i + 2], current_piece.__class__) and \
                isinstance(line[i + 3], current_piece.__class__) and \
                isinstance(line[i + 4], opponent_type):
                    to_capture.update({(index, j) if is_row else (j, index) for j in range(i, i + 4)})
                    i += 4
                    continue

                i += 1
            return to_capture

        captured_positions = set()

        # Check and capture along rows
        for row in range(ROWS):
            captured_positions.update(capture_line(self.board[row], True, row))

        # Check and capture along columns
        for col in range(COLS):
            column = [self.board[row][col] for row in range(ROWS)]
            captured_positions.update(capture_line(column, False, col))

        # Remove captured pieces
        for row, col in captured_positions:
            if 0 <= row < ROWS and 0 <= col < COLS:  # Ensure indices are within bounds
                if self.board[row][col]:  # Ensure the cell is not empty
                    self.board[row][col] = 0
                    if isinstance(self.board[row][col], TrianglePiece):
                        self.triangle_left -= 1
                    elif isinstance(self.board[row][col], CirclePiece):
                        self.circle_left -= 1


# Minimax Algorithm
def minimax(board, depth, maximizing_player, game):
    if depth == 0 or board.triangle_left == 0 or board.circle_left == 0:
        return board.triangle_left - board.circle_left, None

    best_move = None
    if maximizing_player:
        max_eval = -math.inf
        for piece, moves in get_all_moves(board, TrianglePiece, game):
            for move in moves:
                # Move'u tuple formatına zorla
                if isinstance(move, int):
                    move = (move, 0)  # Varsayılan bir sütun değeri ekle, gerekirse ayarla
                elif not isinstance(move, tuple):
                    print(f"Invalid move format in minimax. Expected tuple, got {type(move)}: {move}")
                    continue

                temp_board = deepcopy(board)
                temp_board.move(piece, *move)
                temp_board.check_and_capture()
                eval, _ = minimax(temp_board, depth - 1, False, game)
                if eval > max_eval:
                    max_eval = eval
                    best_move = (piece, move)
        return max_eval, best_move
    else:
        min_eval = math.inf
        for piece, moves in get_all_moves(board, CirclePiece, game):
            for move in moves:
                # Move'u tuple formatına zorla
                if isinstance(move, int):
                    move = (move, 0)  # Varsayılan bir sütun değeri ekle, gerekirse ayarla
                elif not isinstance(move, tuple):
                    print(f"Invalid move format in minimax. Expected tuple, got {type(move)}: {move}")
                    continue

                temp_board = deepcopy(board)
                temp_board.move(piece, *move)
                temp_board.check_and_capture()
                eval, _ = minimax(temp_board, depth - 1, True, game)
                if eval < min_eval:
                    min_eval = eval
                    best_move = (piece, move)
        return min_eval, best_move



def get_all_moves(board, piece_type, game):
    all_moves = []
    for row in range(ROWS):
        for col in range(COLS):
            piece = board.board[row][col]
            if isinstance(piece, piece_type):
                valid_moves = board.get_valid_moves(piece)
                for move in valid_moves.keys():
                    all_moves.append((piece, move))
    return all_moves


# Game Class
class Game:
    def __init__(self, win):
        self.board = Board()
        self.win = win
        self.turn = TrianglePiece
        self.selected = None
        self.moves_left = 2  # AI için iki hamle hakkı
        self.update()
        self.valid_moves = []
        if self.turn == TrianglePiece:
            self.ai_move()

    def update(self):
        self.board.draw(self.win)
        if self.selected:
            self.board.draw_valid_moves(self.win, self.valid_moves)
        pygame.display.update()

    def select(self, row, col):
        # Boundary check to prevent out-of-range errors
        if row < 0 or row >= ROWS or col < 0 or col >= COLS:
            print("Invalid selection. Click within the board.")
            return

        piece = self.board.board[row][col]
        if self.selected:
            result = self._move(row, col)
            if result:
                return
            self.selected = None

        if piece and isinstance(piece, self.turn):
            if self.move_count == 1 and self.first_piece == piece:
                print("You must select a different piece for the second move.")
                return
            self.selected = piece
            self.valid_moves = list(self.board.get_valid_moves(piece).keys())


    def _move(self, row, col):
        """
        Move the selected piece and handle turn logic.
        """
        if self.selected and (row, col) in self.valid_moves:
            self.board.move(self.selected, row, col)
            self.board.check_and_capture()  # Ensure this method exists in Board
            self.move_count += 1

            # Get the remaining pieces for the current player
            remaining_pieces = (
                self.board.triangle_left if self.turn == TrianglePiece else self.board.circle_left
            )

            if remaining_pieces > 1 and self.move_count == 2:
                self.change_turn()
            elif remaining_pieces == 1 and self.move_count == 1:
                self.change_turn()
            elif self.move_count == 1:
                self.first_piece = self.selected

            self.selected = None
            self.valid_moves = []
            return True
        return False


    def ai_move(self):
        if self.turn == TrianglePiece and self.moves_left > 0:
            _, best_move = minimax(self.board, 2, True, self)
            if best_move:
                piece, move = best_move
                self.board.move(piece, move[0], move[1])
                self.board.check_and_capture()
                self.update()  # Tahtayı her hamle sonrasında güncelle
                self.moves_left -= 1  # Bir hamle yapıldı, kalan hamle sayısını azalt
                pygame.time.delay(500)  # Hamleler arasında görsel bir gecikme ekle
                if self.moves_left > 0:
                    self.ai_move()  # AI ikinci hamlesini yapmalı
                else:
                    self.change_turn()  # Sıra insan oyuncuya geçer




    def change_turn(self):
        if self.turn == TrianglePiece and self.moves_left > 0:
            self.moves_left -= 1  # AI'nin ikinci hamlesine izin ver
            if self.moves_left == 0:
                self.turn = CirclePiece
                self.moves_left = 2  # İnsan sırasından sonra AI için sıfırla
        elif self.turn == CirclePiece:
            self.turn = TrianglePiece
            self.moves_left = 2


# Main Loop
def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Strategic Game with AI")
    game = Game(win)
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
                game.select(row, col)

        game.update()

    pygame.quit()


if __name__ == "__main__":
    main()
