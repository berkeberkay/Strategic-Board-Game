import pygame
from pieces import TrianglePiece, CirclePiece
from AI import get_ai_move
from copy import deepcopy

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 7, 7
SQUARE_SIZE = WIDTH // COLS
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
HIGHLIGHT = (0, 255, 0)
INFO_BAR_HEIGHT = 30

class Board:
    ROWS = ROWS
    COLS = COLS

    def __init__(self):
        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.triangle_left = 4
        self.circle_left = 4
        self.create_board()

    def create_board(self):
        # Başlangıç konumlarınızı buraya yerleştirebilirsiniz.
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
        for p in pieces:
            self.board[p.row][p.col] = 0
            if isinstance(p, TrianglePiece):
                self.triangle_left -= 1
            elif isinstance(p, CirclePiece):
                self.circle_left -= 1

    def get_valid_moves(self, piece):
        moves = {}
        row, col = piece.row, piece.col
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for drow, dcol in directions:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < self.ROWS and 0 <= new_col < self.COLS:
                if self.board[new_row][new_col] == 0:
                    moves[(new_row, new_col)] = []
        return moves

    ###########################################################################
    # capture_row & capture_col: Sıkıştırma bazlı yakalama
    ###########################################################################
    def capture_row(self, row_index):
        line = self.board[row_index]
        to_capture = set()
        n = len(line)
        i = 0
        while i < n:
            if not line[i]:
                i += 1
                continue

            current_piece = line[i]
            opponent_type = CirclePiece if isinstance(current_piece, TrianglePiece) else TrianglePiece

            if i == 0 and i + 4 < n and isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], current_piece.__class__) and \
               isinstance(line[i + 3], current_piece.__class__) and \
               isinstance(line[i + 4], opponent_type):
                to_capture.update({(row_index, j) for j in range(i, i + 4)})
                i += 4
                continue

            if i == 0 and i + 3 < n and isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], current_piece.__class__) and \
               isinstance(line[i + 3], opponent_type):
                to_capture.update({(row_index, j) for j in range(i, i + 3)})
                i += 3
                continue

            if i == 0 and i + 2 < n and isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], opponent_type):
                to_capture.update({(row_index, j) for j in range(i, i + 2)})
                i += 2
                continue

            if i == 0 and i + 1 < n and isinstance(line[i + 1], opponent_type):
                to_capture.add((row_index, i))
                i += 1
                continue

            if i == n - 1 and i - 1 >= 0 and isinstance(line[i - 1], opponent_type):
                to_capture.add((row_index, i))

            if i == n - 1 and i - 2 >= 0 and isinstance(line[i - 1], current_piece.__class__) and \
               isinstance(line[i - 2], opponent_type):
                to_capture.update({(row_index, j) for j in range(i - 2, i + 1)})

            if i == n - 1 and i - 3 >= 0 and isinstance(line[i - 1], current_piece.__class__) and \
               isinstance(line[i - 2], current_piece.__class__) and \
               isinstance(line[i - 3], opponent_type):
                to_capture.update({(row_index, j) for j in range(i - 3, i + 1)})

            if 0 < i < n - 1 and isinstance(line[i - 1], opponent_type) and \
               isinstance(line[i + 1], opponent_type):
                to_capture.add((row_index, i))
                i += 1
                continue

            if 0 < i < n - 2 and isinstance(line[i - 1], opponent_type) and \
               isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], opponent_type):
                to_capture.update({(row_index, j) for j in range(i, i + 2)})
                i += 2
                continue

            if 0 < i < n - 3 and isinstance(line[i - 1], opponent_type) and \
               isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], current_piece.__class__) and \
               isinstance(line[i + 3], opponent_type):
                to_capture.update({(row_index, j) for j in range(i, i + 3)})
                i += 3
                continue

            if 0 < i < n - 4 and isinstance(line[i - 1], opponent_type) and \
               isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], current_piece.__class__) and \
               isinstance(line[i + 3], current_piece.__class__) and \
               isinstance(line[i + 4], opponent_type):
                to_capture.update({(row_index, j) for j in range(i, i + 4)})
                i += 4
                continue

            i += 1
        return to_capture

    def capture_col(self, col_index):
        line = [self.board[r][col_index] for r in range(self.ROWS)]
        to_capture = set()
        n = len(line)
        i = 0
        while i < n:
            if not line[i]:
                i += 1
                continue

            current_piece = line[i]
            opponent_type = CirclePiece if isinstance(current_piece, TrianglePiece) else TrianglePiece

            if i == 0 and i + 4 < n and isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], current_piece.__class__) and \
               isinstance(line[i + 3], current_piece.__class__) and \
               isinstance(line[i + 4], opponent_type):
                to_capture.update({(j, col_index) for j in range(i, i + 4)})
                i += 4
                continue

            if i == 0 and i + 3 < n and isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], current_piece.__class__) and \
               isinstance(line[i + 3], opponent_type):
                to_capture.update({(j, col_index) for j in range(i, i + 3)})
                i += 3
                continue

            if i == 0 and i + 2 < n and isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], opponent_type):
                to_capture.update({(j, col_index) for j in range(i, i + 2)})
                i += 2
                continue

            if i == 0 and i + 1 < n and isinstance(line[i + 1], opponent_type):
                to_capture.add((i, col_index))
                i += 1
                continue

            if i == n - 1 and i - 1 >= 0 and isinstance(line[i - 1], opponent_type):
                to_capture.add((i, col_index))

            if i == n - 1 and i - 2 >= 0 and isinstance(line[i - 1], current_piece.__class__) and \
               isinstance(line[i - 2], opponent_type):
                to_capture.update({(j, col_index) for j in range(i - 2, i + 1)})

            if i == n - 1 and i - 3 >= 0 and isinstance(line[i - 1], current_piece.__class__) and \
               isinstance(line[i - 2], current_piece.__class__) and \
               isinstance(line[i - 3], opponent_type):
                to_capture.update({(j, col_index) for j in range(i - 3, i + 1)})

            if 0 < i < n - 1 and isinstance(line[i - 1], opponent_type) and \
               isinstance(line[i + 1], opponent_type):
                to_capture.add((i, col_index))
                i += 1
                continue

            if 0 < i < n - 2 and isinstance(line[i - 1], opponent_type) and \
               isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], opponent_type):
                to_capture.update({(j, col_index) for j in range(i, i + 2)})
                i += 2
                continue

            if 0 < i < n - 3 and isinstance(line[i - 1], opponent_type) and \
               isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], current_piece.__class__) and \
               isinstance(line[i + 3], opponent_type):
                to_capture.update({(j, col_index) for j in range(i, i + 3)})
                i += 3
                continue

            if 0 < i < n - 4 and isinstance(line[i - 1], opponent_type) and \
               isinstance(line[i + 1], current_piece.__class__) and \
               isinstance(line[i + 2], current_piece.__class__) and \
               isinstance(line[i + 3], current_piece.__class__) and \
               isinstance(line[i + 4], opponent_type):
                to_capture.update({(j, col_index) for j in range(i, i + 4)})
                i += 4
                continue

            i += 1
        return to_capture

    ###########################################################################
    # get_all_opponent_moves: Rakibin tüm taş-hamle çiftlerini döndürür.
    ###########################################################################
    def get_all_opponent_moves(self, opponent_class):
        moves = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = self.board[r][c]
                if piece and isinstance(piece, opponent_class):
                    valid_moves = self.get_valid_moves(piece)
                    for mv in valid_moves.keys():
                        moves.append((piece, mv))
        return moves

    ###########################################################################
    # can_opponent_capture_in_two_moves: Rakip 2 hamle (tek tur) içinde taşı öldürebilir mi?
    ###########################################################################
    def can_opponent_capture_in_two_moves(self, piece):
        """
        Rakibin 2 hamlede bu piece'i öldürüp öldüremeyeceğini kontrol eder.
        """
        if not piece:
            return False

        # eğer piece board'da değilse zaten ölü
        if self.board[piece.row][piece.col] != piece:
            return True

        opponent_class = CirclePiece if isinstance(piece, TrianglePiece) else TrianglePiece

        # 1. hamle simülasyonu
        first_moves = self.get_all_opponent_moves(opponent_class)
        for (opp_piece_1, move_pos_1) in first_moves:
            temp_board_1 = deepcopy(self)
            p1 = temp_board_1.board[opp_piece_1.row][opp_piece_1.col]
            temp_board_1.move(p1, move_pos_1[0], move_pos_1[1])
            temp_board_1.check_and_capture()

            # öldü mü?
            if temp_board_1.board[piece.row][piece.col] != piece:
                return True

            # 2. hamle simülasyonu
            second_moves = temp_board_1.get_all_opponent_moves(opponent_class)
            for (opp_piece_2, move_pos_2) in second_moves:
                temp_board_2 = deepcopy(temp_board_1)
                p2 = temp_board_2.board[opp_piece_2.row][opp_piece_2.col]
                temp_board_2.move(p2, move_pos_2[0], move_pos_2[1])
                temp_board_2.check_and_capture()

                if temp_board_2.board[piece.row][piece.col] != piece:
                    return True

        return False

    ###########################################################################
    # is_threatened_after_move: Hem anlık sıkıştırma hem de 2 hamle derinliğine rakip kontrolü
    ###########################################################################
    def is_threatened_after_move(self, piece):
        if not piece:
            return False

        r, c = piece.row, piece.col

        # 1) Hemen satır-sütun sıkıştırma
        if (r, c) in self.capture_row(r):
            return True
        if (r, c) in self.capture_col(c):
            return True

        # 2) Rakibin 2 hamlede capture imkanı
        if self.can_opponent_capture_in_two_moves(piece):
            return True

        return False

    ###########################################################################
    # check_and_capture
    ###########################################################################
    def check_and_capture(self):
        captured_positions = set()

        # Rows
        for row in range(self.ROWS):
            row_captures = self.capture_row(row)
            captured_positions.update(row_captures)

        # Columns
        for col in range(self.COLS):
            col_captures = self.capture_col(col)
            captured_positions.update(col_captures)

        for (r, c) in captured_positions:
            if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                if self.board[r][c] != 0:
                    dead_piece = self.board[r][c]
                    self.board[r][c] = 0
                    if isinstance(dead_piece, TrianglePiece):
                        self.triangle_left -= 1
                    elif isinstance(dead_piece, CirclePiece):
                        self.circle_left -= 1


class Game:
    TrianglePiece = TrianglePiece
    CirclePiece = CirclePiece

    def __init__(self, win):
        self.board = Board()
        self.win = win
        self.selected = None
        self.valid_moves = []
        self.turn = TrianglePiece  # AI başlasın
        self.move_count = 0
        self.first_piece = None
        self.total_moves = 0
        self.game_over_flag = False
        self.font = pygame.font.Font(None, 36)

    def draw_game_info(self):
        pygame.draw.rect(self.win, (10, 10, 10), (0, HEIGHT, WIDTH, INFO_BAR_HEIGHT))
        current_turn = "Triangle (AI)" if self.turn == self.TrianglePiece else "Circle (Human)"
        info_text = [f"Turn: {current_turn}  Total Moves: {self.total_moves}/50"]
        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.win.blit(text_surface, (20, HEIGHT + 5 + i * 25))

    def update(self):
        self.board.draw(self.win)
        if self.selected:
            self.board.draw_valid_moves(self.win, self.valid_moves)
        self.draw_game_info()
        pygame.display.update()

    def select(self, row, col):
        if self.turn == self.CirclePiece:
            if not (0 <= row < ROWS and 0 <= col < COLS):
                return
            piece = self.board.board[row][col]
            if self.selected:
                if self._move(row, col):
                    return
                self.selected = None
            if piece and isinstance(piece, self.turn):
                if self.move_count == 1 and self.first_piece == piece:
                    print("Aynı tası ikinci hamlede kullanamazsınız!")
                    return
                self.selected = piece
                self.valid_moves = list(self.board.get_valid_moves(piece).keys())

    def _move(self, row, col):
        if self.selected and (row, col) in self.valid_moves:
            self.board.move(self.selected, row, col)
            self.board.check_and_capture()
            self.move_count += 1
            self.total_moves += 1

            remaining_pieces = (
                self.board.triangle_left if self.turn == TrianglePiece else self.board.circle_left
            )
            if (remaining_pieces > 1 and self.move_count == 2):
                self.change_turn()
            elif (remaining_pieces == 1 and self.move_count == 1):
                self.change_turn()    
            elif self.move_count == 1 and remaining_pieces > 1:
                self.first_piece = self.selected

            self.selected = None
            self.valid_moves = []
            return True
        return False

    def change_turn(self):
        self.turn = self.CirclePiece if self.turn == self.TrianglePiece else self.TrianglePiece
        self.move_count = 0
        self.first_piece = None
        print(f"Sira degisti. Su an {self.turn.__name__} oynuyor.")

    def check_game_end_conditions(self):
        tri = self.board.triangle_left
        cir = self.board.circle_left

        if tri == 0 and cir == 0:
            print("Oyun Bitti: Berabere (Her iki taraf da 0 tas).")
            self.game_over_flag = True
        elif tri == 0 and cir > 0:
            print("Oyun Bitti: Human kazandı.")
            self.game_over_flag = True
        elif cir == 0 and tri > 0:
            print("Oyun Bitti: AI kazandı.")
            self.game_over_flag = True
        elif tri == 1 and cir == 1:
            print("Oyun Bitti: Berabere (Her iki taraf 1 tas).")
            self.game_over_flag = True

        if self.total_moves >= 50 and not self.game_over_flag:
            print("50 Hamle Kuralı devreye girdi.")
            if tri == cir:
                print("Oyun Bitti: Berabere (50 hamlede tas sayıları esit).")
            elif tri > cir:
                print("Oyun Bitti: AI kazandı (50 hamlede daha çok tas).")
            else:
                print("Oyun Bitti: Human kazandı (50 hamlede daha çok tas).")
            self.game_over_flag = True

    def ai_make_move(self):
        self.update()
        remaining_pieces = self.board.triangle_left
        moves_to_make = 1 if remaining_pieces == 1 else 2

        excluded_piece = None
        for i in range(moves_to_make):
            if self.total_moves >= 50:
                break
            move = get_ai_move(self, depth=5, time_limit=15, excluded_piece=excluded_piece)
            if move is None:
                break
            start_row, start_col, end_row, end_col = move
            piece = self.board.board[start_row][start_col]
            if i == 0:
                first_piece = piece
            self.board.move(piece, end_row, end_col)
            self.board.check_and_capture()
            self.move_count += 1
            self.total_moves += 1
            if i == 0 and moves_to_make == 2:
                excluded_piece = first_piece

        self.change_turn()

    def game_over(self, winner=None):
        self.win.fill((0, 0, 0))
        
        font = pygame.font.Font(None, 80)
        text = font.render('Game Over!', True, (255, 0, 0))
        text_rect = text.get_rect(center=(self.win.get_width()/2, self.win.get_height()/2 - 40))
        
        result_font = pygame.font.Font(None, 50)
        if self.total_moves >= 50:
            tri = self.board.triangle_left
            cir = self.board.circle_left
            if tri == cir:
                result_text = result_font.render('Berabere!', True, (255, 255, 255))
            elif tri > cir:
                result_text = result_font.render('Triangle (AI) Kazandı!', True, (255, 255, 255))
            else:
                result_text = result_font.render('Circle (İnsan) Kazandı!', True, (255, 255, 255))
        elif winner is None:
            result_text = result_font.render('Berabere!', True, (255, 255, 255))
        else:
            result_text = result_font.render(f'{winner} Kazandı!', True, (255, 255, 255))
        
        result_rect = result_text.get_rect(center=(self.win.get_width()/2, self.win.get_height()/2 + 40))
        self.win.blit(text, text_rect)
        self.win.blit(result_text, result_rect)
        pygame.display.flip()
        pygame.time.wait(2000)
        pygame.quit()

    def main_loop(self):
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            self.check_game_end_conditions()
            if self.game_over_flag:
                self.update()
                pygame.time.wait(2000)
                self.game_over()
                break

            if self.turn == self.TrianglePiece:  # AI
                self.ai_make_move()
                self.update()
            else:  # İnsan
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
                        self.select(row, col)
            self.update()
        pygame.quit()


def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT + INFO_BAR_HEIGHT))
    pygame.display.set_caption("Strategic Game Board")
    game = Game(win)
    game.update()
    game.main_loop()

if __name__ == "__main__":
    main()
