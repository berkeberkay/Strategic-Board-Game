import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
import time

from pieces import TrianglePiece, CirclePiece

transposition_table = {}


def board_to_hash(board):
    return tuple(tuple(str(cell) if cell else '0' for cell in row) for row in board.board)


def simulate_move(board, piece, move_pos):
    """
    Orijinal mantığınızda 'board.move' + 'board.check_and_capture' işlemlerini
    kopyalayarak uyguluyor.
    """
    new_board = deepcopy(board)
    p = new_board.board[piece.row][piece.col]
    new_board.move(p, move_pos[0], move_pos[1])
    new_board.check_and_capture()
    return new_board


@lru_cache(maxsize=10000)
def cached_evaluate(board_hash):
    return 0


def mobility(board, piece):
    valid_moves = board.get_valid_moves(piece)
    return len(valid_moves)


def is_threatened_after_move(board, piece):
    """
    Sizin 'board.is_threatened_after_move(piece)' fonksiyonunuzu çağırıyoruz.
    """
    if piece is None:
        return False
    return board.is_threatened_after_move(piece)


###############################################################################
# Geliştirilmiş evaluate: 
# 1) Taş farkı 
# 2) Merkez bonusu 
# 3) Kill potential 
# 4) Güvenli mi 
# 5) Mobilite
###############################################################################
def evaluate(board):
    # 1) Taş farkı
    score = (board.triangle_left - board.circle_left) * 200

    # Merkez hesabı yaparken board'un ortası (3,3) etrafı vb.
    # Örnek olarak "distance to (3,3)".
    # 7x7 board olduğunu varsayıyoruz (ROWS=7, COLS=7).
    center_r = board.ROWS // 2
    center_c = board.COLS // 2

    def position_score(r, c):
        # Merkeze yakınlık
        dist = abs(r - center_r) + abs(c - center_c)
        # Örnek: distance 0 -> +20, distance 1 -> +15, distance 2 -> +10, distance 3 -> +5, ...
        # Sıfıra düşmezse negatif olmayacak. (İstediğiniz gibi ince ayar yapabilirsiniz.)
        bonus = max(5 * (3 - dist), 0)
        # Kenar/köşe cezasını koruyoruz (örnek)
        corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        if (r, c) in corners:
            bonus -= 5
        elif r in [0, 6] or c in [0, 6]:
            bonus -= 2
        return bonus

    def kill_potential(piece, r, c):
        opponent_type = CirclePiece if isinstance(piece, TrianglePiece) else TrianglePiece
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        kill_score = 0
        for drow, dcol in directions:
            adj_row, adj_col = r + drow, c + dcol
            jump_row, jump_col = r + 2*drow, c + 2*dcol
            if (
                0 <= adj_row < board.ROWS and 0 <= adj_col < board.COLS
                and isinstance(board.board[adj_row][adj_col], opponent_type)
                and 0 <= jump_row < board.ROWS and 0 <= jump_col < board.COLS
                and board.board[jump_row][jump_col] == 0
            ):
                kill_score += 60
        return kill_score

    def is_safe(piece, r, c):
        opponent_type = CirclePiece if isinstance(piece, TrianglePiece) else TrianglePiece
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for drow, dcol in directions:
            adj_row, adj_col = r + drow, c + dcol
            jump_row, jump_col = r - drow, c - dcol
            if (
                0 <= adj_row < board.ROWS and 0 <= adj_col < board.COLS
                and isinstance(board.board[adj_row][adj_col], opponent_type)
            ):
                if (
                    0 <= jump_row < board.ROWS and 0 <= jump_col < board.COLS
                    and board.board[jump_row][jump_col] == 0
                ):
                    return False
        return True

    for r in range(board.ROWS):
        for c in range(board.COLS):
            piece = board.board[r][c]
            if piece:
                pos_score = position_score(r, c)
                kill_score = kill_potential(piece, r, c)
                safe_bonus = 10 if is_safe(piece, r, c) else -20
                move_bonus = mobility(board, piece) * 2

                piece_eval = pos_score + kill_score + safe_bonus + move_bonus
                if isinstance(piece, TrianglePiece):
                    score += piece_eval
                else:
                    score -= piece_eval

    return score


###############################################################################
# SCORE_MOVE İyileştirme:
# "move_priority" benzeri bir mantık + (merkez, capture, güvenlik).
###############################################################################
def score_move(board, piece, move_pos):
    # Mevcut taş sayıları
    tri_count = board.triangle_left
    cir_count = board.circle_left

    temp_board = simulate_move(board, piece, move_pos)
    base_eval = evaluate(temp_board)

    # Rakip öldürme bonusu & tehdit cezası
    old_circ = cir_count
    new_circ = temp_board.circle_left

    capture_bonus = 0
    threatened_penalty = 0

    # Yakaladığımız parça var mı?
    if isinstance(piece, TrianglePiece):
        if old_circ - new_circ > 0:  # Rakip Circle taştan en az biri öldü
            new_piece = temp_board.board[move_pos[0]][move_pos[1]]
            if new_piece and not is_threatened_after_move(temp_board, new_piece):
                capture_bonus = 999999

    # Hamle sonrası tehdit
    new_piece_2 = temp_board.board[move_pos[0]][move_pos[1]]
    if new_piece_2 and is_threatened_after_move(temp_board, new_piece_2):
        threatened_penalty = -50000

    # Taş sayısı farkına göre risk ayarı:
    # AI (TrianglePiece) > rakip => riskten kaç
    # AI == rakip => tam güvenli
    # AI < rakip  => fırsat varsa capture ama "threat" yine ceza
    tri_count = board.triangle_left
    cir_count = board.circle_left

    if tri_count > cir_count:
        # önde => tehdit cezasını artır
        if threatened_penalty < 0:
            threatened_penalty *= 2
    elif tri_count == cir_count:
        # eşit => tamamen güvenli oyna
        if threatened_penalty < 0:
            threatened_penalty *= 2

    return base_eval + capture_bonus + threatened_penalty


def get_all_moves(board, player_class, excluded_piece=None):
    moves = []
    for row in range(board.ROWS):
        for col in range(board.COLS):
            piece = board.board[row][col]
            if piece and isinstance(piece, player_class):
                if excluded_piece is not None and piece == excluded_piece:
                    continue
                valid_moves = board.get_valid_moves(piece)
                for move_pos in valid_moves.keys():
                    moves.append((piece, move_pos))
    return moves

def can_opponent_capture_in_two_moves(self, piece):
    """
    Rakibin 2 hamle (aynı turda) içinde bu 'piece'i yakalayıp yakalayamayacağını kontrol eder.
    'piece' in konumu (piece.row, piece.col).
    Dönüş: True => Bu taş 2 hamlede rakip tarafından öldürülebilir.
           False => Güvende (en azından 2 hamlede yakalanamıyor).
    """

    if not piece:
        return False

    # Eğer piece board'da yoksa zaten yakalanmış sayılır
    if self.board[piece.row][piece.col] != piece:
        return True  # yoksa => tehdide girdi

    # Rakip:
    opponent_class = CirclePiece if isinstance(piece, TrianglePiece) else TrianglePiece

    # 1) Rakibin 1. hamlesindeki tüm olasılıkları dolaş:
    first_moves = self.get_all_opponent_moves(opponent_class)  # bir sonraki fonksiyonda tanımlayacağız

    # Her hamlede simulate_move + check_and_capture
    for (opp_piece_1, move_pos_1) in first_moves:
        # Simülasyon
        new_board_1 = deepcopy(self)  # Board'ı kopyala
        # Opp_piece_1 row,col => simulate
        p1 = new_board_1.board[opp_piece_1.row][opp_piece_1.col]
        new_board_1.move(p1, move_pos_1[0], move_pos_1[1])
        new_board_1.check_and_capture()

        # Bu noktada, eğer piece ölmüşse => True
        if new_board_1.board[piece.row][piece.col] != piece:
            return True

        # 2) Rakibin 2. hamlesi (aynı turdaki ikinci hamle):
        second_moves = new_board_1.get_all_opponent_moves(opponent_class)
        for (opp_piece_2, move_pos_2) in second_moves:
            new_board_2 = deepcopy(new_board_1)
            p2 = new_board_2.board[opp_piece_2.row][opp_piece_2.col]
            new_board_2.move(p2, move_pos_2[0], move_pos_2[1])
            new_board_2.check_and_capture()

            # Tekrar bak, piece hala hayatta mı?
            if new_board_2.board[piece.row][piece.col] != piece:
                return True  # yakalandı => tehdit var

    return False  # tüm dallanmalarda hayattaysa => tehdit yok


def game_over(board, game):
    tri = board.triangle_left
    cir = board.circle_left
    if tri == 0 or cir == 0:
        return True
    if tri == 1 and cir == 1:
        return True
    return False


def minimax(board, depth, alpha, beta, maximizing_player, game, excluded_piece=None):
    board_hash = board_to_hash(board)
    if (board_hash, depth, maximizing_player) in transposition_table:
        entry = transposition_table[(board_hash, depth, maximizing_player)]
        if entry['stored_depth'] >= depth:
            if entry['flag'] == 'EXACT':
                return entry['score'], entry['best_move']
            elif entry['flag'] == 'LOWER' and entry['score'] > alpha:
                alpha = entry['score']
            elif entry['flag'] == 'UPPER' and entry['score'] < beta:
                beta = entry['score']
            if alpha >= beta:
                return entry['score'], entry['best_move']

    # Terminal veya derinlik sonu
    if depth == 0 or game_over(board, game) or game.total_moves >= 50:
        return evaluate(board), None

    player_class = game.TrianglePiece if maximizing_player else game.CirclePiece
    moves = get_all_moves(board, player_class, excluded_piece)
    if not moves:
        return evaluate(board), None

    best_move = None
    if maximizing_player:
        value = float('-inf')
        for piece, move_pos in moves:
            new_board = simulate_move(board, piece, move_pos)
            new_score, _ = minimax(new_board, depth - 1, alpha, beta, False, game)
            if new_score > value:
                value = new_score
                best_move = (piece.row, piece.col, move_pos[0], move_pos[1])
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = float('inf')
        for piece, move_pos in moves:
            new_board = simulate_move(board, piece, move_pos)
            new_score, _ = minimax(new_board, depth - 1, alpha, beta, True, game)
            if new_score < value:
                value = new_score
                best_move = (piece.row, piece.col, move_pos[0], move_pos[1])
            beta = min(beta, value)
            if alpha >= beta:
                break

    flag = 'EXACT'
    if value <= alpha:
        flag = 'UPPER'
    elif value >= beta:
        flag = 'LOWER'

    transposition_table[(board_hash, depth, maximizing_player)] = {
        'score': value,
        'flag': flag,
        'alpha': alpha,
        'beta': beta,
        'best_move': best_move,
        'stored_depth': depth
    }
    return value, best_move


def get_ai_move(game, depth=5, time_limit=20, excluded_piece=None):
    start_time = time.time()
    best_move_found = None

    for d in range(1, depth + 1):
        if time.time() - start_time > time_limit:
            break

        moves = get_all_moves(game.board, game.TrianglePiece, excluded_piece)
        if not moves:
            break

        # "score_move" ile hamle önceliği 
        scored_moves = []
        for piece, move_pos in moves:
            sc = score_move(game.board, piece, move_pos)
            scored_moves.append((sc, piece, move_pos))

        # En yüksek skor önce
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        # Paralelde bir kısım hamleyi deneyelim
        top_moves = scored_moves[:8]

        futures = []
        best_eval = float('-inf')
        local_best_move = None

        with ThreadPoolExecutor(max_workers=4) as executor:
            for _, piece, move_pos in top_moves:
                new_board = simulate_move(game.board, piece, move_pos)
                future = executor.submit(
                    minimax, new_board, d-1, float('-inf'), float('inf'), False, game
                )
                futures.append((future, piece, move_pos)) 

            for fut, piece, move_pos in futures:
                val, _ = fut.result()
                if val > best_eval:
                    best_eval = val
                    local_best_move = (piece.row, piece.col, move_pos[0], move_pos[1])

        if local_best_move is not None:
            best_move_found = local_best_move

    return best_move_found
