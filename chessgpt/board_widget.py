"""Interactive chess board widget for Jupyter notebooks using anywidget."""

import json
import chess
import anywidget
import traitlets


_CSS = """
.chess-container {
    display: inline-block;
    font-family: system-ui, -apple-system, sans-serif;
    user-select: none;
    -webkit-user-select: none;
}
.chess-board {
    display: grid;
    grid-template-columns: repeat(8, 50px);
    grid-template-rows: repeat(8, 50px);
    border: 2px solid #333;
    position: relative;
}
.chess-square {
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 40px;
    line-height: 1;
    cursor: pointer;
    position: relative;
    box-sizing: border-box;
}
/* White pieces: white fill with dark outline */
.chess-square .piece-white {
    color: #fff;
    text-shadow:
        -1px -1px 0 #333,
         1px -1px 0 #333,
        -1px  1px 0 #333,
         1px  1px 0 #333,
         0   -1px 0 #333,
         0    1px 0 #333,
        -1px  0   0 #333,
         1px  0   0 #333;
}
/* Black pieces: dark fill with subtle outline */
.chess-square .piece-black {
    color: #333;
    text-shadow:
        -1px -1px 0 #666,
         1px -1px 0 #666,
        -1px  1px 0 #666,
         1px  1px 0 #666;
}
.chess-square.light { background: #f0d9b5; }
.chess-square.dark { background: #b58863; }
.chess-square.selected { background: #ffff66 !important; }
.chess-square.last-move { background: #cdd16a !important; }
.chess-square.check {
    background: radial-gradient(ellipse at center, #ff0000 0%, #e60000 25%, rgba(200,0,0,0.5) 89%, rgba(0,0,0,0) 100%) !important;
}
/* Legal move dot (empty square) */
.chess-square.legal-target::after {
    content: '';
    position: absolute;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.25);
}
/* Legal move ring (capture on occupied square) */
.chess-square.legal-capture::after {
    content: '';
    position: absolute;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    border: 3px solid rgba(0, 0, 0, 0.25);
    box-sizing: border-box;
}
/* Coordinates */
.chess-board-wrapper {
    display: inline-grid;
    grid-template-columns: 20px 400px;
    grid-template-rows: 400px 20px;
    gap: 0;
}
.rank-labels {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    align-items: center;
    font-size: 11px;
    color: #666;
    padding-right: 2px;
}
.file-labels {
    display: flex;
    flex-direction: row;
    justify-content: space-around;
    align-items: center;
    font-size: 11px;
    color: #666;
    padding-top: 2px;
    grid-column: 2;
}
/* Status text */
.chess-status {
    margin-top: 8px;
    font-size: 14px;
    color: #333;
    min-height: 20px;
}
/* Move history */
.chess-history {
    margin-top: 8px;
    font-size: 13px;
    color: #555;
    max-height: 120px;
    overflow-y: auto;
    line-height: 1.6;
}
/* Promotion dialog */
.promotion-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
}
.promotion-choices {
    display: flex;
    flex-direction: column;
    background: white;
    border-radius: 6px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    overflow: hidden;
}
.promotion-choice {
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 42px;
    cursor: pointer;
    transition: background 0.1s;
}
.promotion-choice:hover {
    background: #e0e0e0;
}
"""

_ESM = """
// Use filled glyphs for all pieces — white vs black distinguished by CSS class
const PIECE_GLYPH = {
    'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♟',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
};

const FILES = ['a','b','c','d','e','f','g','h'];
const RANKS = ['1','2','3','4','5','6','7','8'];

function parseFen(fen) {
    // Returns a 8x8 array [rank0=rank8..rank7=rank1][file a..h]
    const board = Array.from({length: 8}, () => Array(8).fill(null));
    const ranks = fen.split(' ')[0].split('/');
    for (let r = 0; r < 8; r++) {
        let col = 0;
        for (const ch of ranks[r]) {
            if (ch >= '1' && ch <= '8') {
                col += parseInt(ch);
            } else {
                board[r][col] = ch;
                col++;
            }
        }
    }
    return board;
}

function squareToAlg(row, col, flipped) {
    // row/col are visual (0=top-left of rendered board)
    const r = flipped ? row : 7 - row;
    const c = flipped ? 7 - col : col;
    return FILES[c] + RANKS[r];
}

function algToVisual(alg, flipped) {
    const c = FILES.indexOf(alg[0]);
    const r = parseInt(alg[1]) - 1;
    const vRow = flipped ? r : 7 - r;
    const vCol = flipped ? 7 - c : c;
    return [vRow, vCol];
}

function isWhitePiece(p) { return p && p === p.toUpperCase(); }
function isBlackPiece(p) { return p && p === p.toLowerCase(); }

function render({ model, el }) {
    let selectedSquare = null;  // algebraic notation e.g. "e2"
    let legalTargets = [];      // [{to: "e4", full: "e2e4"}, ...]
    let promotionPending = null; // {from: "e7", to: "e8"}

    function draw() {
        const fen = model.get('fen');
        const legalMovesJson = model.get('legal_moves');
        const lastMove = model.get('last_move');
        const playerColor = model.get('player_color');
        const statusText = model.get('status_text');
        const historyText = model.get('history_text');
        const flipped = playerColor === 'black';

        const board = parseFen(fen);
        const legalMoves = JSON.parse(legalMovesJson || '[]');

        // Determine if current side is in check
        const fenParts = fen.split(' ');
        const activeColor = fenParts.length > 1 ? fenParts[1] : 'w';

        // Find the king square of the active color for check highlighting
        let kingSquare = null;
        // We get check info from status_text containing "Check" or from model
        const isCheck = statusText.toLowerCase().includes('check') ||
                        statusText.toLowerCase().includes('échec');

        if (isCheck) {
            const kingChar = activeColor === 'w' ? 'K' : 'k';
            for (let r = 0; r < 8; r++) {
                for (let c = 0; c < 8; c++) {
                    if (board[r][c] === kingChar) {
                        const kr = 7 - r;
                        kingSquare = FILES[c] + RANKS[kr];
                    }
                }
            }
        }

        // Parse last move
        let lastFrom = null, lastTo = null;
        if (lastMove && lastMove.length >= 4) {
            lastFrom = lastMove.substring(0, 2);
            lastTo = lastMove.substring(2, 4);
        }

        // Is it the player's turn?
        const isPlayerTurn = (activeColor === 'w' && playerColor === 'white') ||
                             (activeColor === 'b' && playerColor === 'black');

        el.innerHTML = '';
        const container = document.createElement('div');
        container.className = 'chess-container';

        const wrapper = document.createElement('div');
        wrapper.className = 'chess-board-wrapper';

        // Rank labels
        const rankLabels = document.createElement('div');
        rankLabels.className = 'rank-labels';
        for (let i = 0; i < 8; i++) {
            const label = document.createElement('div');
            label.textContent = flipped ? RANKS[i] : RANKS[7 - i];
            rankLabels.appendChild(label);
        }
        wrapper.appendChild(rankLabels);

        // Board
        const boardEl = document.createElement('div');
        boardEl.className = 'chess-board';

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const sq = document.createElement('div');
                const alg = squareToAlg(row, col, flipped);

                // Board row/col in FEN terms
                const [fenRow, fenCol] = (() => {
                    const c = FILES.indexOf(alg[0]);
                    const r = parseInt(alg[1]) - 1;
                    return [7 - r, c];
                })();

                const piece = board[fenRow][fenCol];
                const isLight = (row + col) % 2 === 0;

                sq.className = 'chess-square ' + (isLight ? 'light' : 'dark');

                // Last move highlight
                if (alg === lastFrom || alg === lastTo) {
                    sq.classList.add('last-move');
                }

                // Selected square
                if (alg === selectedSquare) {
                    sq.classList.add('selected');
                }

                // Check highlight
                if (alg === kingSquare) {
                    sq.classList.add('check');
                }

                // Legal target indicators
                const targetInfo = legalTargets.find(t => t.to === alg);
                if (targetInfo) {
                    if (piece) {
                        sq.classList.add('legal-capture');
                    } else {
                        sq.classList.add('legal-target');
                    }
                }

                // Piece
                if (piece) {
                    const span = document.createElement('span');
                    span.className = isWhitePiece(piece) ? 'piece-white' : 'piece-black';
                    span.textContent = PIECE_GLYPH[piece] || '';
                    sq.appendChild(span);
                }

                // Click handler
                sq.addEventListener('click', () => {
                    if (promotionPending) return;
                    if (!isPlayerTurn) return;

                    const target = legalTargets.find(t => t.to === alg);
                    if (target) {
                        // Check if this is a promotion
                        const isPromotion = target.promotions && target.promotions.length > 0;
                        if (isPromotion) {
                            promotionPending = {from: selectedSquare, to: alg, promotions: target.promotions};
                            draw();
                            return;
                        }
                        // Play the move
                        selectedSquare = null;
                        legalTargets = [];
                        model.set('pending_move', target.full);
                        model.save_changes();
                        draw();
                        return;
                    }

                    // Select a piece
                    const isPlayerPiece = (playerColor === 'white' && isWhitePiece(piece)) ||
                                          (playerColor === 'black' && isBlackPiece(piece));
                    if (isPlayerPiece) {
                        selectedSquare = alg;
                        // Find legal moves from this square
                        legalTargets = [];
                        const targetMap = {};
                        for (const m of legalMoves) {
                            if (m.substring(0, 2) === alg) {
                                const to = m.substring(2, 4);
                                const promo = m.length > 4 ? m[4] : null;
                                if (!targetMap[to]) {
                                    targetMap[to] = {to, full: m, promotions: []};
                                }
                                if (promo) {
                                    targetMap[to].promotions.push(promo);
                                    // Keep the queen promotion as default full move
                                    if (promo === 'q') targetMap[to].full = m;
                                }
                            }
                        }
                        legalTargets = Object.values(targetMap);
                        draw();
                    } else {
                        // Deselect
                        selectedSquare = null;
                        legalTargets = [];
                        draw();
                    }
                });

                boardEl.appendChild(sq);
            }
        }

        wrapper.appendChild(boardEl);

        // File labels
        const fileLabels = document.createElement('div');
        fileLabels.className = 'file-labels';
        for (let i = 0; i < 8; i++) {
            const label = document.createElement('div');
            label.textContent = flipped ? FILES[7 - i] : FILES[i];
            fileLabels.appendChild(label);
        }
        wrapper.appendChild(fileLabels);

        container.appendChild(wrapper);

        // Promotion overlay
        if (promotionPending) {
            const overlay = document.createElement('div');
            overlay.className = 'promotion-overlay';
            overlay.style.position = 'absolute';
            overlay.style.top = '0';
            overlay.style.left = '20px';
            overlay.style.width = '400px';
            overlay.style.height = '400px';

            const choices = document.createElement('div');
            choices.className = 'promotion-choices';

            const promoChars = playerColor === 'white'
                ? ['Q', 'R', 'B', 'N']
                : ['q', 'r', 'b', 'n'];
            const promoUci = ['q', 'r', 'b', 'n'];

            for (let i = 0; i < 4; i++) {
                if (!promotionPending.promotions.includes(promoUci[i])) continue;
                const choice = document.createElement('div');
                choice.className = 'promotion-choice';
                const promoSpan = document.createElement('span');
                promoSpan.className = playerColor === 'white' ? 'piece-white' : 'piece-black';
                promoSpan.textContent = PIECE_GLYPH[promoChars[i]];
                choice.appendChild(promoSpan);
                choice.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const moveUci = promotionPending.from + promotionPending.to + promoUci[i];
                    promotionPending = null;
                    selectedSquare = null;
                    legalTargets = [];
                    model.set('pending_move', moveUci);
                    model.save_changes();
                    draw();
                });
                choices.appendChild(choice);
            }

            overlay.appendChild(choices);

            // Click outside to cancel
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    promotionPending = null;
                    selectedSquare = null;
                    legalTargets = [];
                    draw();
                }
            });

            boardEl.style.position = 'relative';
            boardEl.appendChild(overlay);
        }

        // Status
        const statusEl = document.createElement('div');
        statusEl.className = 'chess-status';
        statusEl.textContent = statusText;
        container.appendChild(statusEl);

        // Move history
        if (historyText) {
            const historyEl = document.createElement('div');
            historyEl.className = 'chess-history';
            historyEl.innerHTML = historyText;
            container.appendChild(historyEl);
        }

        el.appendChild(container);
    }

    // Listen for model changes
    model.on('change:fen', () => {
        selectedSquare = null;
        legalTargets = [];
        promotionPending = null;
        draw();
    });
    model.on('change:legal_moves', draw);
    model.on('change:last_move', draw);
    model.on('change:status_text', draw);
    model.on('change:history_text', draw);

    draw();
}

export default { render };
"""


class ChessBoardWidget(anywidget.AnyWidget):
    """Interactive chess board widget for Jupyter notebooks.

    Usage:
        board = chess.Board()
        widget = ChessBoardWidget(player_color="white")
        widget.update_position(board)
        display(widget)

        def on_move(move_uci):
            mv = chess.Move.from_uci(move_uci)
            board.push(mv)
            # ... run model inference ...
            widget.update_position(board, last_move=mv)

        widget.on_move(on_move)
    """

    _esm = traitlets.Unicode(_ESM).tag(sync=True)
    _css = traitlets.Unicode(_CSS).tag(sync=True)

    fen = traitlets.Unicode(chess.STARTING_FEN).tag(sync=True)
    legal_moves = traitlets.Unicode("[]").tag(sync=True)
    last_move = traitlets.Unicode("").tag(sync=True)
    player_color = traitlets.Unicode("white").tag(sync=True)
    pending_move = traitlets.Unicode("").tag(sync=True)
    status_text = traitlets.Unicode("").tag(sync=True)
    history_text = traitlets.Unicode("").tag(sync=True)

    def __init__(self, player_color="white", **kwargs):
        super().__init__(player_color=player_color, **kwargs)
        self._move_callbacks = []
        self.observe(self._on_pending_move_change, names=["pending_move"])

    def _on_pending_move_change(self, change):
        move_uci = change["new"]
        if move_uci:
            # Reset pending_move so the same move can be played again in another context
            self.pending_move = ""
            for cb in self._move_callbacks:
                cb(move_uci)

    def on_move(self, callback):
        """Register a callback called when the player makes a move.

        The callback receives the UCI move string (e.g. "e2e4").
        """
        self._move_callbacks.append(callback)

    def update_position(self, board: chess.Board, last_move=None):
        """Update the displayed position from a python-chess Board."""
        self.fen = board.fen()
        moves = [m.uci() for m in board.legal_moves]
        self.legal_moves = json.dumps(moves)
        self.last_move = last_move.uci() if last_move else ""

    def set_status(self, text: str):
        """Update the status message displayed below the board."""
        self.status_text = text

    def set_history(self, html: str):
        """Update the move history displayed below the board."""
        self.history_text = html
