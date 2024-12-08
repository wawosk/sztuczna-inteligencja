import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import sac.game.GameState;
import sac.game.GameStateImpl;

public class Mlynek extends GameStateImpl {

    //w - stopien zaglebienia,
    //k - indeks dookola, przeciwnie do zegara zaczynając od lewego rogu
    //0 - puste pole, 1 - biały, 2 - czarny
    //biały maksymalizuje, czarny minimalizuje

    static final byte EMPTY = 0;
    static final byte WHITE = 1;
    static final byte BLACK = 2;
    byte[][] board = new byte[3][8];
    byte whitePieces = 9;
    byte blackPieces = 9;
    byte remainingPiecesToPlace = 18;

    public Mlynek()
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 8; j++)
            {
                board[i][j]=0;
            }
        }
    }

    public Mlynek(Mlynek other) {
        for (int i = 0; i < 3; i++) {
            this.board[i] = Arrays.copyOf(other.board[i], 8);
        }
        this.whitePieces = other.whitePieces;
        this.blackPieces = other.blackPieces;
        this.remainingPiecesToPlace = other.remainingPiecesToPlace;
    }

    private boolean isMill(int ring, int position)
    {
        byte color = board[ring][position];
        if (position % 2 == 0) return
        (color == board[ring][(position + 1) % 8] && color == board[ring][(position + 2) % 8]) || (color == board[ring][(position + 6) % 8] && color == board[ring][(position + 7) % 8]);
        else return
        (board[0][position] == board[1][position] && board[1][position] == board[2][position]) || (color == board[ring][(position + 1) % 8] && color == board[ring][(position + 7) % 8]);
    }

    @Override
    public String toString() {
        return "Mlynek{" +
                "board=" + Arrays.deepToString(board) +
                '}';
    }

    public int hashCode() {
        return Arrays.deepHashCode(board);
    }

    boolean isEndgamePhase() {
        return (whitePieces == 3 && maximizingTurnNow) || (blackPieces == 3 && !maximizingTurnNow);
    }
    public List<GameState> generateChildren() {
        List<GameState> children = new ArrayList<>();
        if (isEndgamePhase()) {
            generateEndgameMoves(children);
        } else if (remainingPiecesToPlace > 0) {
            generatePlacementMoves(children);
        } else {
            generateMovementPhaseMoves(children);
        }

        return children;
    }


    private void generatePlacementMoves(List<GameState> children)
    {
        byte playerPiece = maximizingTurnNow ? WHITE : BLACK;

        for (int ring = 0; ring < 3; ring++)
        {
            for (int position = 0; position < 8; position++)
            {
                if (board[ring][position] == EMPTY)
                {
                    Mlynek newState = new Mlynek(this);
                    newState.maximizingTurnNow = !maximizingTurnNow;
                    newState.remainingPiecesToPlace--;

                    newState.board[ring][position] = playerPiece;
                    if (isMill(ring, position))
                    {
                        newState.board[ring][position] = EMPTY;
                        children.addAll(handleMill(ring, position));
                    }
                    else
                    {
                        children.add(newState);
                    }
                }
            }
        }
    }


    private void generateEndgameMoves(List<GameState> children)
    {
        byte playerPiece = maximizingTurnNow ? WHITE : BLACK;

        for (int ring = 0; ring < 3; ring++)
        {
            for (int position = 0; position < 8; position++) {
                if (board[ring][position] == playerPiece) {
                    children.addAll(generateMovesForPiece(ring, position, true));
                }
            }
        }
    }


    private void generateMovementPhaseMoves(List<GameState> children)
    {
        byte playerPiece = maximizingTurnNow ? WHITE : BLACK;
        for (int ring = 0; ring < 3; ring++)
        {
            for (int position = 0; position < 8; position++)
            {
                if (board[ring][position] == playerPiece)
                {
                    children.addAll(generateMovesForPiece(ring, position, false));
                }
            }
        }
    }

    private List<GameState> handleMill(int ring, int position)
    {
        List<GameState> children = new ArrayList<>();
        boolean pieceRemoved = false;

        for (int r = 0; r < 3; r++) {
            for (int pos = 0; pos < 8; pos++) {
                if (removeEnemyPiece(r, pos, ring, position, children)) {
                    pieceRemoved = true;
                }
            }
        }
        if (!pieceRemoved) {
            removeAnyEnemyPiece(ring, position, children);
        }

        return children;
    }

    private boolean removeEnemyPiece(int ring, int position, int targetRing, int targetPosition, List<GameState> children) {
        byte enemyPiece = maximizingTurnNow ? BLACK : WHITE;

        if (board[ring][position] == enemyPiece && !isMill(ring, position)) {

            Mlynek newState = new Mlynek(this);
            newState.maximizingTurnNow = !maximizingTurnNow;
            newState.board[targetRing][targetPosition] = maximizingTurnNow ? WHITE : BLACK;
            newState.board[ring][position] = EMPTY;

            if (maximizingTurnNow) {
                newState.blackPieces--;
            } else {
                newState.whitePieces--;
            }

            children.add(newState);
            return true;
        }

        return false;
    }

    private void removeAnyEnemyPiece(int targetRing, int targetPosition, List<GameState> children)
    {
        byte enemyPiece = maximizingTurnNow ? BLACK : WHITE;

        for (int ring = 0; ring < 3; ring++) {
            for (int position = 0; position < 8; position++) {
                if (board[ring][position] == enemyPiece)
                {
                    Mlynek newState = new Mlynek(this);
                    newState.maximizingTurnNow = !maximizingTurnNow;
                    newState.board[targetRing][targetPosition] = maximizingTurnNow ? WHITE : BLACK;
                    newState.board[ring][position] = EMPTY;

                    if (maximizingTurnNow) {
                        newState.blackPieces--;
                    } else {
                        newState.whitePieces--;
                    }

                    children.add(newState);
                }
            }
        }
    }

    List<GameState> generateMovesForPiece(int ring, int position, boolean canMoveAnywhere) {
        List<GameState> moves = new ArrayList<>();

        if (canMoveAnywhere) {
            for (int r = 0; r < 3; r++) {
                for (int pos = 0; pos < 8; pos++) {
                    addMoveIfValid(r, pos, ring, position, moves);
                }
            }
        } else {
            generateAdjacentMoves(ring, position, moves);
        }

        return moves;
    }

    private void addMoveIfValid(int targetRing, int targetPosition, int currentRing, int currentPosition, List<GameState> moves) {
        byte color = board[currentRing][currentPosition];
        if (board[targetRing][targetPosition] == EMPTY) {
            Mlynek newState = new Mlynek(this);
            newState.maximizingTurnNow = !maximizingTurnNow;
            newState.board[targetRing][targetPosition] = color;
            newState.board[currentRing][currentPosition] = EMPTY;

            if (isMill(targetRing, targetPosition)) {
                newState.board[targetRing][targetPosition] = EMPTY;
                moves.addAll(handleMill(targetRing, targetPosition));
            } else {
                moves.add(newState);
            }
        }
    }

    private void generateAdjacentMoves(int ring, int position, List<GameState> moves) {
        int[] adjacentPositions = { (position + 1) % 8, (position + 7) % 8 };

        for (int adjPosition : adjacentPositions) {
            addMoveIfValid(ring, adjPosition, ring, position, moves);
        }

        if (position % 2 != 0) {
            switch (ring) {
                case 0 -> addMoveIfValid(1, position, ring, position, moves);
                case 1 -> {
                    addMoveIfValid(0, position, ring, position, moves);
                    addMoveIfValid(2, position, ring, position, moves);
                }
                case 2 -> addMoveIfValid(1, position, ring, position, moves);
            }
        }
    }
}

