import sac.State;
import sac.StateFunction;
import sac.game.GameState;

import java.util.ArrayList;
import java.util.List;

public class evaluation extends StateFunction {

    public boolean isAnyMovePossible(Mlynek m)
    {
        if (m.remainingPiecesToPlace > 0) return true;
        else if (m.isEndgamePhase())
        {
            if(m.isMaximizingTurnNow() && m.whitePieces==3) return true;
            if(!m.isMaximizingTurnNow() && m.blackPieces==3) return true;
        }
        else
        {
            List<GameState> moves = null;
            byte playerPiece = m.isMaximizingTurnNow() ? m.WHITE : m.BLACK;

            for (int ring = 0; ring < 3; ring++) {
                for (int position = 0; position < 8; position++) {
                    if (m.board[ring][position] == playerPiece) {
                        moves = m.generateMovesForPiece(ring, position, true);
                        if (!moves.isEmpty()) return true;
                    }
                }
            }
            if(moves.isEmpty()) return false;
        }

        return false; // Brak możliwych ruchów
    }



    public double calculate(State s)
    {
        if(s instanceof Mlynek)
        {
            Mlynek mlynek = (Mlynek) s;
            if (mlynek.whitePieces<3) return Double.MAX_VALUE;
            if (mlynek.blackPieces<3) return Double.MIN_VALUE;
            if(!mlynek.isMaximizingTurnNow() && !isAnyMovePossible(mlynek)) return Double.MAX_VALUE;
            if(mlynek.isMaximizingTurnNow() && !isAnyMovePossible(mlynek)) return Double.MIN_VALUE;

            return mlynek.whitePieces-mlynek.blackPieces    ;

        }
        return Double.NaN;
    }
}
