import sac.State;
import sac.StateFunction;

public class manhattan extends StateFunction
{
    public double calculate(State s)
    {
        if(s instanceof puzzle)
        {
            puzzle ss = (puzzle) s;
            return ss.manhattanDistance();
        }
        else return Double.NaN;
    }
}
