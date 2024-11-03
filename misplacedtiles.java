import sac.State;
import sac.StateFunction;

public class misplacedtiles extends StateFunction
{
    public double calculate(State s)
    {
        if(s instanceof puzzle)
        {
            puzzle ss = (puzzle) s;
            return ss.misplacedtiles();
        }
        else return Double.NaN;
    }


}
