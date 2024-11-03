import sac.graph.BestFirstSearch;
import sac.graph.GraphSearchConfigurator;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {
    public static void main(String[] args)
    {
        puzzle s = new puzzle(3);
        puzzle.setHFunction(new misplacedtiles());
        System.out.println(s);

        GraphSearchConfigurator gsc = new GraphSearchConfigurator();
        gsc.setWantedNumberOfSolutions(Integer.MAX_VALUE);

        BestFirstSearch bfs = new BestFirstSearch();
        bfs.setInitial(s);
        bfs.execute();
        System.out.println(bfs.getSolutions());

    }
}