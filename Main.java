import sac.game.GameState;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {

    public static void expand(GameState s, int d) {
        long[] v = new long[d];
        expand(s, v, 0);
        // wypisać tablicę
        for (long element: v) System.out.println(element);

    }
    public static void expand(GameState s, long[] v, int d) {
        if (d >= v.length) return;
        for (GameState t : s.generateChildren()) {
            v[d]++;
            expand(t, v, d + 1);
        }
    }

    public static void main(String[] args)
    {
        Mlynek m = new Mlynek();
        expand(m,5);



    }
}