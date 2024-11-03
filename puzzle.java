import sac.*;
import sac.graph.GraphState;
import sac.graph.GraphStateImpl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class puzzle extends GraphStateImpl {

    public static byte[][] losujTablice(int n)
    {
        byte[][] tablica = new byte[n][n];
        List<Byte> liczby = new ArrayList<>();

        // Wypełnienie listy liczbami od 0 do (n^2)-1
        for (byte i = 0; i < n * n; i++) {
            liczby.add(i);
        }
        // Losowe wymieszanie liczb
        Collections.shuffle(liczby);
        // Przypisanie wymieszanych liczb do tablicy 2D
        int indeks = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                tablica[i][j] = liczby.get(indeks++);
            }
        }
        return tablica;
    }



    private byte [][] tablica;
    private int n;
    private int wierkafelek,kolkafelek;

    puzzle(int n)
    {
        tablica = losujTablice(n);
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(tablica[i][j]==0)
                {
                    wierkafelek = i;
                    kolkafelek = j;
                }
            }
        }

        this.n = n;
    }

    @Override
    public List<GraphState> generateChildren()
    {
        List<GraphState> lista = new ArrayList<>();
        puzzle gora = wgore(wierkafelek + 1, kolkafelek);
        if (gora != this) lista.add(gora);

        puzzle dol = wdol(wierkafelek - 1, kolkafelek);
        if (dol != this) lista.add(dol);

        puzzle prawo = wprawo(wierkafelek, kolkafelek - 1);
        if (prawo != this) lista.add(prawo);

        puzzle lewo = wlewo(wierkafelek, kolkafelek + 1);
        if (lewo != this) lista.add(lewo);
        return lista;
    }

    public puzzle(puzzle p)
    {
        this.n = p.n;
        this.wierkafelek = p.wierkafelek;
        this.kolkafelek = p.kolkafelek;
        this.tablica = new byte[n][n];
        for (int i = 0; i < p.tablica.length; i++)
        {
            System.arraycopy(p.tablica[i], 0, this.tablica[i], 0, p.tablica[i].length);
        }
    }

    private puzzle wgore(int w, int k) {
        // Sprawdzenie, czy ruch w górę wykracza poza granice planszy
        if (wierkafelek >= n - 1) return this;  // Ruch nielegalny, nie możemy iść w górę
        puzzle p = new puzzle(this);
        // Zamiana kafelków
        p.tablica[wierkafelek][kolkafelek] = p.tablica[wierkafelek + 1][kolkafelek];
        p.tablica[wierkafelek + 1][kolkafelek] = 0;  // Ustawienie nowego pustego kafelka
        p.wierkafelek++;  // Aktualizacja pozycji pustego kafelka
        return p;
    }

    private puzzle wdol(int w, int k) {
        // Sprawdzenie, czy ruch w dół wykracza poza granice planszy
        if (wierkafelek <= 0) return this;  // Ruch nielegalny, nie możemy iść w dół
        puzzle p = new puzzle(this);
        // Zamiana kafelków
        p.tablica[wierkafelek][kolkafelek] = p.tablica[wierkafelek - 1][kolkafelek];
        p.tablica[wierkafelek - 1][kolkafelek] = 0;  // Ustawienie nowego pustego kafelka
        p.wierkafelek--;  // Aktualizacja pozycji pustego kafelka
        return p;
    }

    private puzzle wprawo(int w, int k) {
        // Sprawdzenie, czy ruch w prawo wykracza poza granice planszy
        if (kolkafelek <= 0) return this;  // Ruch nielegalny, nie możemy iść w lewo
        puzzle p = new puzzle(this);
        // Zamiana kafelków
        p.tablica[wierkafelek][kolkafelek] = p.tablica[wierkafelek][kolkafelek - 1];
        p.tablica[wierkafelek][kolkafelek - 1] = 0;  // Ustawienie nowego pustego kafelka
        p.kolkafelek--;  // Aktualizacja pozycji pustego kafelka
        return p;
    }

    private puzzle wlewo(int w, int k) {
        // Sprawdzenie, czy ruch w lewo wykracza poza granice planszy
        if (kolkafelek >= n - 1) return this;  // Ruch nielegalny, nie możemy iść w prawo
        puzzle p = new puzzle(this);
        // Zamiana kafelków
        p.tablica[wierkafelek][kolkafelek] = p.tablica[wierkafelek][kolkafelek + 1];
        p.tablica[wierkafelek][kolkafelek + 1] = 0;  // Ustawienie nowego pustego kafelka
        p.kolkafelek++;  // Aktualizacja pozycji pustego kafelka
        return p;
    }



    @Override
    public boolean isSolution()
    {
        int actual=0;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(tablica[i][j] != actual) return false;
                actual++;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {

        return this.toString().hashCode();
    }

    @Override
    public String toString()
    {
        String s = "";
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                s+=tablica[i][j]+" ";
            }
            s+="\n";
        }
        return s;
    }


    int misplacedtiles()
    {
        int pola = 0;
        int licznik = 0;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                if( (tablica[i][j]!=0) && tablica[i][j]!=licznik ) pola++;
                licznik++;
            }
        }

        return pola;
    }


    int manhattanDistance()
    {
        int suma = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (tablica[i][j] != 0)
                {
                    int value = tablica[i][j];
                    int targetRow = value / n;
                    int targetCol = value % n;
                    suma += Math.abs(i - targetRow) + Math.abs(j - targetCol);
                }
            }
        }
        return suma;
    }




}
