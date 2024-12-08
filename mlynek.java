import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import sac.*;
import sac.game.GameState;
import sac.game.GameStateImpl;
import sac.graph.GraphState;

public class mlynek extends GameStateImpl {
    byte[][] tablica = new byte[3][8];
    byte bialepionki = 9;
    byte czarnepionki = 9;
    byte dorozmieszczenia = 18;


    //w - stopien zaglebienia,
    //k - indeks dookola, przeciwnie do zegara zaczynając od lewego rogu
    //0 - puste pole, 1 - biały, 2 - czarny
    //biały maksymalizuje, czarny minimalizuje


    boolean czymlynek(int i, int j) {
        byte kolor = tablica[i][j];

        //przypadek parzysty
        if (j % 2 == 0) {
            //POPRAWNIE
            if (kolor == tablica[i][(j + 1) % 8] && kolor == tablica[i][(j + 2) % 8]) return true;
            if (kolor == tablica[i][(j + 6) % 8] && kolor == tablica[i][(j + 7) % 8]) return true;
        }
        //nieparzysty
        else {
            //POPRAWNIE
            if (tablica[0][j] == tablica[1][j] && tablica[1][j] == tablica[2][j]) return true;
            if (kolor == tablica[i][(j + 1) % 8] && kolor == tablica[i][(j - 1) % 8]) return true;
        }

        return false;
    }


    mlynek(mlynek m) {
        this.tablica = m.tablica;
        this.bialepionki = m.bialepionki;
        this.czarnepionki = m.czarnepionki;
        this.dorozmieszczenia = m.dorozmieszczenia;
    }

    @Override
    public String toString() {
        return "mlynek{" +
                "tablica=" + Arrays.toString(tablica) +
                '}';
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(tablica);
    }

    @Override
    public List<GameState> generateChildren() {

        List<GameState> lista = new ArrayList<>();

        //etap1
        if (dorozmieszczenia != 0) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 8; j++) {
                    mlynek m = new mlynek(this);
                    m.dorozmieszczenia--;

                    //tura gracza maksymalizujacego
                    if (maximizingTurnNow) {
                        if (m.tablica[i][j] == 0) {
                            m.tablica[i][j] = 1;
                            if (czymlynek(i, j)) lista.addAll(stanymlynek(i, j));
                            else lista.add(m);
                        }
                    }

                    //tura gracza minimalizującego
                    else if (m.tablica[i][j] == 0) {
                        m.tablica[i][j] = 2;
                        if (czymlynek(i, j)) lista.addAll(stanymlynek(i, j));
                        else lista.add(m);
                    }

                }

            }
        }

        //etap2
        if (dorozmieszczenia == 0) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 8; j++) {
                    if (maximizingTurnNow) {

                    } else {

                    }
                }
            }

        }

        //etap3
        if (bialepionki == 3 || czarnepionki == 3) {

        }

        return null;

    }


    //POPRAWNIE
    List<GameState> stanymlynek(int a, int b) {

        boolean czyzabrano = false;
        List<GameState> lista = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 8; j++) {
                if (maximizingTurnNow) {
                    if (tablica[i][j] == 2 && !czymlynek(i, j)) {
                        mlynek m = new mlynek(this);
                        m.tablica[a][b] = 1;
                        m.czarnepionki--;
                        m.tablica[i][j] = 0;
                        lista.add(m);
                        czyzabrano = true;
                    }

                } else {
                    if (tablica[i][j] == 1 && !czymlynek(i, j)) {
                        mlynek m = new mlynek(this);
                        m.tablica[a][b] = 2;
                        m.bialepionki--;
                        m.tablica[i][j] = 0;
                        lista.add(m);
                        czyzabrano = true;
                    }
                }

            }
        }

        if (!czyzabrano) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 8; j++) {
                    if (maximizingTurnNow) {
                        if (tablica[i][j] == 2) {
                            mlynek m = new mlynek(this);
                            m.tablica[a][b] = 1;
                            m.czarnepionki--;
                            m.tablica[i][j] = 0;
                            lista.add(m);
                        }

                    } else {
                        if (tablica[i][j] == 1) {
                            mlynek m = new mlynek(this);
                            m.tablica[a][b] = 2;
                            m.bialepionki--;
                            m.tablica[i][j] = 0;
                            lista.add(m);
                        }
                    }

                }
            }

        }
        return lista;
    }


    List<GameState> generatemoves(int i, int j, boolean wszedzie) {

        List<GameState> lista = new ArrayList<>();
        if (wszedzie == false)
        {
            if (i % 2 == 0)
            {
                if(tablica[i][(j + 1) % 8] == 0)
                {
                    mlynek m = new mlynek(this);
                    m.tablica[i][(j + 1) % 8] = m.tablica[i][j];
                    m.tablica[i][j]=0;
                    lista.add(m);

                }
                if(tablica[i][(j + 7) % 8] == 0)
                {
                    mlynek m = new mlynek(this);
                    m.tablica[i][(j + 7) % 8] = m.tablica[i][j];
                    m.tablica[i][j]=0;
                    lista.add(m);
                    //dodać obsługe mlynkow
                }
            }
            if (j % 2 == 1) {


            }
        }
        if(wszedzie)
        {




        }

        return lista;
    }










}






