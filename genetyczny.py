import numpy as np 
import matplotlib.pyplot as plt  

def problem(n, skala):
    przedmioty = np.ceil(skala * np.random.rand(n, 2)).astype("int32")
    pojemnosc = int(np.ceil(0.5 * 0.5 * n * skala))
    wartosci = przedmioty[:, 0]  # Wartości przedmiotów
    wagi = przedmioty[:, 1]  # Wagi przedmiotów
    return wartosci, wagi, pojemnosc  # wartości, wagi, pojemność plecaka

# Programowanie dynamiczne
def programowanie_dynamiczne(wartosci, wagi, pojemnosc):
    n = len(wartosci)

    dp = np.zeros((n + 1, pojemnosc + 1), dtype=int)

    for i in range(1, n + 1):  # Iterujemy po przedmiotach
        for w in range(1, pojemnosc + 1):  # po pojemności plecaka
            if wagi[i - 1] <= w:  # Jeśli przedmiot mieści się w plecaku
                # Wybór lepszej opcji: dodać przedmiot lub go pominąć
                dp[i][w] = max(wartosci[i - 1] + dp[i - 1][w - wagi[i - 1]], dp[i - 1][w])
            else:
                # pominięcie jeżeli się nie mieści
                dp[i][w] = dp[i - 1][w]

    # Odtwarzanie rozwiązania
    rozwiazanie = np.zeros(n, dtype=int)
    w = pojemnosc  # Startujemy od maksymalnej pojemności plecaka
    for i in range(n, 0, -1): # od tyłu
        if dp[i][w] != dp[i - 1][w]:  # Jeśli wartość się zmieniła
            rozwiazanie[i - 1] = 1  # Przedmiot został wybrany (1)
            w -= wagi[i - 1]  # zmniejszenie pojemności plecaka
    return dp[n][pojemnosc], rozwiazanie # maksymalna wartość i wektor rozwiązania





class AlgorytmGenetyczny:
    def __init__(self, n, funkcja_przystosowania, argumenty_funkcji):
        self.n = n  # Liczba przedmiotów
        self.funkcja_przystosowania = funkcja_przystosowania  # Funkcja przystosowania - uchwyt na funkcję
        self.argumenty_funkcji = argumenty_funkcji  # Argumenty funkcji przystosowania
        self.rozmiar_populacji = 1000  # Rozmiar populacji
        self.iteracje = 100  # Liczba iteracji
        self.wspolczynnik_mutacji = 0.001  # Współczynnik mutacji
        self.wspolczynnik_krzyzowania = 0.9  # Współczynnik krzyżowania

    # Inicjalizacja populacji
    def inicjalizacja_populacji(self):
        # inicjalizacja populacji tablicą losową 0 i 1
        return np.random.randint(2, size=(self.rozmiar_populacji, self.n))

    # Obliczenie przystosowania
    def przystosowanie(self, osobnik):
        #Dla danego osobnika obliczamy wartość jego przystosowania na bazie funkcji
        return self.funkcja_przystosowania(osobnik, *self.argumenty_funkcji)

    # Selekcja ruletkowa
    def selekcja_ruletkowa(self, populacja, wyniki_przystosowania):
        suma = sum(wyniki_przystosowania)  # suma przystosowania osobników w populacji
        rodzice = []

        for i in range(2):  # Wybieramy dwóch rodziców
            r = np.random.uniform(0, suma)  # losowa liczba od (0, S)
            suma_kumulacyjna = 0
            for j, przystosowanie in enumerate(wyniki_przystosowania):
                suma_kumulacyjna += przystosowanie  # Sumowanie przystosowań kolejnych chromosomów
                if suma_kumulacyjna >= r:
                    rodzice.append(populacja[j]) # przepisywanie do nowej populacji
                    break

        return np.array(rodzice)

    # Krzyżowanie
    def krzyzowanie(self, rodzic1, rodzic2):
        if np.random.random() < self.wspolczynnik_krzyzowania: #sprawdzenie czy nastąpi krzyżowanie
            punkt = np.random.randint(1, self.n)  #losowanie punktu w którym osobniki się krzyżują
            #wymiana genów
            dziecko1 = np.concatenate((rodzic1[:punkt], rodzic2[punkt:]))
            dziecko2 = np.concatenate((rodzic2[:punkt], rodzic1[punkt:]))
            return dziecko1, dziecko2 #zwrot dzieci jak było krzyżowanie
        return rodzic1, rodzic2  #zwrot rodziców

    # Mutacja
    def mutacja(self, osobnik):
        for i in range(self.n):  #iteracja po genach
            if np.random.random() < self.wspolczynnik_mutacji: #sprawdzenie czy nastąpi mutacja
                if osobnik[i] == 1:
                    osobnik[i] = 0
                else:
                    osobnik[i] = 1
        return osobnik  # Zwracamy zmutowanego osobnika

    # Główna pętla algorytmu
    def algorytm(self):
        populacja = self.inicjalizacja_populacji() #inicjalizacja populacji
        przystosowaniadowykresu = []
        for i in range(self.iteracje):  #iterowanie zadną liczbę iteracji

            wyniki_przystosowania = [self.przystosowanie(ind) for ind in populacja]  # Obliczenie przystosowania
            srednie_przystosowanie = np.mean(wyniki_przystosowania)  #obliczenie średniego przystosowania
            przystosowaniadowykresu.append(srednie_przystosowanie)  #dodanie go do wykresu
            nowa_populacja = []  #Tworzenie nowej populacji

            for j in range(self.rozmiar_populacji // 2):  #tworzenie nowego pokolenia
                rodzice = self.selekcja_ruletkowa(populacja, wyniki_przystosowania)  #wybór rodziców
                dziecko1, dziecko2 = self.krzyzowanie(rodzice[0], rodzice[1])  #krzyżowanie rodziców
                nowa_populacja.extend([self.mutacja(dziecko1), self.mutacja(dziecko2)])  #tworzenie potomków
            populacja = np.array(nowa_populacja)  #aktualizacja populacji

        #wybór najlepszego rozwiązania z populacji
        najlepsze_rozwiazanie = populacja[np.argmax([self.przystosowanie(ind) for ind in populacja])]
        return najlepsze_rozwiazanie, self.przystosowanie(najlepsze_rozwiazanie), przystosowaniadowykresu
        # Zwrot najlepsze rozwiązanie, wartość przystosowania, oraz dane do wykresu

# Funkcja przystosowania dla problemu plecakowego
def fprzystosowania(osobnik, wartosci, wagi, pojemnosc):
    wartosc = np.dot(osobnik, wartosci)  # Suma wartości wybranych przedmiotów suma value(ch_ij)
    waga = np.dot(osobnik, wagi)  # Suma wag wybranych przedmiotów
    if waga > pojemnosc:  # Jeśli waga przekracza pojemność plecaka
        return 0  # Kara: wartość 0
    return wartosc

# Funkcja porównująca rozwiązania dokładne i genetyczne
def porownanie(wartosc_dokladna, rozwiazanie_dokladne, wartosc_ga, rozwiazanie_ga):
    stosunek_wartosci = wartosc_ga / wartosc_dokladna  # Stosunek wartości algorytmu genetycznego do dokładnego
    dopasowanie_bitowe = np.sum(rozwiazanie_dokladne == rozwiazanie_ga) / len(rozwiazanie_dokladne) * 100  # Zgodność bitowa
    print(f"Stosunek wartości upakowania genetycznego do dokładnego: {stosunek_wartosci:.2f}")
    print(f"Procentowa zgodność bitowa rozwiązań: {dopasowanie_bitowe:.2f}%")

# Główna część programu
n = 10
skala = 2000
wartosci, wagi, pojemnosc = problem(n, skala)
print(f"Całkowita pojemność plecaka: {pojemnosc}")

wartosc_dokladna, rozwiazanie_dokladne = programowanie_dynamiczne(wartosci, wagi, pojemnosc)
print("\nWybrane przedmioty (dokładne rozwiązanie):")
wybrana_waga = 0
wybrana_wartosc = 0
for i, przedmiot in enumerate(rozwiazanie_dokladne):
    if przedmiot == 1:
        print(f"Przedmiot {i + 1}: Wartość = {wartosci[i]}, Waga = {wagi[i]}")
        wybrana_waga += wagi[i]
        wybrana_wartosc += wartosci[i]
print(f"Łączna waga: {wybrana_waga}, Łączna wartość: {wybrana_wartosc}")

ga = AlgorytmGenetyczny(n, fprzystosowania, (wartosci, wagi, pojemnosc))
rozwiazanie_ga, wartosc_ga, dowykresu = ga.algorytm()
print("\nWybrane przedmioty (algorytm genetyczny):")
wybrana_waga = 0
wybrana_wartosc = 0
for i, przedmiot in enumerate(rozwiazanie_ga):
    if przedmiot == 1:
        print(f"Przedmiot {i + 1}: Wartość = {wartosci[i]}, Waga = {wagi[i]}")
        wybrana_waga += wagi[i]
        wybrana_wartosc += wartosci[i]
print(f"Łączna waga: {wybrana_waga}, Łączna wartość: {wybrana_wartosc}")

porownanie(wartosc_dokladna, rozwiazanie_dokladne, wartosc_ga, rozwiazanie_ga)

# Wykres przystosowania populacji w trakcie pracy algorytmu genetycznego
plt.plot(dowykresu, label="Średnie przystosowanie")
plt.xlabel("Iteracja")
plt.ylabel("Średnie przystosowanie (wartość plecaka)")
plt.title("Przystosowanie populacji w trakcie pracy algorytmu genetycznego")
plt.legend()
plt.show()