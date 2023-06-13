import numpy as np
import random
from sklearn.utils import shuffle  # Do potasowania próbek testowych


class Siec_neuronowa:
    def __init__(self, warstwy, alfa=0.1, beta=1):

        self.WAGI = []  # Lista przechowująca wszystkie wagi, dla wszystkich warstw
        self.Warstwy = warstwy  # Lista przechowująca architekturę sieci neuronowej
        self.Alfa = alfa  # Współczynnik uczenia
        self.Beta = beta  # Parametr Beta
        self.Epoki = 0  # Ilość epok jakie sieć była uczona

        # Generowanie Wag dla, poszczególnych warstw
        for i in range(1, len(self.Warstwy)):
            # Wyznaczenie wymiarów macierzy przechowującej wagi dla danej warstwy neuronów
            # Pierwsza warstwa zawiera tyle neuronów ile jest wejść
            # ilość wierszy, równa liczbie neuronów w aktualnej warstwie
            wiersze = self.Warstwy[i]
            # ilość kolumn, równa liczbie wszystkich wag, przydzielonych do danego neuronu,
            # zależna od ilości neuronów w poprzedniej warstwie
            kolumny = self.Warstwy[i - 1] + 1
            # Deklaracja macierzy dla wag danej warstwy (początkowo wypełniona zerami)
            wagi_macierz = np.zeros([wiersze, kolumny])

            # Przejście za pomocą pętli po wszystkich elementach macierzy
            # i losowana jest dla nich waga z zakresu (-1, 1)
            for i in range(wiersze):
                for j in range(kolumny):
                    znak = -1
                    # Wyznaczenie znaku początkowej wagi (plus lub minus)
                    if (random.random() < 0.5):
                        znak = 1
                    wagi_macierz[i][j] = random.random() * znak
            self.WAGI.append(wagi_macierz)

        # Należy odkomentować, by ustawić wagi dokładnie takie
        # jak w przykładzie w pliku PDF
        # self.WAGI[0][0] = [0.1, 0.2, 0.3]
        # self.WAGI[0][1] = [0.4, 0.5, 0.6]
        # self.WAGI[1][0] = [0.7, -0.8, 0.9]

    # Metoda wyświetlająca aktualne wagi sieci neuronowej
    def pokaz_wagi(self):
        print("Wagi sieci neuronowej:")
        print(self.WAGI)

    # Metoda zwracająca liczbę epok
    def epoki(self):
        return self.Epoki

    # Metoda zwracająca tablicę wielowymiarową (numpy.ndarray) z  przyjmująca jako argument listę,
    # i stosująca dla jej wartości Sigmoidalną funkcję aktywacji
    def funkcja_sigmoidalna(self, tab_wartosci):
        przetworzone = np.ones([1, len(tab_wartosci) + 1])
        for i in range(len(tab_wartosci)):
            przetworzone[0][i] = 1.0 / (1 + np.exp(-self.Beta*tab_wartosci[i]))
        return przetworzone

    # Metoda oblicza pochodną z sigmoidalnej funkcji aktywacyjnej dla wartości
    # podanych w argumencie (tablica numpy.ndarray)
    def oblicz_pochodna(self, tab_wartosci):
        pochodne_tab = np.ones([1, len(tab_wartosci) - 1])
        for i in range(len(tab_wartosci)-1):
            pochodne_tab[0][i] = self.Beta * \
                tab_wartosci[i] * (1 - tab_wartosci[i])
        return pochodne_tab

    # Metoda wyliczająca wartości w poszczególnych warstwach, idąc od początku
    # do końca dla podanego argumentu, by następnie stosując propagację wsteczną,
    # wprowadzać poprawki do wag sieci neuronowej względem oczekiwanej wartości
    def ucz_podana_probka(self, wart_wejsciowe, wart_oczekiwane):
        # Tablica przechowująca wyliczone wartości dla poszczególnych warstw
        aktywowane_warstwy = [np.atleast_2d(wart_wejsciowe)]

        # Wyliczenie warstwa po warstwie
        for warstwa in range(0, len(self.WAGI)):
            zwazone_wartosci = []
            for i in range(len(self.WAGI[warstwa])):
                # Dodanie do listy sumy iloczynów wejść i ich wag
                zwazone_wartosci.append(
                    aktywowane_warstwy[warstwa].dot(self.WAGI[warstwa][i]))
            # Zastosowanie funkcji aktywacji na tablicy wyliczonych wartości
            przetworzone = self.funkcja_sigmoidalna(zwazone_wartosci)

            # Dodanie do tablicy wyliczonych wartości dla danej warstwy
            aktywowane_warstwy.append(przetworzone)

        # Zastosowanie wstecznej propagacji idąc od ostatniej wartości w tablicy
        # [-1] oznacza indeks ostatniego elementu
        wyliczona_wartosc = aktywowane_warstwy[-1][0][0]

        # Obliczenie różnicy między wartością oczekiwaną, a obliczoną
        roznica = wart_oczekiwane[0] - wyliczona_wartosc
        # Wyznaczenie korekty w oparciu o współczynnik uczenia alfa
        korekta = self.Alfa * roznica

        # Lista przechowująca poprawki do wag
        # Pierwsza poprawka to iloczyn wyliczonej korekty i pochodnej wartości ostatniej warstwy
        poprawki = [korekta * self.oblicz_pochodna(aktywowane_warstwy[-1][0])]

        # Przechodzenie po warstwach od końca
        for warstwa in range(len(aktywowane_warstwy) - 2, 0, -1):
            # Wyliczenie poprawki dla danej warstwy, wględem poprawki z poprzedniej warstwy
            poprawki_warstwy = poprawki[-1].dot(self.WAGI[warstwa])
            # usunięcie ostatniej kolumny ze zbądną wartością
            poprawki_warstwy = np.delete(
                poprawki_warstwy, len(poprawki_warstwy[0])-1, 1)
            # Wyznaczenie wartości pochodnych sigmoidalnej funkcji aktywacyjnej
            pochodne = self.oblicz_pochodna(aktywowane_warstwy[warstwa][0])
            for i in range(len(poprawki_warstwy[0])):
                # Iloczyn poprawek danej warstwy z wartościami odpowiadającym im pochodnych
                poprawki_warstwy[0][i] *= pochodne[0][i]
            # Dodanie poprawek wag danej warstwy do listy
            poprawki.append(poprawki_warstwy)

        # Odwrócenie kolejności listy poprawek
        poprawki = poprawki[::-1]

        # Aktualizacja wag na podstawie poprawek danej warstwy
        for nr_warstwy in range(len(self.WAGI)):
            for nr_neuronu in range(len(self.WAGI[nr_warstwy])):
                for nr_wagi in range(len(self.WAGI[nr_warstwy][nr_neuronu])):
                    self.WAGI[nr_warstwy][nr_neuronu][nr_wagi] += aktywowane_warstwy[nr_warstwy][0][nr_wagi] * \
                        poprawki[nr_warstwy][0][nr_neuronu]

    # Metoda ucząca daną sieć neuronową za pomocą podanych danych,
    # przez określoną liczbę epok
    def ucz_siec(self, Wejscia, Wyjscia, liczba_epok=20000, wyswietl_stan=100):
        # Dodanie wejść ukrytych do neuronów (równe 1)
        ukryte_wejscia = np.ones((len(Wejscia), 1))
        Wejscia = np.append(Wejscia, ukryte_wejscia, axis=1)

        for epoka in range(1, liczba_epok+1):
            # losowe rozłożenie próbek uczących i przypisanych do nich wartości oczekiwanych
            Wejscia, Wyjscia = shuffle(Wejscia, Wyjscia, random_state=0)
            # uczenie każdą probką w pętli
            for i in range(len(Wejscia)):
                self.ucz_podana_probka(Wejscia[i], Wyjscia[i])
            # Monitorowanie liczby epok uczenia dla statystyk
            self.Epoki += 1

            # Wyświetlenia aktualnego stanu nauczania, co zadaną liczbę iteracji
            if epoka == 1 or (epoka) % wyswietl_stan == 0:
                modul_bledu = self.wylicz_modul_bledu(Wejscia, Wyjscia)
                print("Epoka: {}, moduł błędu = {:.8f}".format(
                    epoka, modul_bledu))

    # Metoda zwracająca moduł błędu dla podanych danych
    def wylicz_modul_bledu(self, wartosci_wejsciowe, wartosci_oczekiwane):
        oszacowane = []

        for wartosc in wartosci_wejsciowe:
            oszacowane.append(self.oszacuj(wartosc))

        modul_bledu = 0
        # moduł błędu wyliczany jest jako suma wartości bezwględnych z
        # różnicy między wartościami oczekiwanymi, a oszacowanymi przez algorytm
        for indeks in range(len(oszacowane)):
            modul_bledu += abs(wartosci_oczekiwane[indeks]
                               [0] - oszacowane[indeks][0])
        return modul_bledu

    # Metoda zwarcająca ostatnią wartość neuronu z tablicy
    # wyliczoną na wyjściu dla aktualnych wartości wag
    def oszacuj(self, wartosci_wejciowe):

        # Utworzenie tablicy przechowującej wejścia
        wartosci_neuronow = [np.atleast_2d(wartosci_wejciowe)]

        # Do listy dodawane są wartości neuronów po pomnożeniu ich przez wagi
        # następnie zastosowana jest na nich funkcja aktywacji
        for warstwa in range(0, len(self.WAGI)):
            zwazone_wartosci = []
            for i in range(len(self.WAGI[warstwa])):
                zwazone_wartosci.append(wartosci_neuronow[warstwa].dot(
                    self.WAGI[warstwa][i]))
            przetworzone = self.funkcja_sigmoidalna(zwazone_wartosci)
            wartosci_neuronow.append(przetworzone)
        # Indeks 0 ostatniego elementu w tablicy zawiera wyliczoną wartość na wyjściu ostatniego neuronu
        return wartosci_neuronow[-1][0]

    # Metoda do testowania nauczonej sieci neuronowej
    def testuj(self, X, Y):
        for (wejscia, oczekiwane) in zip(X, Y):
            # Dodanie wejść ukrytych do neuronów (równe 1)
            wejscia = np.append(wejscia, 1)
            wyliczone_wartosci = self.oszacuj(wejscia)
            wejscia = np.delete(wejscia, len(wejscia)-1, 0)
            # Wyznaczony wynik w oparciu o oszacowną wartość
            wynik = 1 if wyliczone_wartosci[0] > 0.5 else 0
            print("Epoki uczenia: {}\tWejścia: {}\tOczekiwny: {}\tOszacowano: {:.5f}\tWynik: {}".format(
                self.epoki(), wejscia, oczekiwane[0], wyliczone_wartosci[0], wynik))
