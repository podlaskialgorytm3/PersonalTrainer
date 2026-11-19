Temat:
Inteligentny wirtualny trener personalny: system monitorowania przebiegu treningu, automatycznego zliczania powtórzeń i korekcji techniki w czasie rzeczywistym z wykorzystaniem Computer Vision oraz telemetrii IoT
1. Cel i istota projektu
Opracowanie kompletnego, multimodalnego systemu pełniącego rolę autonomicznego trenera personalnego. Aplikacja monitoruje pełny przebieg jednostki treningowej – automatycznie rejestruje serie i powtórzenia, koryguje technikę użytkownika w czasie rzeczywistym komunikatami głosowymi oraz analizuje zmęczenie fizjologiczne na podstawie danych z urządzeń ubieralnych (smartwatch).
2. Źródła danych i sensoryka
Strumień wideo (kamera smartfona):
Detekcja szkieletu anatomicznego 2D/3D w czasie rzeczywistym (modele brzegowe: YOLO-Pose / MediaPipe).
Śledzenie punktów kluczowych ciała, trajektorii przyborów (tor gryfu/sztangi) i kątów w stawach.
Telemetria biometryczna (smartwatch / pas tętna BLE):
Tętno na żywo (HR), dynamika regeneracji tętna w przerwach między seriami (HR Recovery) oraz czas odpoczynku.
Fuzja danych:
Synchronizacja czasowa parametrów fizjologicznych z klatkami wideo i kolejnymi powtórzeniami.
3. Kluczowe funkcjonalności systemu
Automatyczne zliczanie powtórzeń i rejestracja serii:
Algorytmiczne wykrywanie początku, fazy ekscentrycznej (opuszczanie), koncentrycznej (podnoszenie) oraz końca powtórzenia.
Odrzucanie powtórzeń niepełnych lub niespełniających kryteriów zakresu ruchu (np. zbyt płytki przysiad).
Nadzór nad przebiegiem treningu (Workout Flow):
Automatyczne odmierzanie przerw wypoczynkowych (aktywowane po odłożeniu ciężaru / zakończeniu serii).
Sugestie rozpoczęcia kolejnej serii w oparciu o spadek tętna do strefy gotowości.
Rejestracja tempa powtórzeń (Time Under Tension) oraz prędkości fazy koncentrycznej (Velocity-Based Training).
Interaktywny instruktaż i nauka techniki (Live Audio Cueing):
Tryb nauki wzorca ruchowego (ćwiczenia bez obciążenia lub z samym gryfem).
Natychmiastowe wskazówki głosowe (Text-to-Speech) w trakcie trwania ruchu (np. „unieś wyżej łokieć”, „kolana węziej”, „utrzymaj proste plecy”).
Korelacja zmęczenia ze spadkiem jakości techniki (Fatigue Breakdown):
Wykrywanie momentu, w którym wzrost tętna i zmęczenie prowadzą do kompensacji i błędów biomechanicznych (np. koślawienie kolan, zaokrąglanie odcinka lędźwiowego).
Dziennik treningowy i analiza progresu:
Historia objętości treningowej, wskaźniki powtarzalności technicznej oraz raporty symetrii obciążenia kończyn.
4. Aspekty inżynierskie i naukowe
Edge AI: Przetwarzanie obrazu i generowanie feedbacku bezpośrednio na urządzeniu mobilnym (minimalizacja opóźnień krytycznych dla wskazówek w locie).
Fuzja heterogenicznych strumieni: Integracja sygnałów o różnych częstotliwościach próbkowania (kamera 30–60 Hz vs. telemetria 1 Hz).
Biomechaniczna walidacja ruchu: Zastąpienie subiektywnej oceny wzrokowej twardymi regułami kinematycznymi (kąty stawowe, prędkości kątowe, płaszczyzny ruchu).