# Porównanie techniczne i inżynierskie bibliotek Pose Estimation w Pythonie

Dokument przedstawia zestawienie czterech badanych bibliotek Computer Vision (**MediaPipe Pose**, **Ultralytics YOLOv8/YOLO11-Pose**, **Google MoveNet** oraz **OpenMMLab RTMPose**) pod kątem parametrów technicznych, architektury, wymagań sprzętowych oraz wdrożenia mobilnego (*Edge AI*).

---

## 1. Tabela specyfikacji technicznej i architektonicznej modeli

> **Uwaga dot. rozdzielczości wejściowej (`Input Resolution`):** Wszystkie badane biblioteki przyjmują na wejściu obraz/wideo w **dowolnej rozdzielczości** (np. `1920x1080` Full HD lub `4K`). Podane w tabeli wymiary oznaczają **wewnętrzny rozmiar tensora sieci neuronowej**, do którego biblioteka automatycznie przeskalowuje całą klatkę lub wycięty prostokąt z sylwetką ćwiczącego (`ROI – Region of Interest`).

| Kryterium | MediaPipe Pose (`Lite` / `Full` / `Heavy`) | Ultralytics YOLO-Pose (`YOLOv8n` / `YOLO11n` / `11s`) | Google MoveNet (`Lightning` / `Thunder`) | OpenMMLab RTMPose (`Lightweight` / `Balanced`) |
| :--- | :--- | :--- | :--- | :--- |
| **Rozmiar modelu na dysku** | **Lite:** `5.6 MB`<br>**Full:** `9.0 MB`<br>**Heavy:** `~29.2 MB` | **YOLO11n:** `6.0 MB`<br>**YOLOv8n:** `6.6 MB`<br>**YOLO11s:** `20.0 MB` | **Lightning (FP16):** `4.6 MB`<br>**Thunder (FP16):** `~12.6 MB` | **Lightweight:** `41.0 MB` (`20 MB` det + `21 MB` pose)<br>**Balanced:** `149.0 MB` (`97 MB` det + `52 MB` pose) |
| **Wewnętrzna rozdzielczość wejściowa sieci** | **Stała:** `256 x 256 px` (sieć dostaje tylko wycięty `ROI` z człowiekiem po detekcji `224x224`, dzięki czemu nie traci pikseli na tło siłowni) | **Konfigurowalna (`imgsz`):** domyślnie `640 x 640 px` (cała klatka); można zmienić w kodzie np. na `320x320` (szybciej) lub `1280x1280` (dokładniej) | **Stała:**<br>**Lightning:** `192 x 192 px`<br>**Thunder:** `256 x 256 px`<br>(skaluje całą klatkę razem z tłem) | **Stała (2 etapy):**<br>**Detektor (`YOLOX`):** `416x416` / `640x640 px`<br>**Szkielet (`RTMPose`):** `256 x 192 px` (wycięty `ROI`) |
| **Liczba punktów (`Keypoints`) i wsparcie 3D** | **33 punkty (`BlazePose`)** (zawiera stopy: pięta/palce oraz dłonie) + **natywne współrzędne 3D (`x, y, z` w metrach)** | **17 punktów (`COCO`)** (tylko współrzędne 2D `x, y` na ekranie) | **17 punktów (`COCO`)** (tylko współrzędne 2D `y, x` na ekranie) | **17 punktów (`COCO`)** (współrzędne 2D `x, y`; dostępne też cięższe modele `WholeBody` 133 pkt) |
| **Obsługa wielu osób (`Multi-Person`)** *(Ciekawostka)* | Zoptymalizowany pod **1 osobę ćwiczącą** (można ustawić `num_poses > 1`, ale czas rośnie liniowo dla każdej kolejnej osoby) | **Natywny `Multi-Person`** (architektura `Single-Stage`: wykrywa wszystkich ludzi na siłowni w jednym przebiegu bez utraty prędkości) | **1 osoba** (`SinglePose`); istnieje osobny, wolniejszy wariant `MultiPose` (do 6 osób) | **`Multi-Person`** (architektura `Two-Stage Top-Down`: odpala model szkieletu osobno dla każdego wykrytego `bounding boxa`) |
| **Licencja komercyjna** | **Apache 2.0** (w pełni darmowa do użytku komercyjnego i zamkniętego kodu) | **AGPL-3.0** (wymaga otwarcia kodu aplikacji LUB zakupu płatnej licencji `Ultralytics Enterprise` przy zamkniętym projekcie komercyjnym) | **Apache 2.0** (w pełni darmowa do użytku komercyjnego i zamkniętego kodu) | **Apache 2.0** (w pełni darmowa do użytku komercyjnego i zamkniętego kodu) |
| **Wsparcie mobilne (`Edge AI`: Android / iOS)** | **Celujące (Natywne):** Oficjalne `MediaPipe Tasks SDK` na Androida (Kotlin/Java), iOS (Swift) i Fluttera z akceleracją `GPU/NPU` | **Dobre (przez eksport):** Eksport do `TFLite`, `CoreML`, `ONNX`, `NCNN`, ale wejście `640x640` wymaga mocnego mobilnego `GPU/NPU` do utrzymania >30 FPS | **Celujące (Natywne `TFLite`):** Plik `.tflite` działa bezpośrednio w `LiteRT` na Androidzie i iOS (`NNAPI` / `Metal`) | **Trudne:** Wymaga spięcia na telefonie **dwóch osobnych modeli** (`YOLOX` + `RTMPose` w `ONNX`/`NCNN`) oraz ręcznego wycinania `ROI` |
| **Zużycie pamięci RAM (w trakcie działania)** | **Niskie:** `~180 – 260 MB` RAM | **Średnie / Wysokie:** `~450 – 750 MB` RAM (ładuje środowisko `PyTorch` w Pythonie) | **Bardzo niskie:** `~110 – 160 MB` RAM (przy lekkim `ai-edge-litert`) | **Wysokie:** `~500 – 950 MB` RAM (trzyma w pamięci dwie sesje `ONNXRuntime`) |
| **Możliwość dotrenowania (`Fine-Tuning`)** | **Brak** (zamknięty, gotowy model `.task` od Google; nie można dotrenować własnych punktów szkieletu ani detekcji sztangi) | **Celująca (Out-of-the-box):** Prosty trening w 1 komendzie na własnych nagraniach z siłowni + możliwość jednoczesnego śledzenia gryfu/sztangi | **Brak** (zamknięte wagi `.tflite` z Kaggle/TF Hub bez oficjalnego pipeline'u do dotrenowywania szkieletu) | **Zaawansowana (Badawcza):** Pełna możliwość treningu w ekosystemie `OpenMMLab / mmpose`, ale trudna konfiguracja |

---

## 2. Przepustowość: Ile filmików `/bench-press/bench-press-01` przetworzymy w 1 godzinę?

Poniższe zestawienie pokazuje realną wydajność na testowym laptopie (przetwarzanie na CPU) w przeliczeniu na to, **ile pełnych nagrań `/bench-press/bench-press-01.mp4` dany model jest w stanie przeanalizować w ciągu 1 godziny (3600 sekund)**.

* **Wideo referencyjne:** `data/bench-press/bench-press-01.mp4`
* **Parametry nagrania:** `526 klatek` przy `30.0 FPS` (czas trwania nagrania: **`17.53 sekundy`**)
* **Wzór obliczeniowy:**
  $$\text{Czas na 1 film (s)} = \frac{526 \times \text{Avg Latency (ms)}}{1000}, \quad \text{Filmów na godzinę} = \frac{3600\text{ s}}{\text{Czas na 1 film (s)}}$$

| Wariant modelu | Średni czas klatki (`Avg Latency`) | Średnie `FPS` | Czas analizy 1 filmu (`526 klatek`) | Liczba filmów na godzinę (1 wątek CPU) | Przyspieszenie względem czasu rzeczywistego (`17.53 s`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MoveNet (Lightning)** | `4.75 ms` | `210.64 FPS` | **`2.50 s`** | **~1 440 filmów / godz.** | **`7.0x` szybciej niż real-time** *(Uwaga: tylko 34% wykrytych klatek na ławce!)* |
| **MediaPipe Pose (Lite)** | `9.24 ms` | `108.26 FPS` | **`4.86 s`** | **~740 filmów / godz.** | **`3.6x` szybciej niż real-time** (`100%` wykrytych klatek) |
| **MediaPipe Pose (Full)** | `13.56 ms` | `73.74 FPS` | **`7.13 s`** | **~504 filmy / godz.** | **`2.5x` szybciej niż real-time** (`100%` wykrytych klatek) |
| **YOLO11n-Pose (Nano)** | `78.78 ms` | `12.69 FPS` | **`41.44 s`** | **~87 filmów / godz.** | **`0.42x` real-time** (`100%` wykrytych klatek) |
| **YOLOv8n-Pose (Nano)** | `80.24 ms` | `12.46 FPS` | **`42.21 s`** | **~85 filmów / godz.** | **`0.42x` real-time** (`100%` wykrytych klatek) |
| **RTMPose (Lightweight)** | `311.83 ms` | `3.21 FPS` | **`164.02 s` (`2 min 44 s`)** | **~22 filmy / godz.** | **`0.11x` real-time** (`99.3%` wykrytych klatek) |
| **RTMPose (Balanced)** | `1318.75 ms` | `0.76 FPS` | **`693.66 s` (`11 min 34 s`)** | **~5.2 filmu / godz.** | **`0.025x` real-time** (`100%` wykrytych klatek) |

---

## 3. Kluczowe wnioski inżynierskie

1. **Gotowość pod aplikację mobilną i licencja:**
   * **MediaPipe Pose** oraz **MoveNet** posiadają otwartą licencję **Apache 2.0**, ważą poniżej `10 MB`, zużywają mało pamięci RAM (`<260 MB`) i mają oficjalne, natywne SDK na Androida oraz iOS.
   * **Ultralytics YOLO-Pose** jest objęty restrykcyjną licencją **AGPL-3.0** (wymaga udostępnienia kodu źródłowego aplikacji mobilnej na AGPL albo zakupu komercyjnej licencji) i przetwarza pełną klatkę `640x640`, co mocniej obciąża procesor bez akceleracji GPU/NPU.
2. **Szybkość a niezawodność na siłowni (pozycja leżąca i zasłonięcia przez sztangę):**
   * Chociaż **MoveNet Lightning** jest najszybszy (`~1 440 filmów/godz.`, `4.6 MB`), to przez skalowanie całego kadru razem z tłem do `192x192 px` zgubił sylwetkę ćwiczącego na **66% klatek** w `bench-press-01` (pozycja pozioma na ławce, zasłonięta talerzami sztangi).
   * **MediaPipe Pose (Lite oraz Full)** przerabia **`504–740 filmów/godz.`** (`2.5x–3.6x` szybciej niż czas rzeczywisty na samym CPU), osiąga **97–100% skuteczności detekcji** przy standardowym ustawieniu kamery i jako **jedyny zwraca 33 punkty (w tym stopy: pięta/palce) oraz współrzędne 3D (`x, y, z` w metrach)**.
   * **YOLOv8n / YOLO11n-Pose** jest najbardziej odporny na ekstremalne kąty kamery (`97.3–100%` detekcji nawet na trudnym nagraniu `bench-press-04`, gdzie MediaPipe spadł do `69–84%`) i jako **jedyny pozwala łatwo dotrenować model** (np. do jednoczesnego śledzenia toru sztangi), ale do osiągnięcia >30 FPS wymaga akceleracji GPU/NPU.
