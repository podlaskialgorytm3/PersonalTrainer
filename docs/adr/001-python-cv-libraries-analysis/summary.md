# Podsumowanie analizy bibliotek CV (ADR 001)

W ramach analizy porównano 4 biblioteki do estymacji pozy w Pythonie pod kątem ekstrakcji kątów biomechanicznych, zliczania powtórzeń w czasie rzeczywistym (`30–60 FPS`) oraz wdrożenia w aplikacji mobilnej:

1. **MediaPipe Pose (`Full` / `Lite`) — `92%`**
   Osiąga bardzo wysoką wydajność na samym CPU (`50–108 FPS`, czyli `504–740 filmów/godz.`) przy zachowaniu `97–100%` skuteczności detekcji na standardowych ujęciach i wadze zaledwie `5.6–9.0 MB`. Jako jedyny model zwraca aż **33 punkty anatomiczne** (w tym pięty i palce stóp) oraz **współrzędne 3D w metrach**, a także posiada darmową licencję `Apache 2.0` i natywne SDK na Androida oraz iOS.

2. **Ultralytics YOLOv8n / YOLO11n-Pose — `78%`**
   Najbardziej odporny na trudne kąty kamery i zasłonięcia przez talerze sztangi (`97–100%` wykrytych klatek na każdym nagraniu) oraz jako jedyny pozwala łatwo dotrenować model np. do śledzenia toru sztangi (*barbell tracking*). Zwraca jednak tylko `17 punktów 2D (COCO)`, na samym CPU działa wolniej (`8–15 FPS`, wymaga `GPU/NPU`), a licencja `AGPL-3.0` ogranicza zamknięte zastosowania komercyjne.

3. **Google MoveNet (`Lightning`) — `52%`**
   Jest ekstremalnie szybki (`176–224 FPS`, `~1 440 filmów/godz.`) i lekki (`4.6 MB`, licencja `Apache 2.0`), świetnie nadając się na telefony do prostych ćwiczeń w staniu. Przez niską rozdzielczość wejściową (`192x192 px` na cały kadr) gubił jednak ćwiczącego w pozycji leżącej na ławce (`tylko 34%` wykrytych klatek w `bench-press-01`) i zwraca wyłącznie `17 punktów 2D`.

4. **OpenMMLab RTMPose (`Lightweight` / `Balanced`) — `35%`**
   Zapewnia bardzo wysoką dokładność detekcji (`99–100%` klatek) i otwartą licencję `Apache 2.0`, ale jego dwuetapowa architektura (osobny detektor `YOLOX` + `RTMPose` na każdej klatce) jest zbyt ciężka. Na CPU osiąga zaledwie `0.8–6.1 FPS` (`5–22 filmy/godz.`), waży `41–149 MB` i jest najtrudniejszy do wdrożenia na smartfonie.

---

### Ostateczna decyzja
Wybieramy **MediaPipe Pose (`92%`)** jako główny silnik do wyliczania kątów biomechanicznych 2D/3D, zliczania powtórzeń i wdrożenia mobilnego, zostawiając **YOLO-Pose (`78%`)** jako opcję rezerwową do ewentualnego śledzenia toru sztangi.
