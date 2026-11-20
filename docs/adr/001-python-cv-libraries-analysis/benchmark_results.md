# Wyniki benchmarku wydajnościowego (`benchmark_results.md`)

Zestawienie pomiarów ilościowych dla 4 badanych bibliotek Pose Estimation wykonanych na 4 nagraniach wyciskania sztangi leżąc (`bench-press-01` .. `04`, po `150 klatek` każde).

* **Legenda metryk:**
  * **Punkty (`Keypoints`):** Liczba zwracanych punktów anatomicznych oraz układ współrzędnych (`2D` w pikselach ekranu vs `3D World` w metrach).
  * **Skuteczność detekcji (`Detection Rate`):** Odsetek klatek, w których model poprawnie wykrył sylwetkę ćwiczącego na ławce.
  * **Średni czas klatki (`Avg Latency`):** Średni czas inferencji jednej klatki wideo w milisekundach (poniżej `33.3 ms` = płynne `30+ FPS`).
  * **95. percentyl (`P95 Latency`):** Czas inferencji, poniżej którego mieści się 95% klatek (mierzy stabilność i brak przycięć).
  * **Średnie `FPS` (`Avg FPS`):** Średnia liczba klatek przetwarzanych na sekundę (`1000 / Avg Latency`).

---

### Quantitative Benchmark Results (`/bench-press/bench-press-01`)

| Model / Biblioteka | Punkty (`Keypoints 2D/3D`) | Skuteczność detekcji (`Detection Rate`) | Średni czas (`Avg Latency`) | 95. percentyl (`P95 Latency`) | Średnie `FPS` (`Avg FPS`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Pose (Lite)** | 33 (`2D` + `3D World`) | 150/150 klatek (100.0%) | 9.24 ms | 10.37 ms | **108.26 FPS** |
| **MediaPipe Pose (Full)** | 33 (`2D` + `3D World`) | 150/150 klatek (100.0%) | 13.56 ms | 15.12 ms | **73.74 FPS** |
| **YOLOv8n-Pose (Nano)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 80.24 ms | 98.85 ms | **12.46 FPS** |
| **YOLO11n-Pose (Nano)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 78.78 ms | 87.95 ms | **12.69 FPS** |
| **MoveNet (Lightning)** | 17 `COCO` (`2D`) | 51/150 klatek (34.0%) | 4.75 ms | 6.18 ms | **210.64 FPS** |
| **MoveNet (Thunder)** | 17 `COCO` (`2D`) | ERROR | ERROR | ERROR | **ERROR** |
| **RTMPose (Lightweight)** | 17 `COCO` (`2D`) | 149/150 klatek (99.3%) | 311.83 ms | 402.03 ms | **3.21 FPS** |
| **RTMPose (Balanced)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 1318.75 ms | 1473.58 ms | **0.76 FPS** |

---

### Quantitative Benchmark Results (`/bench-press/bench-press-02`)

| Model / Biblioteka | Punkty (`Keypoints 2D/3D`) | Skuteczność detekcji (`Detection Rate`) | Średni czas (`Avg Latency`) | 95. percentyl (`P95 Latency`) | Średnie `FPS` (`Avg FPS`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Pose (Lite)** | 33 (`2D` + `3D World`) | 149/150 klatek (99.3%) | 10.73 ms | 12.78 ms | **93.20 FPS** |
| **MediaPipe Pose (Full)** | 33 (`2D` + `3D World`) | 146/150 klatek (97.3%) | 14.01 ms | 17.91 ms | **71.35 FPS** |
| **YOLOv8n-Pose (Nano)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 64.15 ms | 111.89 ms | **15.59 FPS** |
| **YOLO11n-Pose (Nano)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 80.28 ms | 85.74 ms | **12.46 FPS** |
| **MoveNet (Lightning)** | 17 `COCO` (`2D`) | 148/150 klatek (98.7%) | 4.45 ms | 6.34 ms | **224.68 FPS** |
| **MoveNet (Thunder)** | 17 `COCO` (`2D`) | ERROR | ERROR | ERROR | **ERROR** |
| **RTMPose (Lightweight)** | 17 `COCO` (`2D`) | 150/150 klatek (150/150, 100.0%) | 250.56 ms | 431.12 ms | **3.99 FPS** |
| **RTMPose (Balanced)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 1264.05 ms | 1586.31 ms | **0.79 FPS** |

---

### Quantitative Benchmark Results (`/bench-press/bench-press-03`)

| Model / Biblioteka | Punkty (`Keypoints 2D/3D`) | Skuteczność detekcji (`Detection Rate`) | Średni czas (`Avg Latency`) | 95. percentyl (`P95 Latency`) | Średnie `FPS` (`Avg FPS`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Pose (Lite)** | 33 (`2D` + `3D World`) | 150/150 klatek (100.0%) | 18.48 ms | 21.26 ms | **54.12 FPS** |
| **MediaPipe Pose (Full)** | 33 (`2D` + `3D World`) | 136/150 klatek (90.7%) | 20.49 ms | 33.99 ms | **48.81 FPS** |
| **YOLOv8n-Pose (Nano)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 113.61 ms | 181.18 ms | **8.80 FPS** |
| **YOLO11n-Pose (Nano)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 125.04 ms | 171.84 ms | **8.00 FPS** |
| **MoveNet (Lightning)** | 17 `COCO` (`2D`) | 107/150 klatek (71.3%) | 5.67 ms | 10.43 ms | **176.34 FPS** |
| **MoveNet (Thunder)** | 17 `COCO` (`2D`) | ERROR | ERROR | ERROR | **ERROR** |
| **RTMPose (Lightweight)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 168.78 ms | 230.97 ms | **5.92 FPS** |
| **RTMPose (Balanced)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 1088.57 ms | 1344.23 ms | **0.92 FPS** |

---

### Quantitative Benchmark Results (`/bench-press/bench-press-04`)

| Model / Biblioteka | Punkty (`Keypoints 2D/3D`) | Skuteczność detekcji (`Detection Rate`) | Średni czas (`Avg Latency`) | 95. percentyl (`P95 Latency`) | Średnie `FPS` (`Avg FPS`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Pose (Lite)** | 33 (`2D` + `3D World`) | 126/150 klatek (84.0%) | 13.92 ms | 25.79 ms | **71.81 FPS** |
| **MediaPipe Pose (Full)** | 33 (`2D` + `3D World`) | 104/150 klatek (69.3%) | 18.31 ms | 29.54 ms | **54.61 FPS** |
| **YOLOv8n-Pose (Nano)** | 17 `COCO` (`2D`) | 149/150 klatek (99.3%) | 92.57 ms | 108.23 ms | **10.80 FPS** |
| **YOLO11n-Pose (Nano)** | 17 `COCO` (`2D`) | 146/150 klatek (97.3%) | 119.79 ms | 200.32 ms | **8.35 FPS** |
| **MoveNet (Lightning)** | 17 `COCO` (`2D`) | 136/150 klatek (90.7%) | 4.94 ms | 6.95 ms | **202.56 FPS** |
| **MoveNet (Thunder)** | 17 `COCO` (`2D`) | ERROR | ERROR | ERROR | **ERROR** |
| **RTMPose (Lightweight)** | 17 `COCO` (`2D`) | 149/150 klatek (99.3%) | 161.95 ms | 249.02 ms | **6.17 FPS** |
| **RTMPose (Balanced)** | 17 `COCO` (`2D`) | 150/150 klatek (100.0%) | 853.32 ms | 976.10 ms | **1.17 FPS** |
