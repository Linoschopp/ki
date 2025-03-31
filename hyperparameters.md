### **Batch Size-Vergleich**  

| Eigenschaft             | **Kleine Batch Size** (z. B. 16, 32, 64) | **Große Batch Size** (z. B. 128, 256) |
|------------------------|--------------------------------|--------------------------------|
| **Generalisierung**    | 🟢 Gut (besser gegen Overfitting) | 🔴 Schlecht (Overfitting-Risiko) |
| **Konvergenzgeschwindigkeit** | 🟢 Schnell (häufige Updates) | 🟠 Langsam (weniger Updates pro Epoche) |
| **Stabilität der Gradienten** | 🟠 Unstabil (mehr Rauschen) | 🟢 Stabil (weniger zufällige Schwankungen) |
| **Speicherverbrauch (RAM/GPU)** | 🟢 Gering (wenig Speicher nötig) | 🔴 Hoch (braucht viel Speicher) |
| **Effizienz auf GPUs** | 🟠 Schlecht (weniger parallelisiert) | 🟢 Gut (bessere GPU-Nutzung) |
| **Gefahr von schlechten Minima** | 🟢 Gering (kommt leichter raus) | 🟠 Hoch (bleibt eher stecken) |