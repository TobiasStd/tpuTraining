import subprocess
import time

# Definieren Sie die Anzahl der Stapel (Chargen) und die Stapelgröße
num_batches = 3
batch_size = 2

# Definieren Sie das gcloud-Kommando als Liste von Argumenten
gcloud_command = [
    'gcloud',
    'compute',
    'tpus',
    'tpu-vm',
    'ssh',
    'tpu-vm-1',
    '--zone=us-east1-d',
    '--project=tifm-llm',
    '--worker=all',
    '--command="PJRT_DEVICE=TPU XLA_USE_BF16=1 python3 tpuTraining/train.py"'
]

# Schleife über die Stapel
for batch_index in range(num_batches):
    print(f"Ausführen von Stapel {batch_index + 1}/{num_batches}")

    # Schleife über die Stapelgröße
    for _ in range(batch_size):
        # Führen Sie das gcloud-Kommando aus und erfassen Sie die Ausgabe
        process = subprocess.Popen(gcloud_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Überprüfen Sie, ob ein Fehler aufgetreten ist
        if process.returncode != 0:
            print("Fehler beim Ausführen des gcloud-Kommandos:")
            print(stderr.decode('utf-8'))  # Fehlerausgabe anzeigen
        else:
            print("gcloud-Kommando erfolgreich ausgeführt:")
            print(stdout.decode('utf-8'))  # Normale Ausgabe anzeigen

    # Warten Sie auf eine Bestätigung von den TPU-Workern
    print("Warte auf Bestätigung von TPU-Workern...")
    time.sleep(10)  # Eine angemessene Wartezeit einfügen, je nach Ihrer Arbeitslast und Netzwerkgeschwindigkeit

print("Alle Stapel wurden erfolgreich ausgeführt.")
