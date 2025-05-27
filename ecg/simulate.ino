// Simulated ECG waveform values (PQRST)
const int ecgWave[] = {
  // P wave
  512, 515, 518, 521, 524, 527, 530, 533, 536, 540,
  // Q drop
  520, 500, 480, 460, 440,
  // R spike
  600, 520, 500,
  // S and up
  520, 530, 540, 530, 520, 510,
  // T wave
  515, 520, 518, 516, 514, 512, 510,
  // Rest between beats
  512, 512, 512, 512, 512
};
const int waveLength = sizeof(ecgWave) / sizeof(ecgWave[0]);
int index = 0;

void setup() {
  pinMode(9, OUTPUT);        // PWM output
  Serial.begin(9600);        // For Serial Plotter
}

void loop() {
  int ecgValue = ecgWave[index++];

  // Output PWM to pin D9 (map 0–1023 to 0–255 for analogWrite)
  analogWrite(9, map(ecgValue, 0, 1023, 0, 255));

  // Send to Serial Plotter
  Serial.println(ecgValue);

  // Loop waveform
  if (index >= waveLength) index = 0;

  delay(30);  // Adjust heartbeat speed (lower is faster)
}
