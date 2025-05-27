void setup() {
  Serial.begin(9600);
  pinMode(10, INPUT); // LO+
  pinMode(11, INPUT); // LO-
}

void loop() {
  if ((digitalRead(10) == 1) || (digitalRead(11) == 1)) {
    // Serial.println("Electrode not connected");
    Serial.println(512);
  } else {
    int ecgValue = analogRead(A0);
    Serial.println(ecgValue);
  }
  delay(10);
}
