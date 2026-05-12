/*
  Yuz Tanima - Arduino Tarafi (Common Anode kirmizi/yesil LED)
  Donanim:
    Buzzer:  bir bacak GND, diger bacak D8 (pasif buzzer)
    LED (COMMON ANODE - ortak/en uzun bacak +5V):
        Kirmizi -> 220 ohm -> D9
        Yesil   -> 220 ohm -> D7   (D11 tone() ile cakistigi icin D7'ye tasindi)
  COMMON ANODE: LOW = LED yanik, HIGH = sonuk.

  Seri komutlar (9600 baud):
    'G' -> Yesil sabit + periyodik kisa bip (taninan)
    'R' -> Kirmizi sabit + periyodik uzun bip (taninmayan)
    'B' -> Bekleme: hepsi kapali, sessiz
    'O' -> Hepsi kapali
    'T' -> Test
*/

const int PIN_BUZZER = 8;
const int PIN_RED    = 9;
const int PIN_GREEN  = 7;   // tone() Timer2 ile cakistigi icin D11 yerine D7

char lastState = 'O';
unsigned long lastBeep = 0;
unsigned long lastBeat = 0;

// COMMON ANODE: LOW = yanik, HIGH = sonuk
void setLeds(int r, int g, int b) {
  digitalWrite(PIN_RED,   (r > 0) ? LOW : HIGH);
  digitalWrite(PIN_GREEN, (g > 0) ? LOW : HIGH);
}

void allOff() {
  setLeds(0, 0, 0);
  noTone(PIN_BUZZER);
}

void beepShort() {
  tone(PIN_BUZZER, 2000, 120);
}

void beepLong() {
  tone(PIN_BUZZER, 500, 350);
}

void startupTest() {
  Serial.println(F("[boot] KIRMIZI"));
  setLeds(255, 0, 0); delay(800);
  Serial.println(F("[boot] YESIL"));
  setLeds(0, 255, 0); delay(800);
  Serial.println(F("[boot] BUZZER"));
  tone(PIN_BUZZER, 1000, 250);
  delay(350);
  allOff();
  allOff();
  Serial.println(F("[boot] hazir."));
}

void setup() {
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_RED,    OUTPUT);
  pinMode(PIN_GREEN,  OUTPUT);

  Serial.begin(9600);
  delay(200);
  Serial.println();
  Serial.println(F("=== Yuz Tanima Arduino ==="));

  allOff();
  startupTest();
  lastState = 'B';
}

void applyState(char s) {
  switch (s) {
    case 'G':
      setLeds(0, 255, 0);   // yesil sabit
      break;
    case 'R':
      setLeds(255, 0, 0);   // kirmizi sabit
      break;
    case 'B':
      allOff();             // bekleme: LED yok
      break;
    case 'O':
      allOff();
      break;
  }
}

void loop() {
  // 1) Gelen komutlari oku
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') continue;
    Serial.print(F("[cmd] ")); Serial.println(c);

    if (c == 'T' || c == 't') {
      startupTest();
      lastState = 'B';
      continue;
    }
    if (c == 'G' || c == 'g') c = 'G';
    else if (c == 'R' || c == 'r') c = 'R';
    else if (c == 'B' || c == 'b') c = 'B';
    else if (c == 'O' || c == 'o') c = 'O';
    else { Serial.println(F("[cmd] bilinmeyen")); continue; }

    lastState = c;
    applyState(c);
  }

  // 2) Durum sabit - LED'i de surekli zorla (tone sonrasi pin11 PWM bozulabilir)
  applyState(lastState);

  // 3) Periyodik bip
  unsigned long now = millis();
  if (lastState == 'G') {
    if (now - lastBeep >= 900) {   // her 0.9 sn bir kisa bip
      beepShort();
      lastBeep = now;
    }
  } else if (lastState == 'R') {
    if (now - lastBeep >= 600) {   // her 0.6 sn bir uzun bip
      beepLong();
      lastBeep = now;
    }
  } else {
    noTone(PIN_BUZZER);
  }

  // 4) Heartbeat
  if (now - lastBeat > 5000) {
    lastBeat = now;
    Serial.print(F("[hb] state=")); Serial.println(lastState);
  }
}
