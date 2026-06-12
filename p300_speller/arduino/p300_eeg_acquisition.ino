/*
 * =============================================================================
 *  p300_eeg_acquisition.ino
 * -----------------------------------------------------------------------------
 *  Firmware for the EEG front-end of the P300 Speller communication prosthesis.
 *
 *  Hardware
 *  --------
 *    * Arduino Uno (ATmega328P)
 *    * Adafruit ADS1115 16-bit ADC over I2C (address 0x48)
 *    * 1 or 3 EEG channels referenced to a common reference + ground:
 *        - Single channel : Pz on ADS A0
 *        - Three channel  : Fz, Cz, Pz on ADS A0, A1, A2
 *
 *  Responsibilities
 *  ----------------
 *    1. Sample the ADS1115 at a deterministic rate (default 250 Hz).
 *    2. Emit one ASCII data packet per sample over Serial @ 115200 baud:
 *
 *         D,<seq>,<micros>,<marker>,<ch0>[,<ch1>,<ch2>]\n
 *
 *       where
 *         D        literal packet-type tag (Data)
 *         seq      uint32 monotonically increasing sample counter
 *         micros   uint32 microsecond timestamp from micros()
 *         marker   uint16 stimulus sync marker (0 = none) latched from host
 *         chN      int16 signed raw ADC counts for each channel
 *
 *    3. Accept inbound sync markers from the host so the exact sample that
 *       coincides with a row/column flash is tagged in the stream. The host
 *       sends a single line:
 *
 *         M<marker>\n              e.g. "M7\n"
 *
 *       The marker is latched onto the *next* emitted sample and then cleared,
 *       guaranteeing each marker is reported exactly once and is hardware-
 *       aligned to a real sample timestamp.
 *
 *  Design notes
 *  ------------
 *    * We read the ADS1115 in single-shot mode per channel. With 3 channels at
 *      a 250 Hz target the conversion budget is tight; the data-rate register
 *      is therefore set to 860 SPS so a single conversion completes in ~1.2 ms.
 *    * Timing is paced by micros() against a fixed sample period rather than
 *      delay(), so jitter introduced by Serial.print is absorbed rather than
 *      accumulated.
 *    * ASCII framing is used for robustness and ease of debugging; at 115200
 *      baud a 3-channel packet (~45 bytes) costs ~3.9 ms of TX time, which is
 *      pipelined with the next conversion. For higher channel counts switch to
 *      a binary frame.
 * =============================================================================
 */

#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ----------------------------- Compile-time config ---------------------------
#define NUM_CHANNELS      3          // 1 (Pz) or 3 (Fz, Cz, Pz)
#define SAMPLE_RATE_HZ    250UL      // MUST match configs/config.yaml
#define SERIAL_BAUD       115200UL
#define SYNC_PIN          13         // on-board LED pulses on each latched marker

static const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

// ----------------------------- Globals ---------------------------------------
Adafruit_ADS1115 ads;                // 16-bit ADC

volatile uint16_t pendingMarker = 0; // latched marker for the next sample
uint32_t          sampleSeq     = 0; // monotonic sample counter
unsigned long     nextSampleUs  = 0; // scheduled time of the next sample

char    rxBuffer[16];                // inbound marker line assembly buffer
uint8_t rxLen = 0;

// ----------------------------- Helpers ---------------------------------------

/**
 * Read all NUM_CHANNELS from the ADS1115 into the provided buffer as signed
 * 16-bit counts. Single-ended reads on A0..A(NUM_CHANNELS-1).
 */
static inline void readChannels(int16_t *out) {
  for (uint8_t ch = 0; ch < NUM_CHANNELS; ++ch) {
    out[ch] = ads.readADC_SingleEnded(ch);
  }
}

/**
 * Drain the serial input and latch any inbound "M<marker>\n" command. Multiple
 * markers in one period collapse to the most recent (a flash never overlaps
 * another flash, so this cannot lose a real event).
 */
static void pollHostMarkers() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen > 0 && rxBuffer[0] == 'M') {
        rxBuffer[rxLen] = '\0';
        long m = atol(&rxBuffer[1]);
        if (m >= 0 && m <= 65535L) {
          pendingMarker = (uint16_t)m;
          digitalWrite(SYNC_PIN, HIGH);  // visual confirmation of marker latch
        }
      }
      rxLen = 0;
    } else if (rxLen < sizeof(rxBuffer) - 1) {
      rxBuffer[rxLen++] = c;
    } else {
      rxLen = 0;  // overflow -> discard malformed line
    }
  }
}

/**
 * Emit one ASCII data packet for the current sample.
 */
static void emitPacket(uint32_t seq, unsigned long ts, uint16_t marker,
                       const int16_t *ch) {
  Serial.print('D');
  Serial.print(',');
  Serial.print(seq);
  Serial.print(',');
  Serial.print(ts);
  Serial.print(',');
  Serial.print(marker);
  for (uint8_t i = 0; i < NUM_CHANNELS; ++i) {
    Serial.print(',');
    Serial.print(ch[i]);
  }
  Serial.print('\n');
}

// ----------------------------- Arduino lifecycle -----------------------------

void setup() {
  pinMode(SYNC_PIN, OUTPUT);
  digitalWrite(SYNC_PIN, LOW);

  Serial.begin(SERIAL_BAUD);
  while (!Serial) { ; }  // wait for native-USB boards; harmless on the Uno

  Wire.begin();
  Wire.setClock(400000);  // 400 kHz fast-mode I2C to keep conversions snappy

  // GAIN_ONE -> +/-4.096 V full scale (matches adc_fullscale_v in config.yaml).
  ads.setGain(GAIN_ONE);

  if (!ads.begin()) {
    // Without the ADC there is nothing to stream; report and halt.
    Serial.println("E,ADS1115_NOT_FOUND");
    while (true) { digitalWrite(SYNC_PIN, HIGH); delay(100);
                   digitalWrite(SYNC_PIN, LOW);  delay(100); }
  }

  // 860 SPS data rate -> ~1.2 ms / conversion, leaving headroom at 250 Hz.
  ads.setDataRate(RATE_ADS1115_860SPS);

  // Announce the stream header so the host can auto-detect channel count.
  Serial.print("H,P300_EEG,");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.print(',');
  Serial.println(NUM_CHANNELS);

  nextSampleUs = micros();
}

void loop() {
  // Service inbound markers as often as possible to minimise latch latency.
  pollHostMarkers();

  // Pace acquisition against the fixed period using micros() arithmetic that
  // is robust to 32-bit rollover (~71 minutes).
  unsigned long now = micros();
  if ((long)(now - nextSampleUs) < 0) {
    return;  // not time for the next sample yet
  }
  nextSampleUs += SAMPLE_PERIOD_US;

  // Latch and clear the pending marker atomically with respect to this sample.
  uint16_t marker = pendingMarker;
  pendingMarker = 0;
  if (marker == 0) {
    digitalWrite(SYNC_PIN, LOW);  // drop the LED once the marker is consumed
  }

  int16_t channels[NUM_CHANNELS];
  readChannels(channels);

  emitPacket(sampleSeq++, now, marker, channels);
}
