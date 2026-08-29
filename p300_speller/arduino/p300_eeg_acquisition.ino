/*
 * =============================================================================
 *  p300_eeg_acquisition.ino
 * -----------------------------------------------------------------------------
 *  Firmware for the EEG front-end of the P300 Speller communication prosthesis.
 *
 *  Hardware
 *  --------
 *    * Arduino Uno (ATmega328P)
 *    * Adafruit ADS1115 16-bit ADC over I2C (address 0x48), fast-mode 400 kHz
 *    * LM393 photodiode comparator module on D2 (INT0) for optical stimulus
 *      onset detection
 *    * 1 or 3 EEG channels referenced to a common reference + ground:
 *        - Single channel : Pz on ADS A0
 *        - Three channel  : Fz, Cz, Pz on ADS A0, A1, A2
 *        - Ground electrode: Fpz
 *
 *  Analogue chain (per channel) — TOTAL GAIN 949
 *  ---------------------------------------------
 *      INA118P instrumentation amp : 23.73   (RG = 2.2 kOhm)
 *      TL072 second stage          : 40
 *      ------------------------------------
 *      Total                       : 949
 *
 *      High-pass corner  : 0.48 Hz
 *      Low-pass          : Sallen-Key, 33.6 Hz, Q = 0.72
 *      Supply            : single +5 V, DC-coupled input
 *      Impedance target  : 10 kOhm
 *
 *    The host MUST divide raw counts by this gain to recover scalp microvolts
 *    (see acquisition.amplifier_gain in configs/config.yaml). 949 is the
 *    DESIGNED value from the component values; it has not been measured on the
 *    bench. Measure it before quoting any absolute amplitude.
 *
 *  MARKER ARCHITECTURE — host arms, photodiode latches
 *  ---------------------------------------------------
 *    The two halves are both required and neither is sufficient:
 *
 *      * The HOST supplies stimulus IDENTITY. Only the host knows which row or
 *        column is about to flash, so it arms a marker code before the flash:
 *            M<code>\n           e.g. "M7\n"
 *      * The PHOTODIODE supplies TIMING. The LM393 fires INT0 when the screen
 *        actually goes bright, and the ISR timestamps that edge with micros().
 *        The comparator cannot know *what* flashed, only *when*.
 *
 *    The previous revision latched the host's marker onto the next emitted
 *    sample. That path carries USB serial transit, host scheduler jitter, and
 *    up to one full sample period of quantisation — 4 ms at 250 Hz from
 *    quantisation alone, before transit. Timestamping the optical edge in an
 *    ISR removes all of it.
 *
 *    For backward compatibility the armed code is STILL latched onto the next
 *    D packet, so a host that ignores S lines behaves exactly as before. Hosts
 *    that want true onsets should epoch from the S line timestamps.
 *
 *  LINE PROTOCOL (device -> host), one ASCII line each, '\n' terminated
 *  --------------------------------------------------------------------
 *    D,<seq>,<micros>,<marker>,<ch0>[,<ch1>,<ch2>]
 *                          Data sample. seq uint32, micros uint32, marker
 *                          uint16 (0 = none), chN int16 raw ADC counts.
 *                          FORMAT IS FROZEN — the host decoder depends on it.
 *    S,<seq>,<onset_us>,<code>,<arm_latency_us>
 *                          Optical stimulus onset detected by the photodiode.
 *                          onset_us is the ISR timestamp of the true screen
 *                          transition; arm_latency_us is the interval from the
 *                          host's arm command to that edge.
 *    H,P300_EEG,<rate_hz>,<n_channels>,<fw_version>
 *                          Stream header, emitted once at boot.
 *    I,<key>=<value>[,...] Configuration / status readout.
 *    M,<code>,<micros>     Acknowledgement that a marker was armed.
 *    T,<key>=<value>[,...] Timing self-test result (see 'T' command).
 *    Z,<ch>,<mean_counts>,<offset_uv_input_referred>
 *                          DC baseline / electrode offset per channel.
 *    P,<key>=<value>[,...] Photodiode diagnostics.
 *    E,<reason>            Error.
 *
 *  HOST COMMANDS (host -> device), '\n' terminated
 *  -----------------------------------------------
 *    M<code>   Arm marker <code> for the next optical onset (and next sample).
 *    T         Run the timing self-test and report the ACHIEVED sample rate.
 *    Z         Run the DC baseline routine (electrode offset characterisation).
 *    P         Report photodiode diagnostics.
 *    I         Report configuration.
 *
 *  KNOWN LIMITATION — INTER-CHANNEL SKEW (disclose in the methodology)
 *  -------------------------------------------------------------------
 *    The ADS1115 has ONE converter behind an input multiplexer, so Fz, Cz and
 *    Pz are NOT sampled simultaneously. Each single-ended conversion at
 *    860 SPS costs ~1.16 ms, so Pz lags Fz by ~2.3 ms in a 3-channel frame.
 *    That skew is a systematic distortion of measured P300 latency across
 *    channels and is not corrected here. The 'T' self-test reports the
 *    measured per-channel conversion time so the true skew can be stated
 *    rather than assumed.
 *
 *  SAMPLE RATE IS MEASURED, NOT DECLARED
 *  -------------------------------------
 *    SAMPLE_RATE_HZ is a compile-time *request*. With three channels the
 *    ADS1115 runs near its aggregate ceiling (860 SPS / 3 = 287 SPS in theory,
 *    and roughly 87% of that in practice once I2C overhead and serial TX are
 *    included). Run the 'T' self-test and report the ACHIEVED rate; do not
 *    quote 250 Hz on the strength of this #define.
 * =============================================================================
 */

#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ----------------------------- Compile-time config ---------------------------
#define FW_VERSION        2          // bumped: photodiode sync + self-test
#define NUM_CHANNELS      3          // 1 (Pz) or 3 (Fz, Cz, Pz)
#define SAMPLE_RATE_HZ    250UL      // REQUESTED rate; verify with 'T'
#define SERIAL_BAUD       115200UL
#define I2C_CLOCK_HZ      400000UL   // fast-mode I2C

#define SYNC_PIN          13         // on-board LED pulses on each latched marker
#define PHOTODIODE_PIN    2          // LM393 comparator output -> INT0

// Total analogue gain ahead of the ADC (INA118P 23.73 x TL072 40).
#define AMPLIFIER_GAIN    949.0f
// ADS1115 GAIN_ONE full scale, and counts at that setting.
#define ADC_FULLSCALE_V   4.096f
#define ADC_FULLSCALE_CNT 32768.0f

// Reject comparator edges closer together than this (contact bounce / ringing
// on the LM393 output). A flash lasts 100 ms and the ISI is 75 ms, so 20 ms is
// comfortably below the shortest real inter-onset interval.
#define PHOTO_DEBOUNCE_US 20000UL

// Self-test sizing.
#define SELFTEST_SAMPLES  500        // frames used to measure the achieved rate
#define BASELINE_SAMPLES  250        // frames averaged for the DC baseline

static const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

// ----------------------------- Globals ---------------------------------------
Adafruit_ADS1115 ads;                // 16-bit ADC

// --- marker state shared with the photodiode ISR ---
volatile uint16_t      armedMarker    = 0;      // identity, supplied by host
volatile bool          markerArmed    = false;
volatile unsigned long armedAtUs      = 0;      // when the host armed it

volatile uint16_t      onsetCode      = 0;      // latched optical onset
volatile unsigned long onsetUs        = 0;
volatile unsigned long onsetArmLatency = 0;
volatile bool          onsetPending   = false;
volatile unsigned long lastEdgeUs     = 0;      // debounce reference
volatile uint32_t      edgeCount      = 0;      // total comparator edges seen

uint16_t          pendingMarker = 0;  // legacy: latched onto the next D packet
uint32_t          sampleSeq     = 0;  // monotonic sample counter
unsigned long     nextSampleUs  = 0;  // scheduled time of the next sample

char    rxBuffer[16];                 // inbound command assembly buffer
uint8_t rxLen = 0;

// ----------------------------- Photodiode ISR --------------------------------

/**
 * Fires on the rising edge of the LM393 comparator output, i.e. the instant the
 * screen actually goes bright. This is the only timestamp in the system that is
 * free of USB transit and host scheduler jitter.
 *
 * Kept deliberately minimal: timestamp, debounce, latch, return. All reporting
 * happens in the main loop.
 */
void photodiodeISR() {
  unsigned long now = micros();
  if ((unsigned long)(now - lastEdgeUs) < PHOTO_DEBOUNCE_US) {
    return;  // bounce / ringing on the comparator output
  }
  lastEdgeUs = now;
  edgeCount++;

  if (markerArmed) {
    onsetCode       = armedMarker;
    onsetUs         = now;
    onsetArmLatency = (unsigned long)(now - armedAtUs);
    onsetPending    = true;
    markerArmed     = false;   // one optical onset consumes one armed code
  }
}

// ----------------------------- Helpers ---------------------------------------

/**
 * Read all NUM_CHANNELS from the ADS1115 into the provided buffer as signed
 * 16-bit counts. Single-ended reads on A0..A(NUM_CHANNELS-1).
 *
 * NOTE: these conversions are SEQUENTIAL, not simultaneous. See the
 * inter-channel skew note in the file header.
 */
static inline void readChannels(int16_t *out) {
  for (uint8_t ch = 0; ch < NUM_CHANNELS; ++ch) {
    out[ch] = ads.readADC_SingleEnded(ch);
  }
}

/** Convert raw ADC counts to input-referred microvolts (divides out the gain). */
static inline float countsToInputUv(float counts) {
  return (counts * (ADC_FULLSCALE_V / ADC_FULLSCALE_CNT) * 1.0e6f)
         / AMPLIFIER_GAIN;
}

/**
 * Emit one ASCII data packet for the current sample. FORMAT IS FROZEN.
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

/** Emit any optical onset latched by the ISR since the last call. */
static void emitPendingOnset() {
  bool pending;
  uint16_t code;
  unsigned long us, latency;

  noInterrupts();
  pending = onsetPending;
  code    = onsetCode;
  us      = onsetUs;
  latency = onsetArmLatency;
  onsetPending = false;
  interrupts();

  if (!pending) {
    return;
  }
  Serial.print("S,");
  Serial.print(sampleSeq);
  Serial.print(',');
  Serial.print(us);
  Serial.print(',');
  Serial.print(code);
  Serial.print(',');
  Serial.print(latency);
  Serial.print('\n');
}

/** Report the static configuration so the host can verify what it is talking to. */
static void reportInfo() {
  Serial.print("I,fw=");           Serial.print(FW_VERSION);
  Serial.print(",channels=");      Serial.print(NUM_CHANNELS);
  Serial.print(",requested_hz=");  Serial.print(SAMPLE_RATE_HZ);
  Serial.print(",i2c_hz=");        Serial.print(I2C_CLOCK_HZ);
  Serial.print(",gain=");          Serial.print(AMPLIFIER_GAIN, 1);
  Serial.print(",fullscale_v=");   Serial.print(ADC_FULLSCALE_V, 3);
  Serial.print(",adc_sps=860");
  Serial.print(",photodiode_pin="); Serial.print(PHOTODIODE_PIN);
  Serial.print('\n');
}

/** Report photodiode / comparator diagnostics. */
static void reportPhotodiode() {
  uint32_t edges;
  unsigned long last;
  noInterrupts();
  edges = edgeCount;
  last  = lastEdgeUs;
  interrupts();

  Serial.print("P,level=");     Serial.print(digitalRead(PHOTODIODE_PIN));
  Serial.print(",edges=");      Serial.print(edges);
  Serial.print(",last_us=");    Serial.print(last);
  Serial.print(",debounce_us="); Serial.print(PHOTO_DEBOUNCE_US);
  Serial.print(",armed=");      Serial.print(markerArmed ? 1 : 0);
  Serial.print('\n');
}

/**
 * Measure the sample rate the hardware can ACTUALLY sustain, plus the
 * per-channel conversion time that sets the inter-channel skew.
 *
 * Runs the acquisition inner loop flat out — no pacing, no serial output — for
 * SELFTEST_SAMPLES frames, then reports. This is the number to quote in the
 * methodology; SAMPLE_RATE_HZ is only a request.
 */
static void runTimingSelfTest() {
  int16_t ch[NUM_CHANNELS];

  // Aggregate: full frames of NUM_CHANNELS conversions.
  unsigned long t0 = micros();
  for (uint16_t i = 0; i < SELFTEST_SAMPLES; ++i) {
    readChannels(ch);
  }
  unsigned long frameElapsed = micros() - t0;

  // Single channel, for the per-conversion cost behind the skew figure.
  t0 = micros();
  for (uint16_t i = 0; i < SELFTEST_SAMPLES; ++i) {
    ch[0] = ads.readADC_SingleEnded(0);
  }
  unsigned long convElapsed = micros() - t0;

  float frameUs = (float)frameElapsed / (float)SELFTEST_SAMPLES;
  float convUs  = (float)convElapsed  / (float)SELFTEST_SAMPLES;
  float achieved = 1.0e6f / frameUs;
  // Pz lags Fz by (NUM_CHANNELS - 1) conversions in every frame.
  float skewUs = convUs * (float)(NUM_CHANNELS - 1);

  Serial.print("T,samples=");        Serial.print(SELFTEST_SAMPLES);
  Serial.print(",frame_us=");        Serial.print(frameUs, 1);
  Serial.print(",achieved_hz=");     Serial.print(achieved, 2);
  Serial.print(",requested_hz=");    Serial.print(SAMPLE_RATE_HZ);
  Serial.print(",conv_us=");         Serial.print(convUs, 1);
  Serial.print(",interchannel_skew_us="); Serial.print(skewUs, 1);
  Serial.print(",sustainable=");
  Serial.print(achieved >= (float)SAMPLE_RATE_HZ ? 1 : 0);
  Serial.print('\n');

  // Re-anchor pacing: the self-test monopolised the ADC for a while.
  nextSampleUs = micros();
}

/**
 * Measure the DC level of each channel for electrode-offset characterisation.
 *
 * Reported both as raw counts and as input-referred microvolts. A large offset
 * matters: stage 1 has a gain of 23.73, so tens of millivolts of electrode
 * offset can saturate the INA118P before the 0.48 Hz high-pass can act on it.
 * Run this with the electrodes on the head, before calibration.
 */
static void runDcBaseline() {
  int32_t sum[NUM_CHANNELS];
  int16_t ch[NUM_CHANNELS];

  for (uint8_t c = 0; c < NUM_CHANNELS; ++c) {
    sum[c] = 0;
  }
  for (uint16_t i = 0; i < BASELINE_SAMPLES; ++i) {
    readChannels(ch);
    for (uint8_t c = 0; c < NUM_CHANNELS; ++c) {
      sum[c] += (int32_t)ch[c];
    }
  }
  for (uint8_t c = 0; c < NUM_CHANNELS; ++c) {
    float meanCounts = (float)sum[c] / (float)BASELINE_SAMPLES;
    Serial.print("Z,");
    Serial.print(c);
    Serial.print(',');
    Serial.print(meanCounts, 1);
    Serial.print(',');
    Serial.print(countsToInputUv(meanCounts), 2);
    Serial.print('\n');
  }
  nextSampleUs = micros();
}

/**
 * Drain the serial input and dispatch any complete host command line.
 *
 * Arming a marker records the arm time so the photodiode ISR can report how
 * long the host->flash path actually took. Multiple arms before an optical
 * onset collapse to the most recent (a flash never overlaps another flash).
 */
static void pollHostCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen > 0) {
        rxBuffer[rxLen] = '\0';
        switch (rxBuffer[0]) {
          case 'M': {
            long m = atol(&rxBuffer[1]);
            if (m >= 0 && m <= 65535L) {
              unsigned long now = micros();
              noInterrupts();
              armedMarker = (uint16_t)m;
              markerArmed = true;
              armedAtUs   = now;
              interrupts();
              pendingMarker = (uint16_t)m;   // legacy next-sample latch
              digitalWrite(SYNC_PIN, HIGH);
              Serial.print("M,");
              Serial.print(m);
              Serial.print(',');
              Serial.print(now);
              Serial.print('\n');
            }
            break;
          }
          case 'T': runTimingSelfTest(); break;
          case 'Z': runDcBaseline();     break;
          case 'P': reportPhotodiode();  break;
          case 'I': reportInfo();        break;
          default:  break;               // unknown command -> ignore silently
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

// ----------------------------- Arduino lifecycle -----------------------------

void setup() {
  pinMode(SYNC_PIN, OUTPUT);
  digitalWrite(SYNC_PIN, LOW);

  // LM393 modules drive their output actively; INPUT_PULLUP is harmless with an
  // open-collector variant and keeps the line defined if the module is absent.
  pinMode(PHOTODIODE_PIN, INPUT_PULLUP);

  Serial.begin(SERIAL_BAUD);
  while (!Serial) { ; }  // wait for native-USB boards; harmless on the Uno

  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);  // 400 kHz fast-mode I2C

  // GAIN_ONE -> +/-4.096 V full scale (matches adc_fullscale_v in config.yaml).
  ads.setGain(GAIN_ONE);

  if (!ads.begin()) {
    // Without the ADC there is nothing to stream; report and halt.
    Serial.println("E,ADS1115_NOT_FOUND");
    while (true) { digitalWrite(SYNC_PIN, HIGH); delay(100);
                   digitalWrite(SYNC_PIN, LOW);  delay(100); }
  }

  // 860 SPS data rate -> ~1.2 ms / conversion, the ceiling this design runs at.
  ads.setDataRate(RATE_ADS1115_860SPS);

  attachInterrupt(digitalPinToInterrupt(PHOTODIODE_PIN), photodiodeISR, RISING);

  // Announce the stream header so the host can auto-detect channel count.
  Serial.print("H,P300_EEG,");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.print(',');
  Serial.print(NUM_CHANNELS);
  Serial.print(',');
  Serial.print(FW_VERSION);
  Serial.print('\n');
  reportInfo();

  nextSampleUs = micros();
}

void loop() {
  // Service inbound commands as often as possible to minimise arm latency.
  pollHostCommands();

  // Report any optical onset the ISR latched since the last iteration.
  emitPendingOnset();

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
