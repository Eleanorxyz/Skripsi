#include <Wire.h>
#include "MAX30105.h"
#include <ArduTFLite.h>
#include "heartRate.h"
#include <math.h>
#include <cmath>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "GoogleSheetsLogger.h"
#include "credentials.h"

// =============== HARDWARE CONFIGURATION ===============
#define I2C_SDA 5
#define I2C_SCL 6
#define BUZZER_PIN 7
#define HR_AVG_WINDOW 6
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

// =============== SYSTEM CONFIGURATION ===============
#define DEBUG_MODE true  // Set false untuk production
#define SAFETY_CHECKS true
#define HEALTH_MONITORING true
#define ADAPTIVE_LEARNING true

// =============== DEKLARASI FUNGSI VISUALISASI ===============
float scaleToBPMRange(float value, float minRaw, float maxRaw);
float generateHeartBeatWave(float bpm, float& phase);
void initSerialPlotter();
void updateSerialPlotter(bool isPeak = false);
void markPeakForPlotter(bool isPeak);

// Variabel global untuk visualisasi
unsigned long lastWaveUpdate = 0;

// =============== EMERGENCY RECOVERY COOLDOWN ===============
unsigned long lastEmergencyRecovery = 0;
const unsigned long RECOVERY_COOLDOWN_MS = 3000;   // 3 detik cooldown - ✅ OPTIMAL
const unsigned long RECOVERY_TIMEOUT_MS = 2000;    // 2 detik timeout - ✅ OPTIMAL  
bool recoveryInProgress = false;
float lastStableHR = 72.0f;  // Nilai default yang lebih baik

// =============== KALMAN FILTER PARAMETERS ===============
float processNoise = 4.0f;           // ↑ Lebih responsive terhadap perubahan
float measurementNoise = 1.2f;       // ↓ Lebih percaya measurement
float signalRange = 0.01f;           // Initialize with minimum range

// =============== MOTION TRACKING ===============
unsigned long lastMotionTime = 0;    // For stillness detection

// =============== ENHANCED SYSTEM VARIABLES ===============
int systemHealthScore = 100;  // 0-100, 100 = sehat sempurna
int consecutiveErrors = 0;
float lastGoodThreshold = 0.05f;
unsigned long lastHealthCheck = 0;
uint32_t totalPeaksDetected = 0;
uint32_t totalPeaksExpected = 0;

// =============== SYSTEM PARAMETERS ===============
const int INPUT_SIZE = 32;
const int SAMPLE_RATE = 64;
const int BUFFER_SIZE = 256;
const int STEP_SIZE_RUN = 8;    // 0.125s overlap (running)
const int STEP_SIZE_NORMAL = 16; // 0.25s overlap (normal)
const int SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE;

// Thresholds
const float HR_ALERT_HIGH = 160.0;
const float HR_ALERT_LOW = 50.0;
const float HIGH_INTENSITY_THRESHOLD = 200.0f;
const float HIGH_INTENSITY_FACTOR = 0.4f;
float contactThreshold = 0;
const int MAX_CONSECUTIVE_INVALID = 5;
const unsigned long CALIBRATION_TIMEOUT = 30000;

// User-adjustable parameters
#define MIN_STDDEV 0.05f
#define USER_FITNESS_LEVEL 0.8
#define FALLBACK_PEAK_TIMEOUT_MS 1200
const float RUNNING_COMPENSATION = 1.3;
const float WALKING_COMPENSATION = 1.05;
const float HR_CORRECTION_FACTOR = 0.96f;
const float MIN_PEAK_DISTANCE_MS = 25.0f;

// ============= Google Sheet Logger ==============
GoogleSheetsLogger dataLogger(GOOGLE_SCRIPT_URL, WIFI_SSID, WIFI_PASSWORD);

// Variables untuk tracking changes
float lastRedValue = 0;
float lastIRValue = 0;
unsigned long lastLogTime = 0;
bool sheetsLoggerInitialized = false;

// =============== GLOBAL VARIABLES ===============
// Hardware Objects
MAX30105 particleSensor;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Data Processing
float ppgBuffer[BUFFER_SIZE];
float lastProcessingTimeMs = 0;
int currentStepSize = STEP_SIZE_NORMAL;
bool isRunningMode = false;
bool isVeryStill = false;
int bufferIndex = 0;
bool bufferFull = false;
float dcFiltered = 0;
float instantHR = 0;
float motionLevel;                // ← Tidak ada nilai default
float redMovingAvg;               // ← Tidak ada nilai default
float irMovingAvg;                // ← Tidak ada nilai default
float contactBaseline = 0;
float dynamicFactor = 0;
bool isPeak = false;
float minPeakDistance = 0;
int peakCountLastSecond = 0;

// System State
float heartRate = NAN;
float hrBuffer[HR_AVG_WINDOW] = {0};
int hrIndex = 0;
const byte RATE_SIZE = 4; 
byte rates[RATE_SIZE];
byte rateSpot = 0;
int beatAvg = 0;
String heartStatus = "Initializing...";
bool sensorContactLost = true;
bool needsCalibration = true;
bool alertActive = false;
float stillnessCounter = 0;
float lastModeSwitchTime = 0;
float adaptiveSmoothFactor = 0.9f;
float adaptiveMotionScale = 30000.0f;
int adaptiveStepSize = STEP_SIZE_NORMAL;
float adaptiveModelWeight = 0.7f;
float adaptiveInstantWeight = 0.3f;
int consecutiveInvalidReadings = 0;
unsigned long lastContactCheck = 0;
unsigned long lastContactTime = 0;
unsigned long contactLostTime = 0;
unsigned long lastAlertTime = 0;
unsigned long lastDisplayUpdate = 0;
unsigned long lastDebugTime = 0;
unsigned long lastSampleTime = 0;
unsigned long lastInferenceTime = 0;
enum SystemStatus { STATUS_OK, ERROR_SENSOR };  // Add enum definition
SystemStatus systemStatus = STATUS_OK;  // Add with other global variables

// TFLite Model
#include "hr_model_optimized_hope.h"
#define TENSOR_ARENA_SIZE 5372 * 1024
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;
uint8_t* tensor_arena = nullptr;

// Peak Detection
float lastValues[7] = {0};
bool rising = false;
unsigned long lastPeakTime = 0;
float dynamicThreshold = 0.5f;

// Kalman Filter
float lastValidHR = 72.0f;
float estimateError = 5.0f;

// Motion context structure (add this definition)
struct MotionContext {
    bool isActive() { 
        return motionLevel > 5000.0f; // Example threshold, adjust as needed
    }
} motionContext;

// Constants (add with other #defines)
#define FILTER_WINDOW_SIZE 7  // Or whatever size you need

// Add this with other peak detection variables
unsigned long nextExpectedPeak = 0;

// =============== FUNCTION PROTOTYPES ===============
bool initializeHardware();
void updateDisplay(const char* line1 = nullptr, const char* line2 = nullptr);
void displayError(const char* line1, const char* line2);
void readPPGSamples();
void checkPeakDetection();
void processInference();
void applyHighHROptimizations();
void applyMotionArtifactReduction();
void checkSensorContact();
void checkTemperatureCondition(float temp);
void checkHeartRateCondition(float hr);
void handleAlerts();
void printSystemStatus();
float calculateSignalVariance();
float findMax(float* arr, int n);
float findMin(float* arr, int n);
void resetFilters(bool isPostCalibration = false);
void showCalibrationScreen();
bool performInitialCalibration();
bool performRecalibration();
void handleMissingPeaks();
void checkSystemHealth();
void updateDynamicThresholdLearning();
void enhancedDebugOutput();
float validateHeartRate(float hr);
void updateAdaptiveParameters();

// =============== CORE SYSTEM FUNCTIONS ===============
void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  // Initialize Google Sheets logger (non-blocking)
  if (strlen(WIFI_SSID) > 0 && strlen(WIFI_PASSWORD) > 0) {
    sheetsLoggerInitialized = dataLogger.begin();
    if (sheetsLoggerInitialized) {
      Serial.println("Google Sheets logger initialized");
    } else {
      Serial.println("Google Sheets logger failed - will retry later");
    }
  } else {
    Serial.println("WiFi credentials not set - skipping Google Sheets");
  }

  // Initialize Hardware
  if (!initializeHardware()) {
    while(1);
  }

  // Initialize heart rate variables with realistic defaults
  instantHR = 0;
  lastValidHR = 0;
  heartRate = 0;

  lastPeakTime = 0;

  // Initialize peak detection buffer
  for (int i = 0; i < 7; i++) {
    lastValues[i] = 0.5f;  // Mid value initialization
  }

  initSerialPlotter();
  showCalibrationScreen();
  performInitialCalibration();
  updateDisplay("System Ready", "Place Finger");
  Serial.println("System initialized - waiting for real data...");
}

void loop() {
  //==============================================
  // 0. Grace Period untuk Inisialisasi (TAMBAH INI)
  //==============================================
  static unsigned long startupTime = millis();
  if (millis() - startupTime < 10000) { // 10 detik grace period
    // Selama grace period, update display saja
    if (millis() - lastDisplayUpdate >= 1000) {
      updateDisplay("Initializing", "Please wait...");
      lastDisplayUpdate = millis();
    }
    // TAMBAH INI: Initialize WiFi selama grace period
    if (millis() > 5000 && !sheetsLoggerInitialized) {
      sheetsLoggerInitialized = dataLogger.begin();
    }
    return; // Skip processing lainnya selama grace period
  }

  //==============================================
  // 1. Inisialisasi Variabel Waktu
  //==============================================
  //static unsigned long lastSampleTime = 0;
  //static unsigned long lastInferenceTime = 0;
  //static unsigned long lastDisplayUpdate = 0;
  //static unsigned long lastDebugTime = 0;
  static unsigned long lastStillnessCheck = 0;
  //static unsigned long lastModeSwitchTime = 0;
  unsigned long currentTime = millis();

  //==============================================
  // 2. State Machine Controller
  //==============================================
  enum SystemState { 
    SAMPLE_PPG, 
    PROCESS_DATA, 
    UPDATE_DISPLAY, 
    CHECK_STILLNESS,
    HANDLE_ALERTS
  };
  
  static SystemState state = SAMPLE_PPG;

  switch (state) {
    //--------------------------------------------
    // Case 1: Pembacaan Sinyal PPG
    //--------------------------------------------
    case SAMPLE_PPG:
      if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        if (!sensorContactLost) {
          readPPGSamples();
          checkPeakDetection();
          applyMotionArtifactReduction();
        }
        lastSampleTime = currentTime;
        state = PROCESS_DATA;
      }
      break;

    //--------------------------------------------
    // Case 2: Pemrosesan Data dan Inferensi
    //--------------------------------------------
    case PROCESS_DATA:
      if (currentTime - lastInferenceTime >= 800) {
        if (!sensorContactLost) {
          
          // ==== TAMBAHKAN DEBUG INFO ====
          static unsigned long waitingSince = 0;
          if (!bufferFull) {
            if (waitingSince == 0) {
              waitingSince = currentTime;
              Serial.printf("WAITING FOR BUFFER: %d/%d samples\n", 
                          bufferIndex, BUFFER_SIZE);
            }
            else if (currentTime - waitingSince > 5000) {
              // Force processing setelah 5 detik menunggu
              Serial.printf("TIMEOUT: Using available data %d/%d samples\n",
                          bufferIndex, BUFFER_SIZE);
              bufferFull = true; // Force flag
            }
          }
          // ==============================

          if (bufferFull) {
            // Optimasi khusus untuk aktivitas tinggi
            if (heartRate > 160 || motionLevel > 300000) {
              applyHighHROptimizations();
            }
            
            processInference();
            checkHeartRateCondition(heartRate);
            
            // ===== INTEGRASI KODE BARU =====
            checkSystemHealth();
            updateDynamicThresholdLearning();
            enhancedDebugOutput();
            // ===============================
            
            waitingSince = 0; // Reset waiting timer
          }
        }
        lastInferenceTime = currentTime;
        state = UPDATE_DISPLAY;
      }
      break;

    //--------------------------------------------
    // Case 3: Pembaruan Tampilan
    //--------------------------------------------
    case UPDATE_DISPLAY:
      if (currentTime - lastDisplayUpdate >= 250) {
        updateDisplay();
        logDataToGoogleSheets();
        lastDisplayUpdate = currentTime;
        state = CHECK_STILLNESS;
      }
      break;

    //--------------------------------------------
    // Case 4: Deteksi Diam
    //--------------------------------------------
    case CHECK_STILLNESS:
      // Deteksi stillness setiap 100ms
      if (currentTime - lastStillnessCheck >= 100) {
          checkStillness(); // PANGGIL FUNCTION YANG BARU
          lastStillnessCheck = currentTime;
          
          // =============== ADAPTIVE MOTION CALIBRATION ===============
          // Auto-calibration berdasarkan motion level
          if (motionLevel > 50000.0f && motionLevel < 100000.0f) {
              // Sedang dalam high motion, adjust parameters
              adaptiveMotionScale = 80000.0f + (motionLevel - 50000.0f) * 0.8f;
          }
          else if (motionLevel > 100000.0f) {
              // Extreme motion, clamp values
              adaptiveMotionScale = 120000.0f;
              motionLevel = min(motionLevel, 150000.0f); // Soft clamp
          }
          // ===========================================================
          
          state = HANDLE_ALERTS;
      }
      break;

    //--------------------------------------------
    // Case 5: Penanganan Alarm
    //--------------------------------------------
    case HANDLE_ALERTS:
      handleAlerts();
      
      // Debug output setiap 1 detik
      if (currentTime - lastDebugTime >= 1000) {
        printSystemStatus();
        lastDebugTime = currentTime;
      }
      
      state = SAMPLE_PPG;
      break;
  }

  //==============================================
  // 3. Penanganan Kontak Sensor
  //==============================================
  static unsigned long lastContactCheck = 0;
  if (currentTime - lastContactCheck >= 500) {
    checkSensorContact();
    lastContactCheck = currentTime;
  }

  //==============================================
  // ✅ PERBAIKAN KRITIS: EMERGENCY PROCESSING
  //==============================================
  static unsigned long lastEmergencyProcess = 0;
  if (bufferFull && millis() - lastInferenceTime > 3000 && 
      millis() - lastEmergencyProcess > 1000) {
    processInference();
    lastEmergencyProcess = millis();
    lastInferenceTime = millis();
    Serial.println("EMERGENCY: Manual inference triggered");
  }

  //==============================================
  // 4. Adaptive Delay untuk Konsistensi Timing
  //==============================================
  static unsigned long lastLoopTime = 0;
  unsigned long loopDuration = currentTime - lastLoopTime;
  if (loopDuration < 2) {  // Pertahankan minimum 2ms delay
    delay(2 - loopDuration);
  }
  lastLoopTime = millis();
  
  // ===== INTEGRASI KODE BARU =====
  // Safety check tambahan - ONLY AFTER GRACE PERIOD
  static unsigned long lastSafetyCheck = 0;
  if (millis() - lastSafetyCheck > 1000 && millis() - startupTime > 10000) {
    heartRate = validateHeartRate(heartRate);
    lastSafetyCheck = millis();
  }
  // ===============================
}

// =============== SIGNAL PROCESSING FUNCTIONS ===============
void readPPGSamples() {
  // ✅ 1. PEMBACAAN SENSOR & SAFETY CHECK
  float redValue = particleSensor.getRed();
  float irValue = particleSensor.getIR();

  // ✅ 2. CHECK SENSOR CONTACT FIRST! (PALING PRIORITAS)
  if (sensorContactLost) {
    return; // Jangan proses sama sekali jika tidak ada kontak
  }

  // ✅ 3. SAFETY CHECK BERDASARKAN RAW VALUES
  const float MIN_RED = 10000.0f;    // Minimum untuk kontak
  const float MIN_IR = 8000.0f;      // Minimum untuk kontak  
  const float MAX_VALUE = 250000.0f; // Maksimum valid

  if (redValue < MIN_RED || redValue > MAX_VALUE || 
      irValue < MIN_IR || irValue > MAX_VALUE) {
    if (!sensorContactLost) {
      sensorContactLost = true;
      heartRate = NAN;
      instantHR = NAN;
      heartStatus = "No Contact";
      Serial.printf("SAFETY_TRIGGER: Red=%.0f IR=%.0f\n", redValue, irValue);
    }
    return;
  }

  // ✅ 4. QUALITY CHECK DENGAN VERSI OPTIMAL
  float signalQuality = calculateSignalQuality(redValue, irValue);
  
  // Adaptive threshold berdasarkan buffer readiness
  float qualityThreshold;
  if (motionLevel > 10000.0f) {
      qualityThreshold = 0.4f;  // Sangat tolerant saat high motion
  } else if (motionLevel > 5000.0f) {
      qualityThreshold = 0.3f;  // Tolerant untuk medium motion
  } else if (bufferIndex < 5) {
      qualityThreshold = 0.15f; // Longgar saat startup
  } else {
      qualityThreshold = 0.25f; // Default
  }
  
  if (signalQuality < qualityThreshold) {
      // ✅ SOFT SUPPRESSION (bukan hard skip)
      motionLevel = 0.8f * motionLevel; // Gentle suppression
      
      static unsigned long lastQualityWarning = 0;
      if (millis() - lastQualityWarning > 2000) {
          Serial.printf("QUALITY_SUPPRESSION: %.2f/%.2f\n", signalQuality, qualityThreshold);
          lastQualityWarning = millis();
      }
      // ✅ TIDAK ADA return! Processing tetap lanjut
  }

  // ==================== STEP 2: NATURAL MOTION CALCULATION ====================
  updateAdaptiveParameters(); // ✅ Sudah decoupled!
  static float lastRedValue = 0;
  static float lastIRValue = 0;

  // 1. Hitung perubahan sinyal
  float redChange = abs(redValue - lastRedValue);
  float irChange = abs(irValue - lastIRValue);

  // ✅ PRIORITAS 3: OPTIMASI MOTION CALCULATION - LEBIH RESPONSIVE
  float irMotionComponent = irChange * 0.8f;
  float redMotionComponent = redChange * 1.2f;
  float baseMotion = (redMotionComponent + irMotionComponent) * 0.5f;

  // 2. Moving average update
  redMovingAvg = adaptiveSmoothFactor * redMovingAvg + (1.0f - adaptiveSmoothFactor) * redValue;
  irMovingAvg = adaptiveSmoothFactor * irMovingAvg + (1.0f - adaptiveSmoothFactor) * irValue;

  // ✅ 3. NATURAL MOTION CALCULATION - LEBIH AGGRESSIVE
  float velocityComponent = (redChange + irChange) * 0.002f; // ↑ 4x dari original

  // ✅ 4. MOTION BOOST FACTOR - LEBIH RESPONSIVE (PRIORITAS 3)
  float motionBoost = 1.0f;
  if (baseMotion > 100.0f) motionBoost = 3.0f;       // Very sensitive
  if (baseMotion > 300.0f) motionBoost = 5.0f;       // Walking
  if (baseMotion > 800.0f) motionBoost = 8.0f;       // Running ← OPTIMAL
  if (baseMotion > 1500.0f) motionBoost = 12.0f;     // Extreme motion

  // ✅ 5. NON-LINEAR COMPENSATION (PRIORITAS 3)
  float nonLinearCompensation = 1.0f;
  if (baseMotion > 1000.0f) {
      nonLinearCompensation = 0.8f;  // Mild compression
  } else if (baseMotion < 200.0f) {
      nonLinearCompensation = 1.4f;  // Mild expansion
  }

  // ✅ APLIKASIKAN:
  float naturalMotion = (baseMotion + velocityComponent) * motionBoost * nonLinearCompensation;

  // ✅ 6. MOTION SPIKE FILTER (PRIORITAS 3)
  static float lastNaturalMotion = 0;
  float motionChange = abs(naturalMotion - lastNaturalMotion);
  
  if (motionChange > 40000.0f) {  // Filter extreme spikes
      naturalMotion = (lastNaturalMotion * 0.6f + naturalMotion * 0.4f);
      Serial.println("MOTION_SPIKE: Filtered extreme change");
  }
  lastNaturalMotion = naturalMotion;

  // ✅ 7. ADAPTIVE SMOOTHING - LEBIH PINTAR (PRIORITAS 3)
  float motionIntensity = constrain(motionLevel / 8000.0f, 0.0f, 1.0f);
  float inertiaSmooth = 0.50f - 0.25f * motionIntensity;  // 0.50 → 0.25
  float levelSmooth = 0.70f - 0.20f * motionIntensity;    // 0.70 → 0.50

  static float motionInertia = 0.0f;
  motionInertia = inertiaSmooth * motionInertia + (1.0f - inertiaSmooth) * naturalMotion;
  motionLevel = levelSmooth * motionLevel + (1.0f - levelSmooth) * motionInertia;

  // ✅ 8. WEIGHTED BOOST UNTUK HIGH MOTION
  if (naturalMotion > 12000.0f) {
    motionLevel = 0.45f * motionLevel + 0.55f * naturalMotion;
    Serial.println("HIGH_MOTION: Aggressive boost applied");
  }

  // ✅ 9. FAST DECAY UNTUK TRANSISI CEPAT
  if (naturalMotion < 800.0f && motionLevel > 4000.0f) {
    motionLevel = 0.60f * motionLevel + 0.40f * naturalMotion;
    Serial.println("FAST_DECAY: Accelerated motion reduction");
  }

  // ✅ 10. SAFETY LIMIT
  motionLevel = constrain(motionLevel, 0.0f, 150000.0f); // ↑ Increased max limit

  // ✅ 11. DEBUG OUTPUT UNTUK PRIORITAS 3
  static unsigned long lastMotionDebug = 0;
  if (millis() - lastMotionDebug > 1000) {
    Serial.printf("MOTION_OPTIMIZED: RedΔ=%.0f IRΔ=%.0f Base=%.0f Boost=%.1f Comp=%.1f Final=%.0f\n",
                 redChange, irChange, baseMotion, motionBoost, nonLinearCompensation, motionLevel);
    Serial.printf("ADAPTIVE_SMOOTH: Intensity=%.2f Inertia=%.2f Level=%.2f\n",
                 motionIntensity, inertiaSmooth, levelSmooth);
    lastMotionDebug = millis();
  }

  lastRedValue = redValue;
  lastIRValue = irValue;
  // =============================================================================

  // ✅ 5. SENSOR CONTACT DETECTION YANG LEBIH BAIK
  const float CONTACT_THRESHOLD_ABS = 50000.0f; // Threshold fixed
  
  if (redValue < CONTACT_THRESHOLD_ABS) {
    if (!sensorContactLost && ++consecutiveInvalidReadings >= 3) {
      sensorContactLost = true;
      heartRate = NAN;
      instantHR = NAN;
      heartStatus = "No Contact";
      contactLostTime = millis();
      Serial.printf("CONTACT_LOST: Red=%.0f below threshold\n", redValue);
    }
    return;
  }

  consecutiveInvalidReadings = 0;
  if (sensorContactLost) {
    sensorContactLost = false;
    heartStatus = "Measuring...";
    Serial.println("CONTACT_REGained: Starting measurement");
    resetFilters(true);
  }

  lastContactTime = millis();

  // ✅ 6. HYBRID AC CALCULATION (DC Removal yang sudah baik)
  static float lastRedValueAC = 0;
    
  // ==== IMPROVED DC REMOVAL ====
  static float dcWindow[100] = {0};
  static uint8_t dcWindowIndex = 0;
  static float dcSum = 0;

  // Adaptive DC Baseline Tracking
  static float redDCBaseline = 100000.0f;
  static float irDCBaseline = 110000.0f;

  // Update DC window
  dcSum -= dcWindow[dcWindowIndex];
  dcWindow[dcWindowIndex] = redValue;
  dcSum += dcWindow[dcWindowIndex];
  dcWindowIndex = (dcWindowIndex + 1) % 100;

  // DC Adaptation
  float redDCError = redValue - redDCBaseline;
  if (abs(redDCError) > 1500.0f) {
    redDCBaseline += redDCError * 0.01f;
  }

  float irDCError = irValue - irDCBaseline;
  if (abs(irDCError) > 3000.0f) {
    irDCBaseline += irDCError * 0.005f;
  }

  float simpleAC = redValue - redDCBaseline;
  float irSimpleAC = irValue - irDCBaseline;

  // ✅ 7. BANDPASS FILTER DAN PROCESSING
  static float hpFiltered1 = 0, hpFiltered2 = 0;
  float alphaHP1 = 0.93f;
  float alphaHP2 = 0.89f;
    
  hpFiltered1 = alphaHP1 * (hpFiltered1 + redValue - lastRedValueAC);
  hpFiltered2 = alphaHP2 * (hpFiltered2 + hpFiltered1);
    
  float adaptiveAC = (hpFiltered1 - hpFiltered2) * 6.0f;
  
  // Bandpass Filter
  float filteredAC = bandpassFilter(simpleAC) * 1.2f;
  
  // Hybrid Fusion
  float hybridAC;
  float motionBlend = constrain(motionLevel / 40000.0f, 0.0f, 1.0f);
  
  if (motionLevel < 10000) {
    hybridAC = simpleAC * (0.8f - 0.3f * motionBlend) + filteredAC * (0.2f + 0.3f * motionBlend);
  } else {
    hybridAC = filteredAC * (0.7f - 0.4f * motionBlend) + adaptiveAC * (0.3f + 0.4f * motionBlend);
  }

  // Safety checks
  if (isnan(hybridAC) || isinf(hybridAC)) {
    hybridAC = simpleAC;
  }

  hybridAC = constrain(hybridAC, -30000.0f, 30000.0f);
    
  // ✅ 8. DYNAMIC SCALING
  float acRange = 500.0f + 3500.0f * constrain(motionLevel / 60000.0f, 0.0f, 1.0f);
  acRange = constrain(acRange, 500.0f, 4000.0f);
    
  float scaledValue = 0.5f + (hybridAC / (acRange * 2.0f));
  scaledValue = constrain(scaledValue, 0.1f, 0.9f);

  // Motion suppression
  float motionSuppression = 1.0f / (1.0f + pow(motionLevel/70000.0f, 1.2f));
  scaledValue = 0.5f + (scaledValue - 0.5f) * motionSuppression;
    
  // ✅ 9. SIMPAN KE BUFFER
  ppgBuffer[bufferIndex] = scaledValue;
  bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;

  if (bufferIndex == BUFFER_SIZE - 1) {
    bufferFull = true;
  }

  // ✅ 10. DEBUG OUTPUT
  static unsigned long lastDebug = 0;
  if (millis() - lastDebug > 2000) {
    Serial.printf("PPG_FULL: Raw=%.0f DC=%.0f AC=%.1f Mot=%.0f Qual=%.2f\n",
                redValue, redDCBaseline, hybridAC, motionLevel, signalQuality);
    lastDebug = millis();
  }
    
  lastRedValueAC = redValue;
}

// Bandpass filter
const float b0 = 0.0201, b1 = 0.0402, b2 = 0.0201;
const float a1 = -1.561, a2 = 0.6414;

float bandpassFilter(float input) {
  static float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  // Boost yang lebih conservative
  if (abs(input) < 200.0f && input != 0.0f) {
    float boost = 1.0f + (200.0f - abs(input)) / 200.0f * 1.0f; // Max 3x boost
    input *= boost;
  }

  float output = b0*input + b1*x1 + b2*x2 - a1*y1 - a2*y2;
  
  // Stability check
  if (isnan(output) || isinf(output)) {
    output = input;
    Serial.println("BANDPASS: Stability reset");
  }
  
  x2 = x1; x1 = input; y2 = y1; y1 = output;
  
  return output;
}

void checkPeakDetection() {
  // 1. Gunakan struktur SignalData untuk organisasi yang lebih baik
  static struct {
    float values[5];      // Buffer nilai terakhir
    float mean;           // Rata-rata nilai
    float stddev;         // Standar deviasi
    float derivative;     // Turunan sinyal
  } sig;

  // ✅ ACTIVITY TRANSITION DETECTION
    static float lastMotionLevel = 0.0f;
    static unsigned long lastActivityChange = 0;

    float motionChange = abs(motionLevel - lastMotionLevel);
    float motionChangeRate = motionChange / max(1.0f, (millis() - lastActivityChange) / 1000.0f);

    // Jika detect significant activity change
    if (motionChangeRate > 3000.0f) { // Rapid activity change
        // Apply immediate HR adjustment
        float transitionBoost = 1.0f + 0.2f * constrain(motionChangeRate / 10000.0f, 0.0f, 1.0f);
        instantHR *= transitionBoost;
        lastActivityChange = millis();
        
        Serial.printf("ACTIVITY_TRANSITION: Change=%.0f/s, Boost=%.2f, HR=%.1f\n", 
                      motionChangeRate, transitionBoost, instantHR);
    }

    lastMotionLevel = motionLevel;
  
  // Update buffer dan hitung statistik
  int prevIndex = getBufferIndex(-1);
  memmove(sig.values, sig.values + 1, 4 * sizeof(float));
  sig.values[4] = ppgBuffer[prevIndex];
  
  // 2. Hitung parameter dasar dengan perbaikan statistik
  // Hitung mean dan stddev
  sig.mean = 0;
  for (int i = 0; i < 5; i++) sig.mean += sig.values[i];
  sig.mean /= 5;
  
  sig.stddev = 0;
  for (int i = 0; i < 5; i++) {
    sig.stddev += pow(sig.values[i] - sig.mean, 2);
  }
  sig.stddev = sqrt(sig.stddev / 5);
  
  // Hitung turunan dengan buffer yang lebih besar
  sig.derivative = (sig.values[2] - sig.values[0]) / 2.0f;
  
  // Hitung range sinyal
  signalRange = findMax(sig.values, 5) - findMin(sig.values, 5);
  signalRange = max(signalRange, 0.005f);
  
  long currentTime = millis();
  
  // 3. Dynamic threshold yang adaptif berdasarkan motion level
  float motionFactor = constrain(motionLevel / 60000.0f, 0.0f, 1.0f);
  float baseThreshold = sig.mean + 2.5f * sig.stddev;  // Less strict
  dynamicThreshold = baseThreshold * (0.8f + motionFactor * 0.7f);  // Better scaling
  
  // 4. Hybrid Detection: Coba deteksi dengan library MAX30105 terlebih dahulu
  bool libraryBeat = false;
  float libraryHR = 0;
  
  if (checkForBeat(particleSensor.getIR())) {
      long delta = currentTime - lastPeakTime;
      if (delta > MIN_PEAK_DISTANCE_MS) {
          libraryHR = 60000.0f / delta;
          if (libraryHR > 40 && libraryHR < 240) {
              libraryBeat = true;
              instantHR = libraryHR;
              lastPeakTime = currentTime;
              markPeakForPlotter(true);
              
              // Update BPM average
              rates[rateSpot++] = (byte)libraryHR;
              rateSpot %= RATE_SIZE;
              beatAvg = 0;
              for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
              beatAvg /= RATE_SIZE;
              
              Serial.printf("LIBRARY PEAK: %.1f BPM (Avg: %d)\n", libraryHR, beatAvg);
          }
      }
  }
  
  // 5. UNIVERSAL ADAPTIVE PEAK DETECTION (ganti mode-specific)
  bool customPeak = false;
  float customHR = 0;
  
  // Adaptive parameters berdasarkan motion level
  // ✅ LEBIH RESPONSIVE PEAK DETECTION
  float slopeThreshold = 0.0010f + (motionLevel / 80000.0f);  // ↓ Threshold, ↑ Sensitivity
  float minDerivative = 0.0030f + (motionLevel / 150000.0f); // ↓ Minimum derivative

  // ✅ FASTER RESPONSE UNTUK ACTIVITY TRANSITIONS
  float activityChange = abs(motionLevel - lastMotionLevel);
  if (activityChange > 5000.0f) { // Significant activity change
      slopeThreshold *= 0.7f; // ↓ Threshold 30% untuk transisi cepat
      minDerivative *= 0.6f;  // ↓ Minimum 40%
  }

  float noiseFloor = 0.0005f * (1.0f + (motionLevel / 20000.0f));
  
  // Universal peak detection yang bekerja untuk semua kondisi
  if (motionFactor > 0.6f) {
      // =============== HIGH MOTION DETECTION (Zero-Crossing) ===============
      bool isZeroCrossing = (sig.values[1] < 0.5f - dynamicThreshold * 0.3f) && 
                          (sig.values[2] >= 0.5f + dynamicThreshold * 0.3f);
      
      customPeak = isZeroCrossing && 
                  (sig.derivative > minDerivative) &&
                  (abs(sig.values[3] - sig.values[1]) < 0.15f);
      
      // Debug untuk high motion
      if (isZeroCrossing && millis() - lastDebugTime > 500) {
          Serial.printf("HIGH MOTION: ZCross %.3f->%.3f | Der=%.4f (Need %.4f) | Lib:%d | %s\n",
                      sig.values[1], sig.values[2], sig.derivative, minDerivative,
                      libraryBeat, customPeak ? "PEAK" : "NOISE");
      }
  } 
  else if (motionFactor > 0.3f) {
      // =============== MEDIUM MOTION DETECTION (Slope-Based) ===============
      customPeak = (sig.values[2] - sig.values[1]) > slopeThreshold &&
                  (sig.values[2] - sig.values[3]) > (slopeThreshold * 0.6f) &&
                  (sig.values[2] > dynamicThreshold * 0.4f) &&
                  (sig.values[4] < sig.values[2]);
      
      // Debug untuk medium motion
      if (millis() - lastDebugTime > 500) {
          Serial.printf("MEDIUM MOTION: Slope=%.4f (Need %.4f) | Mot=%.1f | Lib:%d | %s\n",
                       (sig.values[2]-sig.values[1]), slopeThreshold,
                       motionLevel, libraryBeat, customPeak ? "PEAK" : "-");
      }
  } 
  else {
      // =============== LOW MOTION DETECTION (High Sensitivity) ===============
      bool isMainPeak = (sig.values[2] - sig.values[1]) > 0.0001f &&
                       (sig.values[2] > dynamicThreshold * 0.08f);
      
      bool isConfirmed = (sig.values[0] < sig.values[2]) && 
                        (sig.values[4] < sig.values[2]) && 
                        (abs(sig.values[1] - sig.values[3]) < 0.005f);
      
      customPeak = isMainPeak && isConfirmed;
      
      // Debug untuk low motion
      if (customPeak && millis() - lastDebugTime > 500) {
          Serial.printf("LOW MOTION: Peak=%.3f (Thr=%.4f) | NoiseFloor=%.4f | Mot=%.1f | Lib:%d\n",
                       sig.values[2], dynamicThreshold, noiseFloor, motionLevel, libraryBeat);
      }
  }
  
  // Fallback detection untuk semua kondisi
  if (!customPeak && (currentTime - lastPeakTime) > 1000) {  // Shorter timeout
      float avg = (sig.values[0] + sig.values[1] + sig.values[3] + sig.values[4]) / 4.0f;
      customPeak = (sig.values[2] > avg * 1.2f) && (sig.values[2] > 0.52f);  // Less strict
      
      if (customPeak) {
          Serial.println("FALLBACK DETECTION: Using average comparison");
      }
  }
  
  // Adaptive minimum peak distance
  float motionAdaptation = 0.5f + motionFactor * 0.6f;  // Better motion scaling
  minPeakDistance = max(15.0f, 60000.0f / (instantHR * motionAdaptation));
    
  // 6. Penanganan custom peak yang terdeteksi
  if (customPeak && !libraryBeat && (currentTime - lastPeakTime) > minPeakDistance) {
      float detectedHR = 60000.0f / (currentTime - lastPeakTime);
      
      if (detectedHR > 45.0f && detectedHR < 180.0f) {
          // ✅ FASTER INSTANT HR UPDATE UNTUK TRANSITIONS
          float confidence = 1.0f - constrain(abs(detectedHR - instantHR) / 60.0f, 0.0f, 0.8f);
          float activityIntensity = constrain(motionLevel / 20000.0f, 0.0f, 1.0f);
          float baseSmoothing = 0.3f * confidence;
          float adaptiveSmoothing = baseSmoothing * (1.0f + 0.5f * activityIntensity); // ↑ Responsive saat high motion

          instantHR = (1.0f - adaptiveSmoothing) * instantHR + adaptiveSmoothing * detectedHR;
          lastPeakTime = currentTime;
          markPeakForPlotter(true);
          
          rates[rateSpot++] = (byte)instantHR;
          rateSpot %= RATE_SIZE;
          beatAvg = 0;
          for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
          beatAvg /= RATE_SIZE;
          
          Serial.printf("CUSTOM PEAK: %.1f→%.1f BPM (Conf: %.2f|Avg: %d|Motion: %.0f)\n",
                      detectedHR, instantHR, confidence, beatAvg, motionLevel);
      }
  }

  // Fallback mechanism
  static unsigned long lastFallbackCheck = 0;
  if ((instantHR == 0 || isnan(instantHR)) && (millis() - lastFallbackCheck > 4000)) {
      if (beatAvg > 50.0f) {
          instantHR = beatAvg;
          Serial.printf("HR RECOVERY: Restored to avg %.1f BPM\n", beatAvg);
      } else {
          instantHR = 72.0f;
          Serial.println("HR RECOVERY: Using default 72 BPM");
      }
      lastFallbackCheck = millis();
  }

  // Emergency detection
  static unsigned long lastEmergencyCheck = 0;
  if (millis() - lastEmergencyCheck > 3000) {
      bool needsEmergencyRecovery = false;
      
      if ((instantHR < 45.0f || instantHR > 180.0f) && (millis() > 30000)) {
          needsEmergencyRecovery = true;
          Serial.println("EMERGENCY: HR outside valid range");
      }
      
      if ((millis() - lastPeakTime > 5000) && (instantHR > 0)) {
          needsEmergencyRecovery = true; 
          Serial.println("EMERGENCY: No peaks detected");
      }
      
      if ((abs(heartRate - instantHR) > 35.0f) && (millis() > 60000)) {
          needsEmergencyRecovery = true;
          Serial.println("EMERGENCY: Large HR discrepancy");
      }
      
      if (needsEmergencyRecovery) {
          if (beatAvg > 45.0f && beatAvg < 180.0f) {
              instantHR = beatAvg;
              Serial.printf("EMERGENCY RECOVERY: Using avg %.1f BPM\n", beatAvg);
          } else if (heartRate > 45.0f && heartRate < 180.0f) {
              instantHR = heartRate;
              Serial.printf("EMERGENCY RECOVERY: Using model %.1f BPM\n", heartRate);
          } else {
              instantHR = 72.0f;
              Serial.println("EMERGENCY RECOVERY: Using default 72 BPM");
          }
          
          lastPeakTime = millis();
      }
      
      lastEmergencyCheck = millis();
  }
  
  // Conflict resolution
  if (libraryBeat && customPeak && abs(libraryHR - instantHR) > 15.0f) {
      instantHR = 0.7f * libraryHR + 0.3f * instantHR;
      Serial.println("PEAK CONFLICT: Weighted average applied");
  }
  
  // Fallback jika tidak ada peak terdeteksi
  handleMissingPeaks();

  // Update counters untuk adaptive learning
  if (libraryBeat || customPeak) {
    totalPeaksDetected++;
  }
  totalPeaksExpected = expectedPeaks();
}

void processInference() {
  static int lastProcessedIndex = 0;
  static unsigned long lastForceProcess = 0;
  unsigned long currentTime = millis();
  unsigned long startTime = micros();

  // =============== 0. SENSOR CONTACT CHECK - PRIORITAS UTAMA ===============
  if (sensorContactLost) {
    // ✅ JANGAN HANYA RETURN, TAPI SET HR KE NaN
    heartRate = NAN;
    instantHR = NAN;
    lastValidHR = NAN; // ← PENTING: Reset last valid juga
    if (millis() - lastDebugTime > 2000) {
      Serial.println("INFERENCE: Skipped - No sensor contact");
    }
    return;
  }

  // =============== 1. SYSTEM READINESS CHECK ===============
  if ((heartRate == 0 && instantHR == 0 && lastValidHR == 0) || bufferIndex < 10) {
    if (millis() - lastDebugTime > 2000) {
      Serial.printf("INFERENCE: Waiting for data - Buffer=%d/%d\n", 
                   bufferIndex, BUFFER_SIZE);
    }
    return;
  }

  // =============== 2. PROCESSING CONTROL ===============
  int newSamples = (bufferIndex - lastProcessedIndex + BUFFER_SIZE) % BUFFER_SIZE;
  if (newSamples < 0) newSamples += BUFFER_SIZE;
  
  bool shouldProcess = false;
  if (newSamples >= adaptiveStepSize && bufferFull) {
    shouldProcess = true;
  } 
  else if (currentTime - lastForceProcess > 800) {
    shouldProcess = true;
    lastForceProcess = currentTime;
    if (millis() - lastDebugTime > 1000) {
      Serial.println("INFERENCE: Periodic forced processing");
    }
  }

  if (!shouldProcess) {
    return;
  }

  // =============== 3. PREPROCESSING DATA ===============
  float mean = 0.0f, weightedSum = 0.0f;
  float minVal = 1.0f, maxVal = 0.0f;
  
  // ✅ HANN WINDOW dengan boundary check
  for (int i = 0; i < INPUT_SIZE; i++) {
    int idx = getBufferIndex(-INPUT_SIZE + i);
    if (idx < 0 || idx >= BUFFER_SIZE) {
      Serial.println("INFERENCE: Buffer index error, resetting");
      resetFilters();
      return;
    }
    
    float val = ppgBuffer[idx];
    minVal = min(minVal, val);
    maxVal = max(maxVal, val);
    
    float weight = 0.5f * (1 - cos(2 * PI * i / (INPUT_SIZE - 1)));
    mean += val * weight;
    weightedSum += weight;
  }
  
  if (weightedSum < 0.001f) {
    Serial.println("INFERENCE: Weighted sum too small, skipping");
    return;
  }
  mean /= weightedSum;

  // ✅ STANDARD DEVIATION dengan safety
  float stddev = 0.0f;
  for (int i = 0; i < INPUT_SIZE; i++) {
    int idx = getBufferIndex(-INPUT_SIZE + i);
    float weight = 0.5f * (1 - cos(2 * PI * i / (INPUT_SIZE - 1)));
    float diff = ppgBuffer[idx] - mean;
    stddev += weight * diff * diff;
  }
  stddev = max(sqrtf(stddev / weightedSum), MIN_STDDEV);

  // =============== 4. MODEL INPUT PREPARATION ===============
  for (int i = 0; i < 32; i++) {
    if (i < 32 - INPUT_SIZE) {
      input->data.f[i] = 0.0f;
    } else {
      int idx = getBufferIndex(-INPUT_SIZE + i - (32 - INPUT_SIZE));
      if (idx < 0 || idx >= BUFFER_SIZE) {
        input->data.f[i] = 0.0f;
        continue;
      }
      
      float normalized = (ppgBuffer[idx] - mean) / (stddev + 0.1f);
      input->data.f[i] = constrain(normalized, -3.0f, 3.0f);
    }
  }

  // ✅ INPUT VALIDATION
  bool invalidInput = false;
  for (int i = 0; i < 32; i++) {
    if (isnan(input->data.f[i]) || isinf(input->data.f[i])) {
      invalidInput = true;
      break;
    }
  }
  
  if (invalidInput) {
    Serial.println("INFERENCE: Invalid input detected, resetting filters");
    resetFilters();
    return;
  }

  // =============== 5. MODEL INFERENCE ===============
  TfLiteStatus invokeStatus = interpreter->Invoke();
  lastProcessingTimeMs = (micros() - startTime) / 1000.0f;

  // ✅ PROCESSING TIME SAFETY CHECK
  static float avgProcessingTime = 37.5f;
  avgProcessingTime = 0.9f * avgProcessingTime + 0.1f * lastProcessingTimeMs;

  if (lastProcessingTimeMs > avgProcessingTime * 2.0f) {
    Serial.printf("INFERENCE: Slow processing %.1fms, skipping\n", lastProcessingTimeMs);
    return;
  }
  
  if (invokeStatus != kTfLiteOk) {
    Serial.println("INFERENCE: Model invocation failed: " + String(invokeStatus));
    return;
  }

  // =============== 6. POST-PROCESSING ===============
  float rawHR = output->data.f[0];
  
  // ✅ MODEL OUTPUT VALIDATION
  if (isnan(rawHR) || rawHR <= 0 || rawHR > 250.0f) {
    Serial.printf("INFERENCE: Invalid model output: %.1f, using instantHR: %.1f\n", 
                 rawHR, instantHR);
    rawHR = instantHR;
  }

  // ✅ MOTION SUPPRESSION ADAPTIVE
  float motionSuppression = 1.0f / (1.0f + pow(motionLevel/250000.0f, 1.5f));
  float motionAdjustedHR = rawHR * motionSuppression;

  // ✅ SIMPLEAC CORRECTION untuk sinyal stabil
  if (motionLevel < 3000 && bufferIndex > 20) {
    float recentAC = 0.0f;
    for (int i = 0; i < 5; i++) {
      int idx = getBufferIndex(-i-1);
      recentAC += ppgBuffer[idx] - 0.5f;
    }
    recentAC /= 5.0f;
    
    if (abs(recentAC) < 0.03f) { // Sangat stabil
      motionAdjustedHR = 0.8f * motionAdjustedHR + 0.2f * instantHR;
    }
  }

  // =============== 7. HEART RATE FUSION ===============
  float fusedHR = fuseHeartRate(motionAdjustedHR, instantHR, motionLevel);

  // ✅ FUSION SAFETY CHECK
  if (isnan(fusedHR) || fusedHR < 40.0f || fusedHR > 220.0f) {
    if (instantHR > 40.0f && instantHR < 220.0f) {
      fusedHR = instantHR;
    } else if (lastValidHR > 40.0f && lastValidHR < 220.0f) {
      fusedHR = lastValidHR;
    } else {
      fusedHR = 72.0f; // Fallback
    }
    Serial.printf("INFERENCE: Fused HR invalid, using: %.1f\n", fusedHR);
  }

  // =============== PERBAIKI HR CALCULATION ===============
  // ✅ MOTION-BASED HR COMPENSATION (TAMBAHKAN DI SINI)
  float hrIncrease = 0.0f;
  if (motionLevel > 3000.0f) {
      hrIncrease = (motionLevel - 3000.0f) / 25000.0f;
      fusedHR = fusedHR * (1.0f + hrIncrease);
      fusedHR = constrain(fusedHR, 40.0f, 180.0f);
      
      // ✅ DEBUG OUTPUT
      static unsigned long lastHRDebug = 0;
      if (millis() - lastHRDebug > 1000) {
          Serial.printf("HR_COMPENSATION: Motion=%.0f Boost=%.2f HR=%.1f\n",
                      motionLevel, (1.0f + hrIncrease), fusedHR);
          lastHRDebug = millis();
      }
  }
  // ======================================================

  // =============== 8. KALMAN FILTER FUSION ===============
  float smoothingFactor;
  float hrChange = abs(fusedHR - lastValidHR);

  // ✅ DECOUPLED SMOOTHING - TIDAK TERGANTUNG MOTION LEVEL
  float baseSmoothing = 0.82f; // Base lebih responsive
  float motionInfluence = 0.08f; // Pengaruh motion lebih kecil

  smoothingFactor = baseSmoothing - motionInfluence * constrain(motionLevel / 30000.0f, 0.0f, 1.0f);
  smoothingFactor = constrain(smoothingFactor, 0.7f, 0.85f); // Safety bounds

  lastValidHR = smoothingFactor * lastValidHR + (1.0f - smoothingFactor) * fusedHR;

  // ✅ ADAPTIVE KALMAN GAIN
  estimateError += processNoise;
  float kalmanGain = estimateError / (estimateError + measurementNoise);
  kalmanGain = constrain(kalmanGain, 0.3f, 0.7f); // Lebih stabil
  estimateError *= (1.0f - kalmanGain);

  lastValidHR = constrain(lastValidHR, 40.0f, 220.0f);

  // =============== 8.5 TIME-BASED RESPONSIVENESS ===============
  static unsigned long lastSignificantChange = 0;
  static float lastStableHR = 0.0f;

  // Detect significant HR change
  if (abs(fusedHR - lastStableHR) > 10.0f) {
      lastSignificantChange = millis();
      lastStableHR = fusedHR;
  }

  // Time-based window sizing
  unsigned long timeSinceChange = millis() - lastSignificantChange;
  int effectiveWindow;

  if (timeSinceChange < 4000) { // 4 detik pertama
      effectiveWindow = 4; // Very responsive
  } else if (timeSinceChange < 10000) { // 4-10 detik
      effectiveWindow = 6; // Responsive
  } else {
      effectiveWindow = 8; // Stable
  }

  effectiveWindow = constrain(effectiveWindow, 4, HR_AVG_WINDOW);

  // =============== 8.6 PREDICTIVE MOTION COMPENSATION ===============
  static float motionPrediction = 0.0f;
  static float motionPeak = 0.0f;

  // Update prediction and peak
  motionPrediction = 0.9f * motionPrediction + 0.1f * motionLevel;
  motionPeak = max(motionPeak * 0.97f, motionLevel);

  // Jika predicted motion masih tinggi, reduce smoothing
  if (motionPrediction > motionPeak * 0.4f) {
      smoothingFactor = max(smoothingFactor - 0.1f, 0.7f);
      Serial.printf("PREDICTIVE BOOST: Pred=%.0f Peak=%.0f\n", motionPrediction, motionPeak);
  }

  // =============== 9. FINAL SMOOTHING ===============
  hrBuffer[hrIndex] = lastValidHR;
  hrIndex = (hrIndex + 1) % HR_AVG_WINDOW;

  // ✅ DYNAMIC WEIGHTED MOVING AVERAGE - LEBIH RESPONSIF BERDASARKAN MOTION
  heartRate = 0;
  float weightSum = 0;

  // ✅ WINDOW SIZE BERDASARKAN MOTION LEVEL - LEBIH ADAPTIF
  if (motionLevel > 8000.0f) {
      effectiveWindow = 4;  // Very responsive untuk lari (2 detik)
  } else if (motionLevel > 3000.0f) {
      effectiveWindow = 6;  // Responsive untuk jalan (3 detik)
  } else {
      effectiveWindow = 8;  // Smooth untuk diam (4 detik)
  }

  // ✅ SAFETY BOUNDS UNTUK WINDOW SIZE
  effectiveWindow = constrain(effectiveWindow, 4, HR_AVG_WINDOW);

  // ✅ WEIGHTED MOVING AVERAGE DENGAN COSINE WINDOW
  for (int i = 0; i < effectiveWindow; i++) {
      float weight = 0.5f * (1 + cos(PI * i / (effectiveWindow - 1)));
      heartRate += hrBuffer[(hrIndex - i + HR_AVG_WINDOW) % HR_AVG_WINDOW] * weight;
      weightSum += weight;
  }

  // ✅ FALLBACK JIKA WEIGHT SUM TERLALU KECIL
  if (weightSum > 0.001f) {
      heartRate /= weightSum;
  } else {
      heartRate = lastValidHR;
  }

  // ✅ DEBUG OUTPUT UNTUK SMOOTHING
  static unsigned long lastSmoothDebug = 0;
  if (millis() - lastSmoothDebug > 2000) {
      Serial.printf("SMOOTHING: Motion=%.0f Window=%d HR=%.1f→%.1f\n",
                  motionLevel, effectiveWindow, lastValidHR, heartRate);
      lastSmoothDebug = millis();
  }

  // =============== 10. FINAL VALIDATION & SAFETY ===============
  heartRate = validateHeartRate(heartRate);
  instantHR = validateHeartRate(instantHR);

  // ✅ GUARANTEE NaN IF NO CONTACT (safety net)
  if (sensorContactLost) {
    heartRate = NAN;
    instantHR = NAN;
  }

  // ✅ DEBUG OUTPUT
  if (millis() - lastDebugTime > 1000) {
    Serial.printf("INFERENCE_FINAL: Model=%.1f Fused=%.1f Final=%.1f Motion=%.0f\n",
                 rawHR, fusedHR, heartRate, motionLevel);
  }

  lastProcessedIndex = bufferIndex;
}

float fuseHeartRate(float modelHR, float instantHR, float motionLevel) {
  // ✅ VALIDASI INPUT LEBIH ROBUST
  if (isnan(modelHR) || isinf(modelHR)) modelHR = instantHR;
  if (isnan(instantHR) || isinf(instantHR)) instantHR = modelHR;
  if (isnan(modelHR) && isnan(instantHR)) return 72.0f; // Fallback
  
  modelHR = constrain(modelHR, 40.0f, 220.0f);
  instantHR = constrain(instantHR, 40.0f, 220.0f);

  // Debug Input
  Serial.printf("FUSION DEBUG: lastPeakTime=%lu, millis()=%lu, diff=%lus\n",
             lastPeakTime, millis(), (millis() - lastPeakTime)/1000);
             
  Serial.printf("FUSION INPUT: Model=%.1f, Instant=%.1f, Motion=%.0f\n",
               modelHR, instantHR, motionLevel);
  
  // 1. Hitung Confidence - ✅ LEBIH AGGRESSIVE UNTUK HIGH MOTION
  float motionFactor = constrain(motionLevel / 20000.0f, 0.0f, 1.5f); // ↓ Threshold dari 80k ke 20k
  
  float modelConfidence, instantConfidence;
  
  // ✅ LOGIKA LEBIH RESPONSIF: High motion → lebih percaya instant HR
  if (motionLevel > 15000.0f) { // High motion
    modelConfidence = 0.3f - 0.2f * motionFactor;  // 0.3 → 0.1 (lebih aggressive)
    instantConfidence = 0.7f + 0.2f * motionFactor; // 0.7 → 0.9 (lebih aggressive)
  } else { // Low to medium motion
    modelConfidence = 0.6f - 0.3f * motionFactor;  // 0.6 → 0.3 (lebih responsive)  
    instantConfidence = 0.4f + 0.3f * motionFactor; // 0.4 → 0.7 (lebih responsive)
  }
  
  // Time-based confidence adjustment - ✅ OPTIMIZED
  if (lastPeakTime > 0 && (millis() - lastPeakTime) < 2000) {
      float recencyBoost = 1.0f - (millis() - lastPeakTime) / 2000.0f;
      instantConfidence += 0.4f * recencyBoost; // ↑ Lebih aggressive

      // Additional boost untuk activity transitions
      if (motionLevel > 8000.0f) {
          instantConfidence += 0.3f * recencyBoost;
      }

      instantConfidence = min(instantConfidence, 0.85f); // Safety cap
  }

  // ✅ VALIDASI CONFIDENCE VALUES
  modelConfidence = constrain(modelConfidence, 0.1f, 0.8f);
  instantConfidence = constrain(instantConfidence, 0.1f, 0.8f);

  // 2. Penanganan discrepancy - ✅ LEBIH AGGRESSIVE
  if (abs(modelHR - instantHR) > 25.0f) { // ↓ Threshold dari 30 ke 25
    if (motionLevel > 15000.0f) { // High motion - ↓ Threshold dari 20k ke 15k
      instantConfidence = min(instantConfidence + 0.3f, 0.8f); // ↑ Lebih aggressive
      modelConfidence = max(modelConfidence - 0.15f, 0.1f);    // ↑ Lebih aggressive
    } else { // Low motion
      modelConfidence = min(modelConfidence + 0.3f, 0.8f);     // ↑ Lebih aggressive
      instantConfidence = max(instantConfidence - 0.15f, 0.1f); // ↑ Lebih aggressive
    }
  }

  // 3. Fusi Adaptif - ✅ TAMBAH MOTION BOOST EXPLICIT
  float fusedHR;
  if (instantConfidence > 0.15f) {
    fusedHR = (modelHR * modelConfidence + instantHR * instantConfidence) 
             / (modelConfidence + instantConfidence);
  } else {
    fusedHR = modelHR; // Fallback ke model jika instant tidak reliable
  }
  
  // ✅ MOTION BOOST EXPLICIT - TAMBAH INI
  if (motionLevel > 3000.0f) {
    float motionBoost = 1.0f + 0.7f * constrain((motionLevel - 2000.0f) / 20000.0f, 0.0f, 1.0f);
    fusedHR *= motionBoost;
    Serial.printf("MOTION BOOST: %.1f → %.1f (x%.2f)\n", fusedHR/motionBoost, fusedHR, motionBoost);
  }
  
  // 4. Motion Compensation - ✅ FIXED COMPENSATION
  if (motionLevel > 30000.0f) {
    float compensation = 1.0f + 0.2f * constrain((motionLevel-30000)/100000.0f, 0.0f, 1.0f); // ↑ Lebih aggressive
    float compensatedHR = fusedHR * compensation;
    
    // Safety check untuk compensation
    if (compensatedHR <= 220.0f) {
      fusedHR = compensatedHR;
      Serial.printf("MOTION COMP: %.1f → %.1f (x%.2f)\n", fusedHR/compensation, fusedHR, compensation);
    }
  }
  
  // Debug Output
  Serial.printf("FUSION OUTPUT: %.1f (MConf:%.2f, IConf:%.2f, Mot:%.0f)\n",
               fusedHR, modelConfidence, instantConfidence, motionLevel);
               
  // 5. Dynamic cap berdasarkan motion level
  float maxHR = 200.0f + 30.0f * motionFactor; // ↑ Lebih tinggi untuk aktivitas intense
  return constrain(fusedHR, 45.0f, maxHR);
}

void applyHighHROptimizations() {
  // Virtual upsampling 2x
  float tempBuffer[64];
  for (int i = 0; i < 32; i++) {
    tempBuffer[2*i] = ppgBuffer[i];
    tempBuffer[2*i+1] = (ppgBuffer[i] + ppgBuffer[(i+1)%32]) / 2.0f;
  }

  // High-pass filter
  static float hpState = 0;
  const float hpCoeff = 0.92f;
  for (int i = 0; i < 64; i++) {
    float newVal = tempBuffer[i] - hpState;
    hpState = tempBuffer[i] * hpCoeff + hpState * (1.0f-hpCoeff);
    tempBuffer[i] = newVal * 1.2f;
  }

  // Downsample back to 32Hz
  for (int i = 0; i < 32; i++) {
    ppgBuffer[i] = tempBuffer[2*i];
  }
}

void applyMotionArtifactReduction() {
  static float motionNoiseLevel = 0.0f;
  float currentNoise = calculateSignalVariance();
  
  if (isnan(currentNoise)) currentNoise = 0.0f;
  
  // Smooth noise estimation dengan adaptive smoothing
  float motionFactor = constrain(motionLevel / 60000.0f, 0.0f, 1.0f);
  float smoothFactor = 0.85f + 0.1f * (1.0f - motionFactor); // Lebih smooth saat diam
  motionNoiseLevel = smoothFactor * motionNoiseLevel + (1.0f - smoothFactor) * currentNoise;
  
  // Adaptive threshold berdasarkan motion level
  float noiseThreshold = 0.01f + 0.04f * motionFactor; // 0.01-0.05
  noiseThreshold *= (1.0f + motionFactor); // Scale dengan motion level
  
  // Phase 1 - Global reduction (optimized for motion level)
  if (motionNoiseLevel > noiseThreshold) {
    // Continuous reduction factor berdasarkan motion level
    float reductionFactor;
    if (motionLevel > 100000.0f) {
      // Extreme motion - aggressive reduction (60-75%)
      reductionFactor = map(constrain(motionLevel, 100000.0f, 200000.0f), 
                          100000.0f, 200000.0f, 0.75f, 0.6f);
    } 
    else if (motionLevel > 50000.0f) {
      // High motion - moderate reduction (75-90%)
      reductionFactor = map(constrain(motionLevel, 50000.0f, 100000.0f), 
                          50000.0f, 100000.0f, 0.9f, 0.75f);
    }
    else if (motionLevel > 20000.0f) {
      // Medium motion - light reduction (90-97%)
      reductionFactor = map(constrain(motionLevel, 20000.0f, 50000.0f), 
                          20000.0f, 50000.0f, 0.97f, 0.9f);
    }
    else {
      // Low motion - minimal reduction (97-99%)
      reductionFactor = map(constrain(motionLevel, 0.0f, 20000.0f), 
                          0.0f, 20000.0f, 0.99f, 0.97f);
    }

    // Apply reduction ke seluruh buffer
    for (int i = 0; i < BUFFER_SIZE; i++) {
      ppgBuffer[i] *= reductionFactor;
    }

    // Debug output
    static unsigned long lastReductionLog = 0;
    if (millis() - lastReductionLog > 2000) {
      Serial.printf("MOTION REDUCTION: %.0f->%.0f (Factor: %.2f | Noise: %.3f/%.3f)\n",
                   motionLevel, motionLevel * reductionFactor, reductionFactor, 
                   motionNoiseLevel, noiseThreshold);
      lastReductionLog = millis();
    }
  }

  // Phase 2 - Selective spike removal (improved condition)
  if (motionLevel > 15000.0f) {
    float spikeThreshold = noiseThreshold * 2.5f;
    
    for (int i = 1; i < BUFFER_SIZE - 1; i++) {
      float diffPrev = abs(ppgBuffer[i] - ppgBuffer[i-1]);
      float diffNext = abs(ppgBuffer[i] - ppgBuffer[i+1]);
      
      // Detect spikes berdasarkan deviation dari neighbors
      if ((diffPrev > 0.08f || diffNext > 0.08f) && ppgBuffer[i] > spikeThreshold) {
        float avgNeighbor = (ppgBuffer[i-1] + ppgBuffer[i+1]) / 2.0f;
        
        // Only remove jika deviation significant
        if (abs(ppgBuffer[i] - avgNeighbor) > 0.06f) {
          ppgBuffer[i] = avgNeighbor;
          
          // Debug spike removal
          static unsigned long lastSpikeLog = 0;
          if (millis() - lastSpikeLog > 1000) {
            Serial.printf("SPIKE REMOVED: Index=%d, Value=%.3f->%.3f\n",
                         i, ppgBuffer[i], avgNeighbor);
            lastSpikeLog = millis();
          }
        }
      }
    }
  }

  // Phase 3 - Adaptive smoothing untuk extreme motion
  if (motionLevel > 80000.0f) {
    // Determine smoothing intensity berdasarkan motion level
    int smoothingWindow;
    if (motionLevel > 150000.0f) {
      smoothingWindow = 5; // Very aggressive smoothing
    } 
    else if (motionLevel > 100000.0f) {
      smoothingWindow = 3; // Aggressive smoothing
    }
    else {
      smoothingWindow = 1; // Light smoothing
    }
    
    // Apply smoothing dengan moving average
    for (int i = smoothingWindow; i < BUFFER_SIZE - smoothingWindow; i++) {
      float sum = 0.0f;
      for (int j = -smoothingWindow; j <= smoothingWindow; j++) {
        sum += ppgBuffer[i + j];
      }
      ppgBuffer[i] = sum / (2 * smoothingWindow + 1);
    }
    
    // Debug smoothing
    static unsigned long lastSmoothLog = 0;
    if (millis() - lastSmoothLog > 2000) {
      Serial.printf("EXTREME SMOOTHING: Motion=%.0f, Window=%d\n", 
                   motionLevel, smoothingWindow);
      lastSmoothLog = millis();
    }
  }

  // Phase 4 - Signal normalization untuk maintain dynamic range
  if (motionLevel > 30000.0f) {
    // Cari min dan max values dalam buffer
    float minVal = ppgBuffer[0];
    float maxVal = ppgBuffer[0];
    
    for (int i = 1; i < BUFFER_SIZE; i++) {
      if (ppgBuffer[i] < minVal) minVal = ppgBuffer[i];
      if (ppgBuffer[i] > maxVal) maxVal = ppgBuffer[i];
    }
    
    float range = maxVal - minVal;
    
    // Normalize hanya jika range reasonable
    if (range > 0.1f && range < 2.0f) {
      float targetRange = 0.5f + 0.3f * motionFactor; // 0.5-0.8 range
      
      for (int i = 0; i < BUFFER_SIZE; i++) {
        // Scale ke range yang diinginkan
        ppgBuffer[i] = ((ppgBuffer[i] - minVal) / range) * targetRange + (0.5f - targetRange/2.0f);
      }
      
      // Debug normalization
      static unsigned long lastNormLog = 0;
      if (millis() - lastNormLog > 3000) {
        Serial.printf("SIGNAL NORMALIZED: Range=%.3f->%.3f\n", range, targetRange);
        lastNormLog = millis();
      }
    }
  }
}

// =============== SUPPORT FUNCTIONS ===============

// =============== FUNGSI UPDATE ADAPTIVE PARAMETERS ===============
void updateAdaptiveParameters() {
    // ✅ DECOUPLE: Gunakan instant motion bukan motionLevel
    static float lastRedForAdaptive = 0, lastIRForAdaptive = 0;
    
    // Dapatkan nilai sensor terkini
    float currentRed = particleSensor.getRed();
    float currentIR = particleSensor.getIR();
    
    // Hitung instant motion (decoupled dari motionLevel)
    float instantRedChange = abs(currentRed - lastRedForAdaptive);
    float instantIRChange = abs(currentIR - lastIRForAdaptive);
    float instantMotion = (instantRedChange + instantIRChange) * 0.5f;
    
    // Update last values
    lastRedForAdaptive = currentRed;
    lastIRForAdaptive = currentIR;
    
    // Calculate motion factor dari instant motion (BUKAN dari motionLevel)
    float motionFactor = constrain(instantMotion / 8000.0f, 0.0f, 2.0f);
    
    // 1. Smooth factor - lebih smooth saat diam, responsive saat gerak
    adaptiveSmoothFactor = 0.93f - 0.23f * motionFactor; // 0.93-0.70
    
    // 2. Motion scale - natural progression
    adaptiveMotionScale = 30000.0f + 170000.0f * motionFactor;
    
    // 3. Step size - adaptive berdasarkan motion
    adaptiveStepSize = STEP_SIZE_NORMAL - (STEP_SIZE_NORMAL - STEP_SIZE_RUN) * motionFactor;
    
    // 4. Fusion weights - balanced approach
    adaptiveModelWeight = 0.65f - 0.35f * motionFactor; // 0.65-0.30
    adaptiveInstantWeight = 0.35f + 0.35f * motionFactor; // 0.35-0.70
    
    // 5. Kalman parameters - adaptive noise
    processNoise = 1.5f + 4.5f * motionFactor;
    measurementNoise = 2.0f + 6.0f * motionFactor;
    
    // Debug output
    static unsigned long lastAdaptiveDebug = 0;
    if (millis() - lastAdaptiveDebug > 2000) {
        Serial.printf("ADAPTIVE_DECOUPLED: Instant=%.0f | Factor=%.2f | Smooth=%.2f\n",
                     instantMotion, motionFactor, adaptiveSmoothFactor);
        lastAdaptiveDebug = millis();
    }
}
// ====================================================

void showCalibrationScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Initial Calibration");
  display.println("Please wait...");
  display.display();
}

// =============== UNIVERSAL CHECK_STILLNESS FUNCTION ===============
void checkStillness() {
    static float stillnessCounter = 0;
    static unsigned long lastStillnessUpdate = 0;
    
    // Update setiap 100ms saja untuk efisiensi
    unsigned long currentTime = millis();
    if (currentTime - lastStillnessUpdate < 100) {
        return;
    }
    lastStillnessUpdate = currentTime;
    
    // Hitung motion factor (0-1) berdasarkan motion level
    float motionFactor = constrain(motionLevel / 60000.0f, 0.0f, 1.0f);
    
    // Stillness detection berdasarkan motion level
    if (motionLevel < 1000.0f) { // Sangat diam
        stillnessCounter += 0.2f; // Increment lebih cepat
        
        // After 3 seconds of continuous stillness
        if (stillnessCounter >= 30.0f) { // 30 x 0.1 = 3 seconds
            // Ultra-stable parameters untuk kondisi sangat diam
            processNoise = 1.0f;
            measurementNoise = 1.5f;
            
            // Dynamic threshold adjustment untuk sensitivity tinggi
            if (signalRange > 0) {
                dynamicThreshold = max(0.001f, 0.05f * signalRange);
            }
            
            // Reset filters untuk optimal performance saat diam
            if (stillnessCounter >= 40.0f) { // Setelah 4 detik
                resetFilters();
                stillnessCounter = 30.0f; // Reset ke 3 detik
                Serial.println("STILLNESS: Ultra-stable mode activated");
            }
        }
    } 
    else if (motionLevel < 5000.0f) { // Cukup diam
        stillnessCounter += 0.1f; // Increment normal
        stillnessCounter = constrain(stillnessCounter, 0.0f, 20.0f);
    }
    else { // Ada movement
        stillnessCounter = max(0.0f, stillnessCounter - 0.3f); // Decrement cepat
    }
    
    // Adaptive smoothing berdasarkan stillness
    float stillnessFactor = constrain(stillnessCounter / 30.0f, 0.0f, 1.0f);
    
    // Apply stillness-based adjustments
    if (stillnessFactor > 0.7f) {
        // High stillness - increase smoothing
        adaptiveSmoothFactor = 0.93f;
        adaptiveModelWeight = 0.8f; // Lebih percaya model
        adaptiveInstantWeight = 0.2f;
    }
    
    // Debug output
    static unsigned long lastStillnessDebug = 0;
    if (currentTime - lastStillnessDebug > 2000) {
        Serial.printf("STILLNESS: Level=%.0f | Counter=%.1f/30 | Factor=%.2f\n",
                     motionLevel, stillnessCounter, stillnessFactor);
        lastStillnessDebug = currentTime;
    }
}
// ================================================================

// =============== BUFFER ACCESS HELPER ===============
int getBufferIndex(int offset) {
    int index = bufferIndex + offset;
    if (index < 0) index += BUFFER_SIZE;
    if (index >= BUFFER_SIZE) index -= BUFFER_SIZE;
    return index;
}

void handleMissingPeaks() {
  unsigned long sinceLastPeak = millis() - lastPeakTime;
  static unsigned long lastAdjustTime = 0;

  // ✅ COOLDOWN CHECK - Jangan proses jika recovery sedang berjalan
  if (recoveryInProgress) {
    if (millis() - lastEmergencyRecovery > RECOVERY_TIMEOUT_MS) {
      recoveryInProgress = false;
      Serial.println("RECOVERY: Timeout reached, resetting flag");
    }
    return;
  }

  // Hanya bertindak jika tidak ada puncak yang ditemukan dalam waktu lama
  if (sinceLastPeak > 2000) {

    // ✅ COOLDOWN VALIDATION - Pastikan tidak dalam cooldown period
    unsigned long sinceLastRecovery = millis() - lastEmergencyRecovery;
    bool recoveryAllowed = (sinceLastRecovery > RECOVERY_COOLDOWN_MS);
    
    if (!recoveryAllowed) {
      if (millis() - lastAdjustTime > 2000) {
        Serial.printf("RECOVERY: Cooldown active (%lu/%lu ms)\n", 
                     sinceLastRecovery, RECOVERY_COOLDOWN_MS);
        lastAdjustTime = millis();
      }
      return;
    }

    // 1. FALLBACK: Jika model tidak valid - ✅ IMPROVED
    if (isnan(heartRate)) {
      if (instantHR < 40.0f || instantHR > 220.0f) {
        instantHR = lastStableHR;  // Gunakan lastStableHR而不是 0
        Serial.printf("RESET: Using stable HR %.1f BPM\n", instantHR);
      }
      
      if (instantHR > 40.0f) {
        lastPeakTime = millis() - (60000.0f / instantHR);
      } else {
        lastPeakTime = millis();
      }
      return;
    }

    // 2. LOGIKA PENYESUAIAN UTAMA - ✅ OPTIMIZED
    if (heartRate > 40.0f && heartRate < 220.0f) {
      float targetHR = heartRate;
      float hrError = targetHR - instantHR;

      // ✅ POST-COOLDOWN AGGRESSIVE CORRECTION (NEW)
      float adjustmentFactor = 0.08f; // Default
      if (sinceLastRecovery > RECOVERY_COOLDOWN_MS * 2) { // 6+ detik tanpa recovery
        adjustmentFactor = 0.15f; // More aggressive
        Serial.println("RECOVERY: Post-cooldown aggressive correction");
      }

      // ✅ Hanya adjust jika error signifikan (LESS RESTRICTIVE)
      if (abs(hrError) > 15.0f && instantHR > 0) { // ✅ dari 20.0f ke 15.0f
        recoveryInProgress = true;
        lastEmergencyRecovery = millis();
        
        float adjustment = hrError * adjustmentFactor; // ✅ Dynamic adjustment
        instantHR += adjustment;
        instantHR = constrain(instantHR, 40.0f, 220.0f);

        Serial.printf("EMERGENCY RECOVERY: %.1f→%.1f BPM (Error: %.1f)\n", 
                     instantHR - adjustment, instantHR, hrError);
        Serial.printf("RECOVERY: Cooldown activated for %lu ms\n", RECOVERY_COOLDOWN_MS);
        
        lastAdjustTime = millis();
      }
    }

    // ✅ SAFETY RELEASE MECHANISM (NEW)
    if (sinceLastPeak > 5000 && instantHR > 0 && instantHR < 60.0f) {
      // Jika terjebak di low HR terlalu lama, force release
      recoveryInProgress = false;
      Serial.println("RECOVERY: Force release - stuck in low HR");
    }

    // 3. Perbarui HR stabil terakhir - ✅ IMPROVED
    if (heartRate > 40.0f && heartRate < 220.0f) {
      lastStableHR = heartRate;  // Simpan nilai stable
    }
  }
}

bool performInitialCalibration() {
  //=============== KONFIGURASI DASAR ===============//
  const uint8_t fastSamples = 8;                // 1.0s @20ms interval
  const uint16_t baseCalibSamples = 80;         // 1.6s @20ms (mode normal)
  const uint16_t motionCalibSamples = 120;      // 2.4s @20ms (mode lari)
  const uint16_t sampleInterval = 20;           // 50Hz sampling rate
  const uint16_t preStabilizeTime = 1000;       // 1s observasi awal
  const uint16_t postStabilizeTime = 500;       // 0.5s validasi akhir
  const float minDynamicRange = 1800.0f;        // Batas sinyal normal
  const float motionRangeMin = 2500.0f;         // Batas sinyal saat lari
  const uint32_t globalTimeout = 8000;          // 8s timeout total
  const float motionNoiseThreshold = 3000.0f;   // Batas deteksi gerakan

  //=============== INISIALISASI VARIABEL ===============//
  unsigned long startTime = millis();
  float redSum = 0, irSum = 0, minRed = 999999, maxRed = 0;
  float baselineNoise = 0, lastRedValue = 0, peakNoise = 0;
  bool isHighMotion = false;
  uint8_t motionEventCount = 0;

  //=============== PHASE 0: PRE-STABILIZATION & MOTION ANALYSIS ===============//
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("Stabilizing...");
  display.setCursor(0, 10);
  display.print("Hold still...");
  display.display();

  unsigned long stabilizeStart = millis();
  while (millis() - stabilizeStart < preStabilizeTime) {
    if (millis() - startTime > globalTimeout) {
      heartStatus = "Timeout: Pre-Stabilize";
      dcFiltered = 100000.0f;
      contactThreshold = dcFiltered * 0.5f;
      return false;
    }

    float redValue = particleSensor.getRed();
    float instantNoise = abs(redValue - lastRedValue);
    baselineNoise += instantNoise;
    lastRedValue = redValue;

    // Deteksi gerakan signifikan
    if (instantNoise > motionNoiseThreshold) {
      motionEventCount++;
      if (motionEventCount > 3) isHighMotion = true;
    }

    // Update progress bar (0-25%)
    display.fillRect(0, 20, map(millis() - stabilizeStart, 0, preStabilizeTime, 0, 32), 5, SSD1306_WHITE);
    display.display();
    delay(50);
  }
  baselineNoise /= (preStabilizeTime / 50); // Hitung rata-rata noise

  //=============== PHASE 1: FAST CALIBRATION ===============//
  display.setCursor(0, 30);
  display.print("Fast Calib...");
  display.display();

  redSum = 0; minRed = 999999; maxRed = 0;
  for (uint8_t i = 0; i < fastSamples; i++) {
    if (millis() - startTime > globalTimeout) {
      heartStatus = "Timeout: Fast Calib";
      return false;
    }

    float redValue = particleSensor.getRed();
    float irValue = particleSensor.getIR();

    // Validasi hardware
    if (redValue < 100 || redValue > 1000000 || irValue < 5000) {
      heartStatus = "Sensor Error";
      return false;
    }

    redSum += redValue;
    minRed = min(minRed, redValue);
    maxRed = max(maxRed, redValue);

    // Progress bar (25-50%)
    display.fillRect(32, 20, map(i, 0, fastSamples, 0, 32), 5, SSD1306_WHITE);
    display.display();
    delay(sampleInterval);
  }

  // Cek kualitas sinyal cepat
  float fastDynamicRange = maxRed - minRed;
  float actualMinRange = isHighMotion ? motionRangeMin : minDynamicRange;
  if (fastDynamicRange < actualMinRange) {
    heartStatus = "Poor Signal Quality";
    dcFiltered = 100000.0f;
    contactThreshold = dcFiltered * 0.5f;
    return false;
  }

  //=============== PHASE 2: DETAILED CALIBRATION ===============//
  display.setCursor(0, 40);
  display.print(isHighMotion ? "Running Calib..." : "Detailed Calib...");
  display.display();

  uint16_t actualSamples = isHighMotion ? motionCalibSamples : baseCalibSamples;
  redSum = 0; minRed = 999999; maxRed = 0; irSum = 0;
  float motionCompensationSum = 0;

  for (uint16_t i = 0; i < actualSamples; i++) {
    if (millis() - startTime > globalTimeout) {
      heartStatus = "Timeout: Detailed Calib";
      return false;
    }

    // Pembacaan sensor dengan filter
    float rawRed = particleSensor.getRed();
    float rawIR = particleSensor.getIR();
    static float filteredRed = rawRed;
    filteredRed = isHighMotion ? 
      0.6f * filteredRed + 0.4f * rawRed : // Filter agresif untuk lari
      0.8f * filteredRed + 0.2f * rawRed;  // Filter halus untuk diam

    // Kompensasi gerakan
    float instantNoise = abs(filteredRed - lastRedValue);
    if (isHighMotion) {
      delay(sampleInterval + (uint8_t)(instantNoise * 1.5f)); // Delay adaptif
      motionCompensationSum += instantNoise;
    }

    // Akumulasi data
    redSum += filteredRed;
    irSum += rawIR;
    minRed = min(minRed, filteredRed);
    maxRed = max(maxRed, filteredRed);
    lastRedValue = filteredRed;

    // Progress bar (50-90%)
    display.fillRect(64, 20, map(i, 0, actualSamples, 0, 64), 5, SSD1306_WHITE);
    if (i % 10 == 0) display.display(); // Update periodic untuk hemat CPU
    delay(isHighMotion ? 10 : sampleInterval); // Interval lebih cepat saat lari
  }

  //=============== PHASE 3: POST-STABILIZATION & VALIDATION ===============//
  display.setCursor(0, 50);
  display.print("Validating...");
  display.fillRect(64, 20, 64, 5, SSD1306_WHITE); // Progress 100%
  display.display();

  // Verifikasi akhir
  float postCalibNoise = 0;
  for (uint8_t i = 0; i < 20; i++) {
    float redValue = particleSensor.getRed();
    postCalibNoise += abs(redValue - lastRedValue);
    lastRedValue = redValue;
    delay(25);
  }
  postCalibNoise /= 20;

  // Reject jika noise tidak stabil
  if (postCalibNoise > baselineNoise * (isHighMotion ? 4.0f : 2.5f)) {
    heartStatus = "Unstable Signal";
    return false;
  }

  //=============== THRESHOLD CALCULATION ===============//
  dcFiltered = redSum / actualSamples;
  contactBaseline = dcFiltered;
  float dynamicRange = maxRed - minRed;
  float motionFactor = motionCompensationSum / actualSamples;

  // Formula hybrid untuk berbagai kondisi
  if (isHighMotion) {
    // Koreksi non-linear untuk lari
    float irRatio = (irSum / actualSamples) / 10000.0f;
    float rangeFactor = dynamicRange / motionRangeMin;
    contactThreshold = dcFiltered * (0.25f + 0.05f * exp(-rangeFactor)) 
                     * constrain(irRatio, 0.7f, 1.3f);
  } else {
    // Threshold standar
    if (dynamicRange > 4000.0f) {
      contactThreshold = dcFiltered * 0.3f;
    } else if (dynamicRange > 2000.0f) {
      contactThreshold = dcFiltered * 0.4f;
    } else {
      contactThreshold = dcFiltered * 0.5f;
    }
  }

  //=============== FINALIZATION ===============//
  needsCalibration = false;
  lastContactTime = millis();

  // [BARU] Panggil resetFilters dengan mode post-calibration
  resetFilters(true); // true = isPostCalibration
  
  // [BARU] Sync parameter dengan resetFilters()
  dynamicThreshold = contactThreshold / max(dcFiltered, 1.0f);

  Serial.printf("Calib Success: DC=%.1f Thresh=%.1f Motion=%d Noise=%.1f\n",
               dcFiltered, contactThreshold, isHighMotion, baselineNoise);
  
  heartStatus = isHighMotion ? "Ready (Running)" : "Ready";

  //====== [TAMBAHAN] HEALTH SCORE RESET AFTER CALIBRATION ======//
  systemHealthScore = 100;
  consecutiveErrors = 0;
  return true;
}

bool performRecalibration() {
  //=============== CONFIGURABLE PARAMS ===============//
  const uint8_t recalSamples = 30;              // 0.6s @20ms (lebih banyak dari sebelumnya)
  const uint16_t sampleInterval = 20;           // 50Hz sampling
  const float minDynamicRange = 1500.0f;        // Batas minimal sinyal
  const uint32_t recalTimeout = 3000;           // Timeout 3 detik
  const float motionNoiseThreshold = 2000.0f;   // Batas deteksi gerakan

  //=============== INISIALISASI ===============//
  unsigned long recalStartTime = millis();
  float redSum = 0, irSum = 0, minRed = 999999, maxRed = 0;
  float baselineNoise = 0, lastRedValue = 0;
  uint8_t motionEventCount = 0;

  //=============== PHASE 1: QUICK STABILIZATION CHECK ===============//
  heartStatus = "Recalibrating...";
  updateDisplay();

  // Analisis noise selama 0.3s (15 samples)
  for (int i = 0; i < 15; i++) {
    if (millis() - recalStartTime > recalTimeout) {
      heartStatus = "Recal Timeout";
      return false;
    }

    float redValue = particleSensor.getRed();
    baselineNoise += abs(redValue - lastRedValue);
    lastRedValue = redValue;

    // Deteksi gerakan signifikan
    if (abs(redValue - lastRedValue) > motionNoiseThreshold) {
      motionEventCount++;
    }
    delay(20);
  }
  baselineNoise /= 15;

  //=============== PHASE 2: MOTION-ADAPTIVE RECALIBRATION ===============//
  bool isHighMotion = (motionEventCount >= 3);
  redSum = 0; minRed = 999999; maxRed = 0;

  for (uint8_t i = 0; i < recalSamples; i++) {
    if (millis() - recalStartTime > recalTimeout) {
      heartStatus = "Recal Timeout";
      return false;
    }

    // Pembacaan dengan filter kondisional
    float redValue = particleSensor.getRed();
    float irValue = particleSensor.getIR();
    
    static float filteredRed = redValue;
    filteredRed = isHighMotion ? 
      0.7f * filteredRed + 0.3f * redValue : // Filter agresif untuk gerakan
      0.9f * filteredRed + 0.1f * redValue;  // Filter halus untuk diam

    // Akumulasi data
    redSum += filteredRed;
    irSum += irValue;
    minRed = min(minRed, filteredRed);
    maxRed = max(maxRed, filteredRed);

    // Progress bar mini
    if (i % 5 == 0) {
      display.fillRect(0, 50, map(i, 0, recalSamples, 0, 128), 3, SSD1306_WHITE);
      display.display();
    }
    delay(sampleInterval);
  }

  //=============== ADAPTIVE THRESHOLD CALCULATION ===============//
  dcFiltered = redSum / recalSamples;
  float signalDynamicRange = maxRed - minRed;

  // Versi sederhana dari formula hybrid
  if (isHighMotion) {
    float irRatio = (irSum / recalSamples) / 10000.0f;
    contactThreshold = dcFiltered * (0.28f * constrain(irRatio, 0.8f, 1.2f));
  } 
  else {
    if (signalDynamicRange < 5000.0f) {
      contactThreshold = dcFiltered * 0.3f;
    } else {
      dynamicFactor = map(constrain(signalDynamicRange, 500.0f, 5000.0f), 500.0f, 5000.0f, 40, 25) / 100.0f;
      contactThreshold = dcFiltered * dynamicFactor;
    }
  }

  //=============== FINAL VALIDATION ===============//
  if (signalDynamicRange < minDynamicRange) {
    heartStatus = "Weak Signal";
    return false;
  }

  //=============== FINALIZATION ===============//
  needsCalibration = false;
  sensorContactLost = false;
  lastContactTime = millis();
  
  // [BARU] Gunakan reset mode post-calibration
  resetFilters(true); 
  
  // [BARU] Update threshold secara konsisten
  dynamicThreshold = contactThreshold / max(dcFiltered, 1.0f);

  Serial.printf("Recal Success: DC=%.1f Thresh=%.1f Motion=%d\n",
               dcFiltered, contactThreshold, isHighMotion);
  
  heartStatus = "Measuring...";
  return true;
}

int expectedPeaks() {
    if (isnan(heartRate)) return NAN;
    return round(heartRate/60);  // Menghitung jumlah peak yang diharapkan per detik
}

void resetFilters(bool isPostCalibration) {
  //====== [0] NAN SAFETY CHECK ======//
  if(isnan(heartRate)) {
    heartRate = instantHR = lastValidHR = 0;
  }
  
  //====== [1] ENHANCED SAFETY CHECK ======//
  if (!particleSensor.begin() || particleSensor.getRed() < 10.0f) {
    Serial.println("Sensor Error/Unstable! Reset aborted.");
    systemStatus = ERROR_SENSOR;
    return;
  }

  //====== [2] CORE RESET ======//
  bufferIndex = 0;
  bufferFull = false;

  // ==== TAMBAH INI: BUFFER BOUNDS SAFETY ====
  if (BUFFER_SIZE <= 0) {
      Serial.println("ERROR: BUFFER_SIZE invalid!");
      systemStatus = ERROR_SENSOR;
      return;
  }

  // ==== TAMBAH INI: INITIALIZE BUFFER DENGAN NILAI DEFAULT ====
  for (int i = 0; i < BUFFER_SIZE; i++) {
      ppgBuffer[i] = 0.5f; // Nilai tengah yang reasonable
  }

  const float currentRed = particleSensor.getRed();

  //====== [3] HYBRID DC OFFSET ======//
  dcFiltered = isPostCalibration ? 
    dcFiltered * 0.94f + currentRed * 0.06f : // Nilai optimal berdasarkan tes
    currentRed;

  //====== [4] SMART THRESHOLD ======//
  float hrContext = constrain(heartRate, 40.0f, 180.0f);
  dynamicThreshold = isPostCalibration ?
    (contactThreshold * (0.15f + 0.05f*(hrContext/180.0f))) / max(dcFiltered,1.0f) :
    lerp(0.08f, 0.15f, hrContext/180.0f);  // DRASTIS LEBIH RENDAH

  //====== [5] OPTIMAL MOTION COMP ======//
  if (motionContext.isActive()) {
    dynamicThreshold *= 1.2f; // Nilai empiris terbaik
    lastMotionTime = millis();
  }

  // Reset stillness counter
  stillnessCounter = 0; // TAMBAH LINE INI

  //====== [6] ADAPTIVE HISTORY INIT ======//
  const float initValue = currentRed / max(dcFiltered, 1.0f);
  float baseSmooth = isPostCalibration ? 0.75f : 0.8f;
  for (int i = 0; i < FILTER_WINDOW_SIZE; i++) {
    lastValues[i] = initValue * (baseSmooth + (1.0f-baseSmooth)*(i/(FILTER_WINDOW_SIZE-1)));
  }

  //====== [7] SAFE PEAK DETECTION ======//
  const float currentBPM = (hrContext > 0) ? hrContext : 72.0f;
  lastPeakTime = millis() - (60000.0f / currentBPM);
  nextExpectedPeak = lastPeakTime + constrain(
    60000.0f / currentBPM * (motionContext.isActive() ? 0.95f : 1.05f),
    400.0f, 1200.0f); // Batasan lebih aman

  //====== [8] DIAGNOSTIC LOGGING ======//
  Serial.printf("[OptReset] HR=%.1f Th=%.3f DC=%.1f Cal=%d Mot=%d\n",
    hrContext, dynamicThreshold, dcFiltered, isPostCalibration, motionContext.isActive());

  // ==== TAMBAH INI: BUFFER STATUS LOG ====
  Serial.printf("[BufferReset] Index=%d, Full=%d, Size=%d\n",
    bufferIndex, bufferFull, BUFFER_SIZE);

  //====== [9] HEALTH MONITORING RESET ======//
  systemHealthScore = 100;
  consecutiveErrors = 0;
  Serial.println("[RESET] Filters and health monitoring reset");
}

void checkSystemHealth() {
  if (!HEALTH_MONITORING) return;
  
  static unsigned long startupTime = millis();
  unsigned long currentTime = millis();
  
  // GRACE PERIOD 15 detik untuk inisialisasi
  if (currentTime - startupTime < 15000) {
    return;
  }
  
  if (currentTime - lastHealthCheck < 2000) return;
  
  lastHealthCheck = currentTime;
  int previousHealth = systemHealthScore;
  
  // === CHECK 1: Valid HR Range ===
  // PERBAIKAN: Terima nilai 0 sebagai valid
  if (isnan(heartRate) || (heartRate != 0 && (heartRate < 40.0f || heartRate > 220.0f))) {
    systemHealthScore -= 15;
    consecutiveErrors++;
    Serial.printf("[HEALTH] Invalid HR: %.1f BPM\n", heartRate);
  }
  
  // === CHECK 2: Sensor Contact ===
  if (sensorContactLost && millis() - contactLostTime > 10000) {
    systemHealthScore -= 10;
    Serial.println("[HEALTH] Sensor contact lost too long");
  }
  
  // === CHECK 3: Peak Detection Performance ===
  if (totalPeaksExpected > 20) {
    float detectionRate = (float)totalPeaksDetected / totalPeaksExpected;
    if (detectionRate < 0.3f) {
      systemHealthScore -= 20;
      Serial.printf("[HEALTH] Poor peak detection: %.1f%%\n", detectionRate*100);
    }
  }
  
  // === CHECK 4: Motion vs HR Consistency ===
  // PERBAIKAN: Exclude nilai 0 dan startup period
  if (motionLevel > 50000 && heartRate != 0 && heartRate < 80.0f && !isnan(heartRate)) {
    systemHealthScore -= 10;
    Serial.printf("[HEALTH] Inconsistency: High motion (%.0f) but low HR (%.1f)\n", 
                 motionLevel, heartRate);
  }
  
  // === AUTO-RECOVERY MECHANISM ===
  if (systemHealthScore < 50 || consecutiveErrors > 5) {
    Serial.println("[HEALTH] Critical error! Auto-recovering...");
    
    // ✅ TAMBAH COOLDOWN CHECK SEBELUM MEMANGGIL RECOVERY
    unsigned long sinceLastRecovery = millis() - lastEmergencyRecovery;
    if (sinceLastRecovery > RECOVERY_COOLDOWN_MS) {
      performEmergencyRecovery();
    } else {
      Serial.printf("[HEALTH] Recovery cooldown: %lu/%lu ms\n", 
                  sinceLastRecovery, RECOVERY_COOLDOWN_MS);
    }
  }
  
  // === GRADUAL HEALTH RECOVERY ===
  if (systemHealthScore < 100 && consecutiveErrors == 0) {
    systemHealthScore = min(100, systemHealthScore + 2);
  }
  
  // === RESET ERROR COUNT IF HEALTHY ===
  if (systemHealthScore > 80) {
    consecutiveErrors = 0;
  }
  
  if (systemHealthScore != previousHealth) {
    Serial.printf("[HEALTH] Score: %d%%\n", systemHealthScore);
  }
}

void performEmergencyRecovery() {
  // ✅ COOLDOWN CHECK - Jangan jalankan recovery terlalu sering
  unsigned long sinceLastRecovery = millis() - lastEmergencyRecovery;
  if (sinceLastRecovery < RECOVERY_COOLDOWN_MS) {
    Serial.printf("RECOVERY COOLDOWN: Skipping, %lu ms remaining\n", 
                 RECOVERY_COOLDOWN_MS - sinceLastRecovery);
    return;
  }
  
  Serial.println("[RECOVERY] Performing emergency reset...");
  recoveryInProgress = true;
  lastEmergencyRecovery = millis();  // ✅ UPDATE TIME FIRST
  
  // Soft reset semua filter dan state
  resetFilters(true);
  
  // Reset health metrics
  systemHealthScore = 80;
  consecutiveErrors = 0;
  
  // Force recalibration jika perlu - ✅ IMPROVED CONDITION
  if (sensorContactLost || millis() - lastContactTime > 30000 || needsCalibration) {
    needsCalibration = true;
    Serial.println("[RECOVERY] Forcing recalibration");
  }
  
  // Reset buffer dan processing state
  bufferIndex = 0;
  bufferFull = false;
  lastProcessingTimeMs = 0;
  
  // ✅ IMPROVED RECOVERY LOGIC (NEW)
  if (lastStableHR > 40.0f && lastStableHR < 220.0f) {
    if (millis() < 30000) { // 30 detik pertama
      // Lebih aggressive di awal
      instantHR = 0.5f * instantHR + 0.5f * lastStableHR;
      Serial.printf("[RECOVERY] Aggressive restore: %.1f BPM\n", instantHR);
    } else {
      // Lebih conservative setelah stabil
      instantHR = 0.8f * instantHR + 0.2f * lastStableHR;
      Serial.printf("[RECOVERY] Conservative restore: %.1f BPM\n", instantHR);
    }
  }
  
  Serial.println("[RECOVERY] System reset completed");
}

float validateHeartRate(float hr) {
  if (!SAFETY_CHECKS) return hr;
  
  static float lastValidHR = 72.0f;
  static unsigned long lastValidTime = 0;

  // ✅ CHECK SENSOR CONTACT TERLEBIH DAHULU
  if (sensorContactLost) {
    return NAN; // ← KEMBALIKAN NaN JIKA SENSOR LEPAS
  }
  
  // Handle invalid values
  if (isnan(hr) || isinf(hr)) {
    return lastValidHR; // Return last valid, bukan NaN
  }
  
  // ✅ RANGE YANG LEBIH REALISTIS untuk aktivitas
  if (hr < 50.0f || hr > 210.0f) {
    Serial.printf("[SAFETY] Extreme HR: %.1f, using last valid: %.1f\n", hr, lastValidHR);
    return lastValidHR;
  }
  
  // Smoothing untuk perubahan drastis
  if (lastValidTime > 0 && millis() - lastValidTime < 5000) {
    float change = abs(hr - lastValidHR);
    if (change > 35.0f) {
      // Smooth transition, bukan reject
      float smoothed = (lastValidHR * 0.7f + hr * 0.3f);
      Serial.printf("[SAFETY] Smoothing change: %.1f → %.1f → %.1f\n", 
                   lastValidHR, hr, smoothed);
      hr = smoothed;
    }
  }
  
  // Update last valid
  lastValidHR = hr;
  lastValidTime = millis();
  return hr;
}

void updateDynamicThresholdLearning() {
  if (!ADAPTIVE_LEARNING) return;
  
  static unsigned long lastUpdate = 0;
  if (millis() - lastUpdate < 5000) return; // Update setiap 5 detik
  
  lastUpdate = millis();
  
  // ✅ PERBAIKAN 1: Gunakan expected peaks berdasarkan instantHR jika heartRate = 0
  int expectedPeaksValue = 0;
  if (totalPeaksExpected > 0) {
    expectedPeaksValue = totalPeaksExpected;
  } else if (instantHR > 40.0f && !isnan(instantHR)) {
    expectedPeaksValue = round(instantHR / 60.0f); // Fallback ke instantHR
  } else {
    expectedPeaksValue = 1; // Minimum expected
  }
  
  // ✅ PERBAIKAN 2: Hitung success rate dengan expected value yang reasonable
  float successRate = (expectedPeaksValue > 0) ? 
                     (float)totalPeaksDetected / expectedPeaksValue : 0.0f;
  
  // ✅ PERBAIKAN 3: Reset mechanism ketika stuck terlalu lama
  if (totalPeaksDetected == 0 && millis() > 30000) {
    dynamicThreshold = 0.05f; // Reset ke reasonable value
    Serial.println("RESET: Threshold reset to 0.05 (no peaks detected)");
    
    // Reset counters
    totalPeaksDetected = 0;
    totalPeaksExpected = 0;
    return;
  }
  
  // ✅ PERBAIKAN 4: Adjust threshold dengan bounds protection
  if (successRate < 0.20f) {
    // Terlalu banyak miss, turunkan threshold tapi jangan terlalu rendah
    float newThreshold = dynamicThreshold * 0.98f;
    dynamicThreshold = max(newThreshold, 0.01f); // Batas bawah 0.01
    Serial.printf("[LEARNING] Lowering threshold to %.4f (success: %.1f%%)\n",
                 dynamicThreshold, successRate*100);
  } 
  else if (successRate > 0.80f && motionLevel < 5000) {
    // Terlalu banyak false positive, naikkan threshold
    float newThreshold = dynamicThreshold * 1.02f;
    dynamicThreshold = min(newThreshold, 0.2f); // Batas atas 0.2
    Serial.printf("[LEARNING] Raising threshold to %.4f (success: %.1f%%)\n",
                 dynamicThreshold, successRate*100);
  }
  
  // ✅ PERBAIKAN 5: Batasi threshold dalam range yang lebih reasonable
  dynamicThreshold = constrain(dynamicThreshold, 0.01f, 0.2f);
  lastGoodThreshold = dynamicThreshold;
  
  // Reset counters
  totalPeaksDetected = 0;
  totalPeaksExpected = 0;
}

float calculateSignalVariance() {
    if(bufferIndex < 2) return 0.0;
    
    float mean = 0.0f;
    int samples = min(bufferIndex, 32);
    
    for(int i = 0; i < samples; i++) {
        int idx = (bufferIndex - i + BUFFER_SIZE) % BUFFER_SIZE;
        mean += ppgBuffer[idx];
    }
    mean /= samples;

    float variance = 0.0f;
    for(int i = 0; i < samples; i++) {
        int idx = (bufferIndex - i + BUFFER_SIZE) % BUFFER_SIZE;
        variance += pow(ppgBuffer[idx] - mean, 2);
    }
    
    return variance / samples;
}

float findMax(float* arr, int n) {
  float maxVal = arr[0];
  for (int i = 1; i < n; i++) {
    if (arr[i] > maxVal) maxVal = arr[i];
  }
  return maxVal;
}

float findMin(float* arr, int n) {
  float minVal = arr[0];
  for (int i = 1; i < n; i++) {
    if (arr[i] < minVal) minVal = arr[i];
  }
  return minVal;
}

void checkSensorContact() {
  float redValue = particleSensor.getRed();
  float irValue = particleSensor.getIR();

  // ✅ DEFINISI THRESHOLDS YANG JELAS BERDASARKAN RAW VALUES
  const float RED_CONTACT_THRESHOLD = 10000.0f;    // Nilai minimum Red untuk kontak
  const float IR_CONTACT_THRESHOLD = 8000.0f;      // Nilai minimum IR untuk kontak
  const float MAX_SENSOR_VALUE = 250000.0f;        // Nilai maksimum valid

  // ✅ CHECK BERDASARKAN RAW VALUES - JELAS DAN PREDICTABLE
  bool isRedValid = (redValue >= RED_CONTACT_THRESHOLD) && (redValue <= MAX_SENSOR_VALUE);
  bool isIRValid = (irValue >= IR_CONTACT_THRESHOLD) && (irValue <= MAX_SENSOR_VALUE);
  
  // ✅ ADDITIONAL PLAUSIBILITY CHECKS
  bool isRedPlausible = (redValue > 1000.0f);      // Absolute minimum
  bool isIRPlausible = (irValue > 1000.0f);        // Absolute minimum
  bool isSignalRatioPlausible = (abs(redValue - irValue) < 50000.0f); // Kedua channel seharusnya tidak berbeda jauh

  bool hasContact = isRedValid && isIRValid && isRedPlausible && isIRPlausible && isSignalRatioPlausible;

  if (!hasContact) {
    if (!sensorContactLost) {
      if (++consecutiveInvalidReadings >= 2) { // ↑ Responsiveness
        sensorContactLost = true;
        heartRate = NAN;
        instantHR = NAN;
        lastValidHR = NAN; // ← ✅ RESET JUGA LAST VALID HR!
        heartStatus = "No Contact";
        contactLostTime = millis();
        
        Serial.printf("CONTACT_LOST: Red=%.0f IR=%.0f\n", redValue, irValue);
      }
    }
  } else {
    consecutiveInvalidReadings = 0;
    if (sensorContactLost) {
      sensorContactLost = false;
      heartStatus = "Measuring...";
      Serial.printf("CONTACT_REGained: Red=%.0f IR=%.0f\n", redValue, irValue);
      
      // ✅ Reset filters ketika kontak kembali
      resetFilters(true);
    }
  }

  // ✅ Update last values untuk delta calculation
  static float lastRedCheck = 0, lastIRCheck = 0;
  lastRedCheck = redValue;
  lastIRCheck = irValue;
}

float scaleToBPMRange(float value, float minRaw, float maxRaw) {
  return map(value, minRaw, maxRaw, 40, 200);
}

float generateHeartBeatWave(float bpm, float& phase) {
  static unsigned long lastUpdate = millis();
  unsigned long currentTime = millis();
  float deltaTime = (currentTime - lastUpdate) / 1000.0f; // dalam detik
  lastUpdate = currentTime;
  
  if (isnan(bpm)) return 120.0f;

  // Pastikan BPM dalam range valid
  bpm = constrain(bpm, 40.0f, 200.0f); 
  
  // Hitung frekuensi dalam radian/detik
  float frequency = (bpm / 60.0f) * TWO_PI;
  
  // Update phase dengan waktu aktual
  phase += frequency * deltaTime;
  
  // Bentuk gelombang yang lebih natural
  float wave = 0.6f * sin(phase) + 
               0.25f * sin(2.0f * phase) + 
               0.15f * sin(3.0f * phase + PI/3);
  
  // Scale ke range 40-200
  return 120.0f + 80.0f * wave;
}

void logDataToGoogleSheets() {
  // Hanya eksekusi setiap 1 detik
  static unsigned long lastLogTime = 0;
  unsigned long currentTime = millis();
  
  if (currentTime - lastLogTime < 1000) {
    return;
  }
  lastLogTime = currentTime;
  
  // Skip jika sensor tidak ada kontak atau data tidak valid
  if (sensorContactLost || isnan(heartRate) || heartRate <= 0) {
    return;
  }
  
  // Coba reconnect WiFi jika perlu
  if (!sheetsLoggerInitialized && millis() > 30000) { // Setelah 30 detik
    sheetsLoggerInitialized = dataLogger.begin();
  }
  
  if (!sheetsLoggerInitialized) {
    return;
  }
  
  // Dapatkan nilai sensor terkini
  float currentRed = particleSensor.getRed();
  float currentIR = particleSensor.getIR();
  
  // Hitung perubahan dari sample sebelumnya
  float deltaRed = currentRed - lastRedValue;
  float deltaIR = currentIR - lastIRValue;
  
  // Update nilai terakhir
  lastRedValue = currentRed;
  lastIRValue = currentIR;
  
  // Dapatkan nilai rawHR dari proses inference
  float rawHR = 0;
  if (interpreter && output) {
    rawHR = output->data.f[0]; // Nilai langsung dari model
  }
  
  // Kirim data ke Google Sheets - PERBAIKAN: gunakan rawHR untuk MODEL
  bool success = dataLogger.sendHRData(
    currentRed,      // RAW_RED
    currentIR,       // RAW_IR
    deltaRed,        // ΔRED
    deltaIR,         // ΔIR
    motionLevel,     // MOTION
    instantHR,       // INSTANT
    rawHR,           // MODEL ← PERBAIKAN: gunakan rawHR bukan heartRate
    heartRate,       // HR_FINAL (hasil akhir setelah processing)
    bufferIndex,     // BUFFER_IDX
    bufferFull,      // BUFFER_FULL
    BUFFER_SIZE      // BUFFER_SIZE
  );
  
  // Debug output
  static unsigned long lastStatusLog = 0;
  if (currentTime - lastStatusLog > 5000) {
    if (success) {
      Serial.println("✓ Data sent to Google Sheets");
      Serial.printf("DATA: RawHR=%.1f, FinalHR=%.1f, InstantHR=%.1f\n", 
                   rawHR, heartRate, instantHR);
    } else {
      Serial.println("✗ Failed to send: " + dataLogger.getLastError());
    }
    lastStatusLog = currentTime;
  }
}

void initSerialPlotter() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("PPG_Signal,PPG_Normalized,Threshold,Heartbeat_Wave,HR_Model,Instant_HR,Peak_Marker");
  Serial.println("All values scaled to 40-200 BPM equivalent range");
}

void updateSerialPlotter(bool isPeak) {
  // Parameter skala
  const float PPG_MIN = 0.0f;
  const float PPG_MAX = 1.0f;
  const float RAW_MIN = 30000.0f;
  const float RAW_MAX = 120000.0f;
  
  // Dapatkan nilai terkini
  float currentRaw = particleSensor.getRed();
  float currentNormalized = ppgBuffer[(bufferIndex - 1 + BUFFER_SIZE) % BUFFER_SIZE];
  
  // Skala nilai untuk visualisasi
  static float minNorm = min(PPG_MIN, currentNormalized);
  static float maxNorm = max(PPG_MAX, currentNormalized);
  float scaledRaw = scaleToBPMRange(currentRaw, RAW_MIN, RAW_MAX);
  float scaledNormalized = scaleToBPMRange(currentNormalized, PPG_MIN, PPG_MAX);
  float scaledThreshold = scaleToBPMRange(dynamicThreshold, PPG_MIN, PPG_MAX);
  
  // Generate synthetic heartbeat wave - HANDLE ZERO VALUES
  static float wavePhase = 0;
  float targetBPM = (heartRate <= 0) ? 72.0f : heartRate; // Fallback hanya untuk wave
  float heartbeatWave = generateHeartBeatWave(targetBPM, wavePhase);
  
  // Peak marker
  float peakMarker = isPeak ? 1.0f : 0.0f;
  
  // Handle zero values for display
  float displayHR = (heartRate <= 0) ? 0 : heartRate;
  float displayInstantHR = (instantHR <= 0) ? 0 : instantHR;
  
  // Format output untuk Serial Plotter - FORMAT YANG BENAR
  Serial.print("Raw:"); Serial.print(scaledRaw); Serial.print(",");
  Serial.print("Normalized:"); Serial.print(scaledNormalized); Serial.print(",");
  Serial.print("Threshold:"); Serial.print(scaledThreshold); Serial.print(",");
  Serial.print("HR_Wave:"); Serial.print(heartbeatWave); Serial.print(",");
  Serial.print("HR_Model:"); Serial.print(displayHR); Serial.print(",");
  Serial.print("Instant_HR:"); Serial.print(displayInstantHR); Serial.print(",");
  Serial.print("Peak:"); Serial.println(peakMarker);
  // Bisa tambahkan ini di akhir jika mau:
  Serial.print("Status:"); 
  if (heartRate <= 0) Serial.print("NoData");
  else if (sensorContactLost) Serial.print("NoContact"); 
  else Serial.print("Measuring");
  Serial.print(",");
}

void markPeakForPlotter(bool isPeak) {
  updateSerialPlotter(isPeak);
}

void checkHeartRateCondition(float hr) {
  // ✅ PERBAIKAN: Handle NaN values terlebih dahulu
  if (isnan(hr)) {
    heartStatus = "Measuring...";
    alertActive = false;
    return;
  }

  if (hr > HR_ALERT_HIGH) {
    if (hr < HIGH_INTENSITY_THRESHOLD + 20.0f) {
        heartStatus = "ALERT: Tachycardia";
        lastAlertTime = millis();
        alertActive = true;
    } else {
        heartStatus = "High HR (Exercise)";
    }
  } else if (hr > 100) {
    heartStatus = "Elevated HR";
    alertActive = false;
  } else if (hr < HR_ALERT_LOW) {
    heartStatus = "ALERT: Bradycardia";
    lastAlertTime = millis();
    alertActive = true;
  } else {
    heartStatus = "Normal";
    alertActive = false;
  }
}

void printSystemStatus() {
  // Header dengan pembatas visual
  Serial.println("\n╔════════════════════════════════════════════════╗");
  Serial.println("║               SYSTEM STATUS (ADAPTIVE)         ║");
  Serial.println("╠════════════════════════════════════════════════╣");

  // Seksi 1: Informasi Inti
  Serial.printf("║ Time: %-6lu s     HR: %-3.0f BPM              ║\n", millis() / 1000, heartRate);
  Serial.printf("║ Motion: %-6.0f     Contact: %-3s              ║\n", motionLevel, 
              sensorContactLost ? "NO" : "YES");

  // Seksi 2: Detail Sinyal
  Serial.println("╠────────────────────────────────────────────────╣");
  Serial.printf("║ Instant HR: %-3.0f BPM (Δ: %-3.0f)                 ║\n", instantHR, abs(heartRate - instantHR));
  
  // Adaptive parameters info
  float motionIntensity = constrain(motionLevel / 60000.0f, 0.0f, 1.0f) * 100.0f;
  Serial.printf("║ Motion Intensity: %-3.0f%%   Smooth: %-3.0f%%       ║\n", 
               motionIntensity, adaptiveSmoothFactor * 100);

  // Seksi 3: Processing Info
  Serial.println("╠────────────────────────────────────────────────╣");
  Serial.printf("║ Proc: %-4.1fms   Peaks: %-2d/%-2d   Step: %-2d    ║\n", 
               lastProcessingTimeMs, peakCountLastSecond, expectedPeaks(), adaptiveStepSize);
  Serial.printf("║ Status: %-30s ║\n", heartStatus.c_str());

  // Seksi 4: Fusion Weights
  Serial.println("╠────────────────────────────────────────────────╣");
  Serial.printf("║ Fusion: Model=%-3.0f%% Instant=%-3.0f%%           ║\n",
               adaptiveModelWeight * 100, adaptiveInstantWeight * 100);
  
  // Health score
  Serial.printf("║ Health: %-3d%%   Errors: %-2d                  ║\n",
               systemHealthScore, consecutiveErrors);

  // Footer
  Serial.println("╚════════════════════════════════════════════════╝");

  // Reset counter peak per detik
  peakCountLastSecond = 0;
}

void enhancedDebugOutput() {
  if (!DEBUG_MODE) return;
  
  static unsigned long lastDebug = 0;
  if (millis() - lastDebug < 2000) return;
  lastDebug = millis();
  
  // Dapatkan nilai sensor terkini
  float currentRed = particleSensor.getRed();
  float currentIR = particleSensor.getIR();
  
  Serial.println("\n╔══════════════════════════════════════════════════════════════╗");
  Serial.println("║                  ENHANCED DEBUG - ADAPTIVE MODE             ║");
  Serial.println("╠══════════════════════════════════════════════════════════════╣");
  
  // Line 1: Core HR Metrics
  Serial.printf("║ HR: %-5.1f | Instant: %-5.1f | Model: %-5.1f | Avg: %-3d       ║\n", 
               heartRate, instantHR, lastValidHR, beatAvg);
  
  // Line 2: Motion & Detection
  Serial.printf("║ Motion: %-6.0f (%-2.0f%%) | Peaks: %-2d/%-2d (%-3.0f%%)        ║\n",
               motionLevel, (motionLevel/100000.0f) * 100,
               totalPeaksDetected, totalPeaksExpected,
               (totalPeaksExpected > 0) ? (100.0f * totalPeaksDetected / totalPeaksExpected) : 0);
  
  // Line 3: Adaptive Parameters
  Serial.printf("║ Smooth: %-3.0f%% | Scale: %-6.0f | Step: %-2d                 ║\n",
               adaptiveSmoothFactor * 100, adaptiveMotionScale, adaptiveStepSize);
  
  // Line 4: Threshold & Health
  Serial.printf("║ Threshold: %-6.4f | Health: %-3d%% | Errors: %-2d            ║\n",
               dynamicThreshold, systemHealthScore, consecutiveErrors);
  
  // Line 5: Fusion Weights & Processing
  Serial.printf("║ Fusion: M=%-3.0f%% I=%-3.0f%% | Proc: %-4.1fms               ║\n",
               adaptiveModelWeight * 100, adaptiveInstantWeight * 100, lastProcessingTimeMs);
  
  // Line 6: Signal Quality & Contact - ✅ PERBAIKI DI SINI
  float signalQuality = calculateSignalQuality(currentRed, currentIR); // ✅ TAMBAH PARAMETER
  Serial.printf("║ Contact: %-3s | Buffer: %-3d/%-3d | Quality: %-2.0f%%        ║\n",
               sensorContactLost ? "NO" : "YES", 
               bufferIndex, BUFFER_SIZE,
               signalQuality * 100); // ✅ SUDAH DIPERBAIKI
  
  Serial.println("╚══════════════════════════════════════════════════════════════╝");
  
  // Reset counters setelah debug output
  totalPeaksDetected = 0;
  totalPeaksExpected = 0;
}

// Tambahkan di bagian fungsi support
float calculateSignalQuality(float redValue, float irValue) {
    static float lastRed = 0, lastIR = 0;
    
    // ✅ 1. INSTANT REAL-TIME SPIKE DETECTION (always works)
    float redChange = abs(redValue - lastRed);
    float irChange = abs(irValue - lastIR);
    
    // Instant spike rejection - sangat conservative
    if (redChange > 30000.0f || irChange > 40000.0f) {
        return 0.05f; // Immediate rejection untuk extreme spikes
    }
    
    // ✅ 2. REAL-TIME QUALITY COMPONENT (always available)
    float realTimeQuality = 1.0f - constrain(redChange / 8000.0f, 0.0f, 1.0f);
    realTimeQuality = min(realTimeQuality, 1.0f - constrain(irChange / 12000.0f, 0.0f, 1.0f));
    
    // ✅ 3. BUFFER-READINESS CHECK
    if (bufferIndex < 2) {
        return realTimeQuality; // Pure real-time jika buffer hampir kosong
    }
    
    // ✅ 4. ADAPTIVE BLENDING BASED ON BUFFER MATURITY
    int usableSamples = min(bufferIndex, 8); // Max 8 samples untuk efisiensi
    float bufferWeight = constrain((usableSamples - 2) / 6.0f, 0.0f, 1.0f);
    
    if (bufferWeight > 0.1f) { // Jika buffer cukup meaningful
        float mean = 0.0f;
        for (int i = 0; i < usableSamples; i++) {
            int idx = (bufferIndex - i - 1 + BUFFER_SIZE) % BUFFER_SIZE;
            mean += ppgBuffer[idx];
        }
        mean /= usableSamples;

        float variance = 0.0f;
        for (int i = 0; i < usableSamples; i++) {
            int idx = (bufferIndex - i - 1 + BUFFER_SIZE) % BUFFER_SIZE;
            variance += pow(ppgBuffer[idx] - mean, 2);
        }
        variance /= usableSamples;

        // ✅ Adaptive variance threshold based on sample count
        float maxVariance = 0.005f + (0.015f * (8 - usableSamples) / 6.0f);
        float bufferQuality = 1.0f - constrain(variance / maxVariance, 0.0f, 1.0f);
        
        // ✅ 5. SMART BLENDING: Real-time + Buffer
        return (bufferQuality * bufferWeight + realTimeQuality * (1.0f - bufferWeight));
    }
    
    lastRed = redValue;
    lastIR = irValue;
    return realTimeQuality;
}

bool initializeHardware() {
  if (!psramInit()) {
    Serial.println("PSRAM Init Failed");
    return false;
  }

  tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
  if (!tensor_arena) {
    Serial.println("Tensor Arena Alloc Failed");
    return false;
  }

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  if (OLED_RESET >= 0) {
    pinMode(OLED_RESET, OUTPUT);
    digitalWrite(OLED_RESET, HIGH);
    delay(10);
    digitalWrite(OLED_RESET, LOW);
    delay(10);
    digitalWrite(OLED_RESET, HIGH);
    delay(10);
  }

  for (uint8_t addr = 0x3C; addr <= 0x3D; addr++) {
    if (display.begin(SSD1306_SWITCHCAPVCC, addr)) {
      display.clearDisplay();
      display.display();
      delay(100);
      display.setTextSize(1);
      display.setTextColor(WHITE);
      display.setCursor(0, 0);
      display.println("Display Test");
      display.display();
      delay(1000);
      break;
    }
    if (addr == 0x3D) {
      Serial.println("OLED init failed");
      return false;
    }
  }

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    displayError("SENSOR ERROR", "Check Wiring");
    return false;
  }
  particleSensor.setup(0x1F, 4, 2, SAMPLE_RATE, 411, 4096);
  particleSensor.enableDIETEMPRDY();

  // ✅✅✅ TAMBAHKAN INISIALISASI DI SINI ✅✅✅
  float firstRed = particleSensor.getRed();
  float firstIR = particleSensor.getIR();

  // Safety check untuk nilai sensor tidak valid
  if (firstRed < 1000 || firstRed > 200000 || firstIR < 5000 || firstIR > 200000) {
    Serial.printf("WARNING: Invalid sensor values - Red: %.1f, IR: %.1f\n", firstRed, firstIR);
    // Fallback ke nilai default yang aman
    redMovingAvg = 50000.0f;
    irMovingAvg = 50000.0f;
  } else {
    // ==================== STEP 1: HYBRID INITIALIZATION ====================
    float typicalRed = 50000.0f;  // Nilai typical untuk RED → LEBIH CONSERVATIVE
    float typicalIR = 50000.0f;   // Nilai typical untuk IR → LEBIH CONSERVATIVE
    
    // Hybrid: 50% nilai aktual + 50% nilai typical
    redMovingAvg = (firstRed * 0.5f) + (typicalRed * 0.5f);
    irMovingAvg = (firstIR * 0.5f) + (typicalIR * 0.5f);
    
    // Safety constrain (tetap pertahankan)
    redMovingAvg = constrain(redMovingAvg, 30000.0f, 80000.0f);
    irMovingAvg = constrain(irMovingAvg, 30000.0f, 80000.0f);
    // =======================================================================
  }

  motionLevel = 300.0f;  // ← Lower initial motion (dari 500 ke 300)

  Serial.printf("Sensor initialized: Red=%.1f->%.1f, IR=%.1f->%.1f, Motion=%.1f\n", 
              firstRed, redMovingAvg, firstIR, irMovingAvg, motionLevel);

  static tflite::AllOpsResolver resolver;
  const tflite::Model* model = HR_MODEL_OPTIMIZED_HOPE_GET_MODEL();
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    displayError("MODEL ERROR", "Alloc Failed");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  return true;
}

void updateDisplay(const char* line1, const char* line2) {
  static unsigned long lastUpdate = 0;
  if (millis() - lastUpdate < 300 && !line1) return;
  lastUpdate = millis();

  display.clearDisplay();
  display.setTextSize(2); // ✅ UKURAN SAMA SEMUA
  
  // ================= LINE 1: HEART RATE =================
  display.setCursor(0, 0);
  display.print("HR:");
  if (isnan(heartRate)) {
    display.print(" --");
  } else {
    if (heartRate < 100) display.print(" ");
    display.print(heartRate, 0);
  }

  // ================= LINE 2: INSTANT HR =================  
  display.setCursor(0, 20);
  display.print("I:");
  if (isnan(instantHR)) {
    display.print(" --");
  } else {
    if (instantHR < 100) display.print(" ");
    display.print(instantHR, 0);
  }

  // ================= LINE 3: MOTION LEVEL =================
  display.setCursor(0, 40);
  display.print("MOT:");
  display.print((int)(motionLevel));

  // ================= LINE 4: STATUS =================
  display.setCursor(0, 60);
  display.setTextSize(1);
  if (sensorContactLost) {
    display.print("NO CONTACT");
  } else if (alertActive) {
    display.print("ALERT ACTIVE");
  } else {
    display.print("MEASURING");
  }
  
  // ================= LINE 5: TREND =================
  display.setCursor(70, 60);
  display.setTextSize(1);
  static float lastHR = 0;
  if (heartRate > lastHR + 3) {
    display.print("UP");
  } else if (heartRate < lastHR - 3) {
    display.print("DN");
  } else {
    display.print("ST");
  }
  lastHR = heartRate;

  display.display();
}

void handleAlerts() {
  if (sensorContactLost) {
    digitalWrite(BUZZER_PIN, LOW);
    return;
  }

  if (alertActive && millis() - lastAlertTime < 10000) {
    if (heartStatus.indexOf("Tachy") >= 0) {
      static unsigned long lastBuzz = 0;
      if (millis() - lastBuzz > 200) {
        digitalWrite(BUZZER_PIN, !digitalRead(BUZZER_PIN));
        lastBuzz = millis();
      }
    } else if (heartStatus.indexOf("Brady") >= 0) {
      static unsigned long lastBuzz = 0;
      if (millis() - lastBuzz > 500) {
        digitalWrite(BUZZER_PIN, !digitalRead(BUZZER_PIN));
        lastBuzz = millis();
      }
    } else if (heartStatus.indexOf("Fever") >= 0) {
      static unsigned long lastBuzz = 0;
      static bool secondBeep = false;

      if (millis() - lastBuzz > 300) {
        digitalWrite(BUZZER_PIN, !secondBeep ? HIGH : LOW);
        secondBeep = !secondBeep;
        lastBuzz = millis();
      }
    }
  } else {
    digitalWrite(BUZZER_PIN, LOW);
    alertActive = false;
  }
}

void displayError(const char* line1, const char* line2) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("ERROR:");
  display.println(line1);
  display.println(line2);
  display.display();

  while (1) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(500);
    digitalWrite(BUZZER_PIN, LOW);
    delay(500);
  }
}