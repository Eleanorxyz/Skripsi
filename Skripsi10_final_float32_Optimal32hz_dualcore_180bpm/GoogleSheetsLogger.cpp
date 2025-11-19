#include "GoogleSheetsLogger.h"

GoogleSheetsLogger::GoogleSheetsLogger(const char* scriptURL, const char* ssid, const char* password)
  : _scriptURL(scriptURL), _ssid(ssid), _password(password) {}

bool GoogleSheetsLogger::begin() {
  Serial.println("Initializing Google Sheets Logger...");
  WiFi.mode(WIFI_STA);
  return connectWiFi();
}

bool GoogleSheetsLogger::connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;
  
  Serial.print("Connecting to: ");
  Serial.println(_ssid);
  
  WiFi.begin(_ssid, _password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    return true;
  }
  
  _lastError = "WiFi failed";
  return false;
}

// TAMBAH PARAMETER BUFFER
bool GoogleSheetsLogger::sendHRData(float rawRed, float rawIR, float deltaRed, float deltaIR,
                                   float motion, float instantHR, float modelHR, float finalHR,
                                   int bufferIndex, bool bufferFull, int bufferSize) {
  
  if (millis() - _lastSendTime < 1000) {
    return true;
  }
  _lastSendTime = millis();
  
  if (!isConnected() && !connectWiFi()) {
    return false;
  }
  
  DynamicJsonDocument doc(512);
  doc["raw_red"] = rawRed;
  doc["raw_ir"] = rawIR;
  doc["delta_red"] = deltaRed;
  doc["delta_ir"] = deltaIR;
  doc["motion"] = motion;
  doc["instant_hr"] = instantHR;
  doc["model_hr"] = modelHR;
  doc["final_hr"] = finalHR;
  // TAMBAH DATA BUFFER
  doc["buffer_idx"] = bufferIndex;
  doc["buffer_full"] = bufferFull;
  doc["buffer_size"] = bufferSize;
  
  String payload;
  serializeJson(doc, payload);
  
  HTTPClient http;
  http.begin(_scriptURL);
  http.addHeader("Content-Type", "application/json");
  
  int httpCode = http.POST(payload);
  bool success = (httpCode == 200);
  
  if (!success) {
    _lastError = "HTTP Error: " + String(httpCode);
  }
  
  http.end();
  return success;
}

bool GoogleSheetsLogger::isConnected() {
  return WiFi.status() == WL_CONNECTED;
}

String GoogleSheetsLogger::getLastError() {
  return _lastError;
}