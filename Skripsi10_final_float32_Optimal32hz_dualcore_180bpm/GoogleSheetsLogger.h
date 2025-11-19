#ifndef GOOGLE_SHEETS_LOGGER_H
#define GOOGLE_SHEETS_LOGGER_H

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

class GoogleSheetsLogger {
public:
  GoogleSheetsLogger(const char* scriptURL, const char* ssid, const char* password);
  
  bool begin();
  // TAMBAH PARAMETER BUFFER
  bool sendHRData(float rawRed, float rawIR, float deltaRed, float deltaIR, 
                 float motion, float instantHR, float modelHR, float finalHR,
                 int bufferIndex, bool bufferFull, int bufferSize); // ← PARAMETER BARU
  bool isConnected();
  String getLastError();

private:
  const char* _scriptURL;
  const char* _ssid;
  const char* _password;
  String _lastError;
  unsigned long _lastSendTime = 0;
  
  bool connectWiFi();
};

#endif