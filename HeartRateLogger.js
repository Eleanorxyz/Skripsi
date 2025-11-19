function doPost(e) {
  var sheet = getSheet();
  var timestamp = new Date();
  
  try {
    var data = JSON.parse(e.postData.contents);
    
    // FORMAT BARU: Tambah kolom DATE dan TIME
    var dateFormatted = Utilities.formatDate(timestamp, Session.getScriptTimeZone(), "yyyy-MM-dd");
    var timeFormatted = Utilities.formatDate(timestamp, Session.getScriptTimeZone(), "HH:mm:ss");
    
    var rowData = [
      dateFormatted,                      // DATE (YYYY-MM-DD)
      timeFormatted,                      // TIME (HH:MM:SS)
      data.raw_red || "",
      data.raw_ir || "",
      data.delta_red || "",
      data.delta_ir || "",
      data.motion || "",
      data.instant_hr || "",
      data.model_hr || "",
      data.final_hr || "",
      data.activity || getActivityLevel(data.motion),
      data.buffer_idx || 0,              // BUFFER_IDX ← BARU
      data.buffer_full || false,         // BUFFER_FULL ← BARU
      data.buffer_size || 0              // BUFFER_SIZE ← BARU
    ];
    
    sheet.appendRow(rowData);
    
    // Hapus data lama jika lebih dari 10000 rows
    if (sheet.getLastRow() > 10000) {
      sheet.deleteRows(2, 1000);
    }
    
    return ContentService.createTextResponse(JSON.stringify({
      status: "success",
      message: "Data received"
    }));
    
  } catch (error) {
    return ContentService.createTextResponse(JSON.stringify({
      status: "error",
      message: error.toString()
    }));
  }
}

function getSheet() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = spreadsheet.getSheetByName("HR_Data");
  
  if (!sheet) {
    sheet = spreadsheet.insertSheet("HR_Data");
    // TAMBAH 3 KOLOM BUFFER
    sheet.appendRow([
      "DATE", "TIME", "RAW_RED", "RAW_IR", "ΔRED", "ΔIR", 
      "MOTION", "INSTANT", "MODEL", "HR_FINAL", "ACTIVITY",
      "BUFFER_IDX", "BUFFER_FULL", "BUFFER_SIZE"  // ← KOLOM BARU
    ]);
    
    // Format header
    var headerRange = sheet.getRange("A1:N1");
    headerRange.setBackground("#d9ead3")
               .setFontWeight("bold")
               .setHorizontalAlignment("center");
    
    // Set column widths
    sheet.setColumnWidth(1, 100);  // DATE
    sheet.setColumnWidth(2, 80);   // TIME
    sheet.setColumnWidth(3, 120);  // RAW_RED
    sheet.setColumnWidth(4, 120);  // RAW_IR
    sheet.setColumnWidth(5, 80);   // ΔRED
    sheet.setColumnWidth(6, 80);   // ΔIR
    sheet.setColumnWidth(7, 100);  // MOTION
    sheet.setColumnWidth(8, 80);   // INSTANT
    sheet.setColumnWidth(9, 80);   // MODEL
    sheet.setColumnWidth(10, 100); // HR_FINAL
    sheet.setColumnWidth(11, 100); // ACTIVITY
    sheet.setColumnWidth(12, 80);   // BUFFER_IDX ← BARU
    sheet.setColumnWidth(13, 80);   // BUFFER_FULL ← BARU
    sheet.setColumnWidth(14, 80);   // BUFFER_SIZE ← BARU
    
    // Freeze header row
    sheet.setFrozenRows(1);
  }
  
  return sheet;
}

function getActivityLevel(motion) {
  if (!motion) return "UNKNOWN";
  motion = parseFloat(motion);
  
  if (motion < 1000) return "RESTING";
  if (motion < 5000) return "LOW";
  if (motion < 20000) return "MODERATE";
  if (motion < 50000) return "HIGH";
  return "EXTREME";
}

function doGet() {
  return ContentService.createTextResponse("Heart Rate Logger Ready");
}