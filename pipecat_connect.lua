-- pipecat_connect.lua
-- FreeSWITCH bu scripti çağrı cevaplandıktan sonra çalıştırır.
-- Channel variable'lardan pipecat WS URL'ini alır ve audio_stream başlatır.

local call_id = session:getVariable("pipecat_call_id")
local ws_url  = session:getVariable("pipecat_ws_url")

if not ws_url then
    freeswitch.consoleLog("ERR", "pipecat_ws_url variable bulunamadı!\n")
    session:hangup()
    return
end

freeswitch.consoleLog("INFO", string.format(
    "Pipecat bağlantısı başlatılıyor. call_id=%s ws_url=%s\n", 
    tostring(call_id), ws_url
))

-- Çağrıyı cevapla
session:answer()

-- 500ms bekle (cevap sonrası stabilizasyon)
session:sleep(500)

-- mod_audio_stream ile Pipecat WebSocket'e bağlan
-- Parametreler: url, stereo(0/1), sample_rate, codec
session:execute("audio_stream", ws_url .. " 0 8000 L16")

freeswitch.consoleLog("INFO", "audio_stream tamamlandı, çağrı kapatılıyor.\n")
session:hangup()
