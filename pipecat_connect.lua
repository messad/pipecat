-- pipecat_connect.lua
-- FreeSWITCH bu scripti çağrı cevaplandıktan sonra çalıştırır (örneğin dialplan'da <action application="lua" data="pipecat_connect.lua"/>)
-- Channel variable'lardan pipecat WS URL'ini alır ve audio_stream başlatır.

local uuid     = session:getVariable("uuid")
local call_id  = session:getVariable("pipecat_call_id") or uuid  -- fallback olarak uuid kullan
local ws_url   = session:getVariable("pipecat_ws_url")

if not ws_url or ws_url == "" then
    freeswitch.consoleLog("ERR", "[pipecat_connect] pipecat_ws_url variable bulunamadı! Hangup.\n")
    session:hangup()
    return
end

freeswitch.consoleLog("INFO", string.format(
    "[pipecat_connect] Bağlantı başlatılıyor → call_id=%s  uuid=%s  ws_url=%s\n",
    tostring(call_id), uuid, ws_url
))

-- Çağrıyı cevapla (eğer dialplan'da answer yoksa)
if not session:answered() then
    session:answer()
    freeswitch.consoleLog("INFO", "[pipecat_connect] Call answered\n")
end

-- Kısa stabilizasyon beklemesi (bazı durumlarda RTP/media bug için gerekli)
session:sleep(800)   -- 500 yerine biraz daha uzun, güvenli olsun

-- mod_audio_stream ile WebSocket'e bağlan
-- Kullanım: freeswitch.API():executeString("uuid_audio_stream " .. uuid .. " start " .. ws_url .. " mono 8000")
-- mono/stereo: genelde mono (0) yeter, Pipecat tarafı destekliyorsa stereo da olur
-- sample rate: 8000 çoğu AI için yeterli, gerekirse 16000 yap

local stream_cmd = string.format(
    "uuid_audio_stream %s start %s mono 8000",
    uuid, ws_url
)

local api = freeswitch.API()
local result = api:executeString(stream_cmd)

if result then
    freeswitch.consoleLog("INFO", "[pipecat_connect] uuid_audio_stream başlatıldı → result: " .. tostring(result) .. "\n")
else
    freeswitch.consoleLog("ERR", "[pipecat_connect] uuid_audio_stream başarısız!\n")
end

-- Channel'ı hemen kapatMA → Pipecat konuşma bitene kadar canlı tutmamız lazım
-- 1. seçenek: sonsuz sleep loop (en basit test için)
freeswitch.consoleLog("INFO", "[pipecat_connect] Channel canlı tutuluyor (sleep loop)...\n")

while session:ready() do
    session:sleep(2000)   -- her 2 saniyede bir kontrol et
    -- İstersen periyodik log: freeswitch.consoleLog("DEBUG", "[pipecat_connect] hala canlı...\n")
end

freeswitch.consoleLog("INFO", "[pipecat_connect] Session bitti, script sonlanıyor.\n")

-- Alternatif (daha temiz) yaklaşımlar için yorum satırları:
-- session:execute("park", "")   -- ama zombie_exec sorunu çıkmasın diye önce set et
-- veya dialplan'da:
-- <action application="set" data="park_after_bridge=true"/>
-- <action application="set" data="api_zombie_exec=true"/>
