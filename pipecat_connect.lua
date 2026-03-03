-- pipecat_connect.lua
-- Güncel versiyon - zombie_exec + park

local uuid     = session:getVariable("uuid")
local call_id  = session:getVariable("pipecat_call_id") or uuid
local ws_url   = session:getVariable("pipecat_ws_url")

if not ws_url or ws_url == "" then
    freeswitch.consoleLog("ERR", "[Pipecat] pipecat_ws_url bulunamadı!\n")
    session:hangup()
    return
end

freeswitch.consoleLog("INFO", string.format("[Pipecat] Başlatılıyor → call_id=%s uuid=%s ws=%s\n", call_id, uuid, ws_url))

-- Çağrıyı cevapla (güvenlik için)
if not session:answered() then
    session:answer()
end

-- ZOMBIE EXEC FLAG → en önemli satır!
session:execute("set_zombie_exec")
freeswitch.consoleLog("INFO", "[Pipecat] Zombie exec flag aktif edildi\n")

session:sleep(1000)

-- Stream başlat (mod_audio_stream önerilen format)
local stream_cmd = string.format("uuid_audio_stream %s start %s mono 8000", uuid, ws_url)
local api = freeswitch.API()
local result = api:executeString(stream_cmd)

freeswitch.consoleLog("INFO", "[Pipecat] Stream result: " .. tostring(result) .. "\n")

if tostring(result):find("OK") then
    freeswitch.consoleLog("INFO", "[Pipecat] Stream BAŞARILI - Pipecat WS bağlantısı kurulmalı!\n")
end

-- Park ile channel'ı canlı tut (zombie olduğu için BYE gelse bile Lua devam eder)
freeswitch.consoleLog("INFO", "[Pipecat] Park başlıyor (zombie mod)...\n")
session:execute("park", "")

freeswitch.consoleLog("INFO", "[Pipecat] Script sonlandı.\n")
