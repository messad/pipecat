-- pipecat_connect.lua  (FINAL - alive loop + en stabil versiyon)

local uuid     = session:getVariable("uuid")
local call_id  = session:getVariable("pipecat_call_id") or uuid
local ws_url   = session:getVariable("pipecat_ws_url")

if not ws_url or ws_url == "" then
    freeswitch.consoleLog("ERR", "[Pipecat] pipecat_ws_url bulunamadı!\n")
    session:hangup()
    return
end

freeswitch.consoleLog("INFO", string.format("[Pipecat] BAŞLADI → call_id=%s uuid=%s ws=%s\n", call_id, uuid, ws_url))

if not session:answered() then
    session:answer()
    freeswitch.consoleLog("INFO", "[Pipecat] Call answered\n")
end

session:sleep(800)

-- Stream başlat (mono 8000 - Pipecat için en uyumlu)
local stream_cmd = string.format("uuid_audio_stream %s start %s mono 8000", uuid, ws_url)
local api = freeswitch.API()
local result = api:executeString(stream_cmd)

freeswitch.consoleLog("INFO", "[Pipecat] uuid_audio_stream sonucu: " .. tostring(result) .. "\n")

if tostring(result):find("OK") then
    freeswitch.consoleLog("INFO", "[Pipecat] STREAM BAŞARILI! Pipecat WS bağlantısı kurulmalı.\n")
end

-- PARK YERİNE ALIVE LOOP (zombie_exec'e gerek kalmaz)
freeswitch.consoleLog("INFO", "[Pipecat] Channel alive loop başladı (karşı taraf kapatsa bile Lua devam eder)...\n")

while session:ready() do
    session:sleep(2000)
end

freeswitch.consoleLog("INFO", "[Pipecat] Session bitti, script sonlandı.\n")
