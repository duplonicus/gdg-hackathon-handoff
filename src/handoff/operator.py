"""The human rung: a small operator page in the browser.

When a question climbs past every model, it appears here with the camera frame, the state text
and one button per option. With voice on, the person can also answer out loud: the page records
the mic, ``voice.py`` proposes an option, and the page reads it back for a confirm. Speech never
answers by itself. While it waits, the robot holds still (the caller passes an ``idle``
callback that keeps stepping the sim). The page is served by a standard-library HTTP server in
a thread on 127.0.0.1, separate from the kit's control panel, which has no extension point for
custom prompts (its request handler is a closure inside ``start``).
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .decider import Question, RungAnswer, timed
from .voice import MAX_AUDIO_BYTES, VoiceInput, is_wav

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Operator</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
#q{font-size:1.3em;margin:8px 0}
.opt{display:inline-block;margin:6px;padding:12px 18px;font-size:1.1em;border-radius:8px;border:1px solid #888;background:#222;color:#eee;cursor:pointer}
.opt:hover{background:#444} small{color:#aaa} img{max-width:640px;border:1px solid #444;border-radius:6px}
pre{white-space:pre-wrap;color:#ccc;background:#1a1a1a;padding:10px;border-radius:6px}
#idle{color:#8a8}
#voice{margin:10px 6px} #voice.off{display:none} #speak.on{background:#a33;border-color:#e66}
#heard{margin:8px 0;font-size:1.1em} #heard:empty,#confirm:empty{display:none}
</style></head><body>
<h2>Robot operator</h2>
<div id="idle">No question pending. The robot is working.</div>
<div id="ask" style="display:none">
  <div id="q"></div><div id="opts"></div>
  <div id="voice" class="off">
    <button class="opt" id="speak" onclick="toggleSpeak()">Answer out loud</button>
    <div id="heard"></div><div id="confirm"></div>
  </div>
  <img id="frame" alt="camera frame"/>
  <pre id="state"></pre>
  <small id="meta"></small>
</div>
<script>
let current=null, rec=null, voiceTurn=0;
async function poll(){
  try{
    const r=await fetch('/api/pending'); const p=await r.json();
    if(p && p.id!==undefined){
      if(!current || current.id!==p.id){
        current=p; resetVoice(); document.getElementById('idle').style.display='none';
        document.getElementById('ask').style.display='block';
        document.getElementById('q').textContent=p.text;
        const o=document.getElementById('opts'); o.innerHTML='';
        for(const opt of p.options){const b=document.createElement('button');b.className='opt';
          b.textContent=opt+' - '+p.criteria[opt];b.onclick=()=>answer(p.id,opt,false);o.appendChild(b);}
        document.getElementById('voice').className=p.voice?'':'off';
        document.getElementById('frame').src=p.frame?('data:image/png;base64,'+p.frame):'';
        document.getElementById('state').textContent=p.state_text;
        document.getElementById('meta').textContent=(p.climbed.includes('ERROR')?'Models below could not answer (a model is down or failed): ':'Models below were not confident enough: ')+p.climbed;
      }
    } else { if(current) resetVoice(); current=null; document.getElementById('idle').style.display='block'; document.getElementById('ask').style.display='none'; }
  }catch(e){}
  setTimeout(poll,500);
}
async function answer(id,opt,spoken){ await fetch('/api/answer',{method:'POST',body:JSON.stringify({id:id,answer:opt,spoken:spoken})}); current=null; resetVoice(); }
function showHeard(text){ document.getElementById('heard').textContent=text; }
function dropRecording(){
  if(!rec) return; clearTimeout(rec.timer); rec.proc.disconnect(); rec.src.disconnect();
  rec.stream.getTracks().forEach(t=>t.stop()); rec.ctx.close();
  const done=rec; rec=null; document.getElementById('speak').classList.remove('on');
  document.getElementById('speak').textContent='Answer out loud'; return done;
}
function resetVoice(){ voiceTurn++; dropRecording(); showHeard(''); document.getElementById('confirm').innerHTML=''; }
async function toggleSpeak(){
  if(rec){ const done=dropRecording(); await send(done); return; }
  if(!current) return;
  resetVoice(); const turn=voiceTurn;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    if(turn!==voiceTurn){ stream.getTracks().forEach(t=>t.stop()); return; } // another click, an answer or a new question got there first
    const ctx=new AudioContext(); const src=ctx.createMediaStreamSource(stream);
    const proc=ctx.createScriptProcessor(4096,1,1); const chunks=[];
    proc.onaudioprocess=e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(proc); proc.connect(ctx.destination);
    rec={stream,ctx,src,proc,chunks,rate:ctx.sampleRate,id:current.id,timer:setTimeout(toggleSpeak,15000)};
    const b=document.getElementById('speak'); b.classList.add('on'); b.textContent='Listening. Click when done';
  }catch(e){ showHeard('No microphone: '+e.message+'. Use the buttons.'); }
}
function wavBytes(chunks,rate){
  const n=chunks.reduce((a,c)=>a+c.length,0); const v=new DataView(new ArrayBuffer(44+n*2));
  const tag=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  tag(0,'RIFF');v.setUint32(4,36+n*2,true);tag(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);tag(36,'data');v.setUint32(40,n*2,true);
  let o=44; for(const c of chunks) for(const s of c){v.setInt16(o,Math.max(-1,Math.min(1,s))*32767,true);o+=2;}
  return v.buffer;
}
async function send(done){
  const turn=voiceTurn; showHeard('Working out what you said...');
  try{
    const r=await fetch('/api/voice?id='+done.id,{method:'POST',headers:{'Content-Type':'audio/wav'},body:wavBytes(done.chunks,done.rate)});
    const h=await r.json(); if(turn!==voiceTurn) return;
    if(!r.ok){ showHeard(h.error||'That did not work. Use the buttons.'); return; }
    const said=h.transcript?'Heard: "'+h.transcript+'". ':'';
    if(!h.option){ showHeard(said+'I did not get a clear choice ('+h.reason+'). Say it again or use a button.'); return; }
    showHeard(said+'That means: '+h.option+'.');
    const c=document.getElementById('confirm'); c.innerHTML='';
    const yes=document.createElement('button'); yes.className='opt'; yes.id='confirm-yes'; yes.textContent='Confirm: '+h.option; yes.onclick=()=>answer(done.id,h.option,true); c.appendChild(yes);
    const no=document.createElement('button'); no.className='opt'; no.id='confirm-no'; no.textContent='Not what I meant'; no.onclick=resetVoice; c.appendChild(no);
    if(window.speechSynthesis) speechSynthesis.speak(new SpeechSynthesisUtterance('I heard '+h.option.replace('_',' ')+'. Please confirm.'));
  }catch(e){ if(turn===voiceTurn) showHeard('That did not work: '+e.message+'. Use the buttons.'); }
}
poll();
</script></body></html>"""


class OperatorPage:
    def __init__(self, port: int = 8790, open_browser: bool = False, voice: VoiceInput | None = None) -> None:
        self.port = port
        self.voice = voice
        self.last_spoken: dict[str, Any] | None = None  # what the person said for the last answer, when they spoke it
        self._pending: dict[str, Any] | None = None
        self._question: Question | None = None
        self._answers: dict[int, str] = {}
        self._heard: dict[int, dict[str, Any]] = {}
        self._spoken: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._next_id = 0
        # Film runs only (demo.py --film): the simulator's answer rides along in /api/pending as "film_truth" so the
        # scripted operator (scripts/film_answerer.py) answers as a person who can see the scene would. The page never
        # shows it. None everywhere else.
        self.truth_fn: Callable[[Question], str | None] | None = None
        page = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/api/pending":
                    with page._lock:
                        body = json.dumps(page._pending or {})
                    self._send(body, "application/json")
                else:
                    self._send(PAGE, "text/html")

            def do_POST(self) -> None:
                url = urlparse(self.path)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if url.path == "/api/voice":
                    self._voice(parse_qs(url.query).get("id", [""])[0], length)
                    return
                data = json.loads(self.rfile.read(max(length, 0)) or b"{}")
                if url.path == "/api/answer":
                    qid, answer = int(data["id"]), str(data["answer"])
                    with page._lock:
                        page._answers[qid] = answer
                        heard = page._heard.pop(qid, None)
                        if data.get("spoken") and heard and heard.get("option") == answer:
                            page._spoken[qid] = heard
                        if page._pending and page._pending["id"] == qid:
                            page._pending = None
                self._send("{}", "application/json")

            def _voice(self, raw_id: str, length: int) -> None:
                """Proposes an option and answers nothing: the page still has to post the person's confirm to /api/answer."""
                self.close_connection = True  # a refused body is left unread, so this connection cannot be reused
                if page.voice is None:
                    self._error(404, "voice answers are off")
                    return
                if not 0 < length <= MAX_AUDIO_BYTES:
                    self._error(413, f"audio must be 1 to {MAX_AUDIO_BYTES} bytes")
                    return
                wav = self.rfile.read(length)
                if not is_wav(wav):
                    self._error(400, "not a WAV file")
                    return
                with page._lock:
                    pending, question = page._pending, page._question
                if pending is None or question is None or str(pending["id"]) != raw_id:
                    self._error(409, "that question is no longer open")
                    return
                result = page.voice.hear(question, wav)
                with page._lock:
                    page._heard[pending["id"]] = result
                self._send(json.dumps(result), "application/json")

            def _error(self, status: int, message: str) -> None:
                self._send(json.dumps({"error": message}), "application/json", status)

            def _send(self, body: str, content_type: str, status: int = 200) -> None:
                raw = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:
                pass

        self._handler = Handler
        self._open_browser = open_browser

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler)
        self.port = self._server.server_address[1]  # port 0 asks the OS for a free one
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        if self._open_browser:
            webbrowser.open(self.url)

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    def ask(self, question: Question, state_text: str, image_b64: str | None, climbed: str, idle: Callable[[], None], timeout_s: float) -> str | None:
        with self._lock:
            qid = self._next_id
            self._next_id += 1
            self.last_spoken = None
            self._question = question
            self._pending = {
                "id": qid,
                "text": question.text,
                "options": list(question.options),
                "criteria": dict(question.criteria),
                "state_text": state_text,
                "frame": image_b64,
                "climbed": climbed,
                "voice": self.voice is not None,
            }
            if self.truth_fn is not None:
                try:
                    self._pending["film_truth"] = self.truth_fn(question)
                except Exception:  # noqa: BLE001 (a film hint must never stop the robot)
                    self._pending["film_truth"] = None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if qid in self._answers:
                    self.last_spoken = self._spoken.pop(qid, None)
                    return self._answers.pop(qid)
            idle()
            time.sleep(0.1)
        with self._lock:
            self._pending = None
            self._heard.pop(qid, None)
        return None


class HumanRung:
    """Asks the operator page; the robot holds (``idle`` keeps the sim stepping) until an answer or timeout."""

    name = "human"

    def __init__(self, page: OperatorPage, idle: Callable[[], None], timeout_s: float = 120.0, climbed: Callable[[], str] = lambda: "") -> None:
        self.page, self.idle, self.timeout_s, self.climbed = page, idle, timeout_s, climbed

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            print(
                f"  waiting for the operator at {self.page.url}: '{question.id}' ({self.climbed() or 'no model rung'}); "
                f"safe option '{question.safe_option}' after {self.timeout_s:.0f} s",
                flush=True,
            )
            chosen = self.page.ask(question, state_text, image_b64, self.climbed(), self.idle, self.timeout_s)
            if chosen is None:
                raise TimeoutError(f"no operator answer within {self.timeout_s:.0f} s")
            extra: dict[str, Any] = {"operator": True}
            if self.page.last_spoken:
                extra["spoken"] = self.page.last_spoken
            return {o: (1.0 if o == chosen else 0.0) for o in question.options}, extra

        return timed(self.name, question.options, call)
