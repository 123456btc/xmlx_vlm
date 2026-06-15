#!/usr/bin/env python3
"""
Local Browser Agent — Direct MLX + Chrome DevTools Protocol.
Handles iframes, Shadow DOM, ProseMirror editors automatically.
"""

import json, os, re, sys, time, asyncio, websockets, urllib.request

# ─── Config ──────────────────────────────────────────────────────────────────

MLX_URL = os.environ.get("MLX_URL", "http://localhost:5118")
CDP_URL = os.environ.get("CDP_URL", "http://127.0.0.1:9222")
MODEL = os.environ.get("MLX_MODEL_NAME", "mlx-community/diffusiongemma-26B-A4B-it-4bit")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "15"))

B, G, Y, R, D, BD, RS = "\033[94m", "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"

SYSTEM = """You are a browser agent. Return ONE JSON tool call per response.

TOOLS:
- navigate(url) — Go to URL
- snapshot() — Get page elements with UIDs + current URL/Title + top links. ALWAYS call after navigate/click.
- click(uid) — Click element
- type_text(uid, text) — Type into element
- scroll(direction) — "up" or "down"
- js(code) — Run JavaScript (use this when clicks don't work or to find elements)
- done(message) — Task complete. Call this when the user's request is fulfilled.

FORMAT: {"tool": "name", "args": {...}}
RULES:
- After EVERY navigate or click, call snapshot() to see the new page state.
- Be fast. No explanations, just JSON.
- NEVER click the same UID more than twice. If it didn't work, try js(code) or a different uid.
- If snapshot shows the SAME URL after a click/type/scroll, the page did NOT change. STOP repeating that action. Try: js(code) to interact, scroll, or navigate to a different URL.
- If the user's task is ONLY to search/navigate to a page, call done() once you reach the relevant page. You do NOT need to click into articles unless the user explicitly asked you to read them.
- If a click opens an image/lightbox overlay, close it with js(code): document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))
- Use js(code) liberally when the page is stuck. Examples:
  - Search: var input=document.querySelector('input[name=p],input[name=q],#ybar-sbq');input.value='query';document.querySelector('button[type=submit]').click()
  - Get price/text: var el=document.querySelector('.price, [data-coin-id]');el?el.innerText:'not found'
  - Get top links: Array.from(document.querySelectorAll('h3 a')).map(a=>a.innerText).slice(0,10).join(' | ')
  - Scroll: window.scrollBy(0,500)
- On Reddit: image posts open lightboxes. Close them first. To find the comment box, scroll down and look for a textbox or use js(code) to find it.
- CRITICAL: Do NOT repeat the same action (click/type/js) on the same element more than twice. If an approach fails twice, try something completely different or call done()."""

# ─── CDP ─────────────────────────────────────────────────────────────────────

class CDP:
    def __init__(self):
        self.ws = None; self.mid = 0

    async def connect(self):
        with urllib.request.urlopen(f"{CDP_URL}/json", timeout=5) as r:
            pages = json.loads(r.read())
        ws_url = next((p.get("webSocketDebuggerUrl") for p in pages if p.get("type")=="page" and "devtools" not in p.get("url","") and p.get("webSocketDebuggerUrl")), None)
        if not ws_url and pages:
            ws_url = next((p.get("webSocketDebuggerUrl") for p in pages if p.get("webSocketDebuggerUrl")), None)
        if not ws_url: print(f"{R}No browser pages{RS}"); sys.exit(1)
        self.ws = await websockets.connect(ws_url, max_size=50*1024*1024)
        for m in ["DOM.enable","Accessibility.enable","Page.enable","Runtime.enable"]: await self.cmd(m)

    async def reconnect(self):
        """Reconnect to the current active page after navigation."""
        try:
            if self.ws: await self.ws.close()
        except Exception: pass
        self.ws = None
        await asyncio.sleep(1)
        try:
            with urllib.request.urlopen(f"{CDP_URL}/json", timeout=5) as r:
                pages = json.loads(r.read())
        except Exception as e:
            print(f"{R}Cannot reach browser: {e}{RS}")
            return
        ws_url = next((p.get("webSocketDebuggerUrl") for p in pages if p.get("type")=="page" and "devtools" not in p.get("url","") and p.get("webSocketDebuggerUrl")), None)
        if not ws_url and pages:
            ws_url = next((p.get("webSocketDebuggerUrl") for p in pages if p.get("webSocketDebuggerUrl")), None)
        if not ws_url:
            print(f"{R}No browser pages to reconnect{RS}")
            return
        self.ws = await websockets.connect(ws_url, max_size=50*1024*1024)
        self.mid = 0
        for m in ["DOM.enable","Accessibility.enable","Page.enable","Runtime.enable"]: await self.cmd(m)

    async def cmd(self, method, params=None):
        self.mid += 1
        msg = {"id":self.mid,"method":method}
        if params: msg["params"] = params
        try:
            if self.ws is None:
                raise RuntimeError("WebSocket not connected")
            await self.ws.send(json.dumps(msg))
            while True:
                r = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=30))
                if r.get("id") == self.mid:
                    return r.get("result", r.get("error", {}))
        except Exception:
            # Reconnect on broken connection (page navigated away)
            try:
                await self.reconnect()
            except Exception as recon_err:
                print(f"{R}Reconnect failed: {recon_err}{RS}")
                self.ws = None
            return {"error": "Connection lost, reconnected. Try again."}

    async def navigate(self, url):
        await self.cmd("Page.navigate", {"url": url}); await asyncio.sleep(3)
        return f"Navigated to {url}"

    async def snapshot(self):
        tree = await self.cmd("Accessibility.getFullAXTree", {"depth": 8})
        nodes = tree.get("nodes", [])
        lines = []
        # Prioritize actionable elements: links, buttons, inputs, headings
        priority_roles = {"link","button","textbox","searchbox","heading","combobox","menuitem","checkbox","radio"}
        for n in nodes:
            # Handle both dict-typed AXValue and plain-string role/name from different Chrome versions
            raw_role = n.get("role", {})
            if isinstance(raw_role, dict):
                role = raw_role.get("value", "")
            else:
                role = str(raw_role)
            raw_name = n.get("name", {})
            if isinstance(raw_name, dict):
                name = raw_name.get("value", "")
            else:
                name = str(raw_name)
            nid = n.get("nodeId", "")
            if not nid: continue
            if not name or len(name) < 3: continue
            if role not in priority_roles and role != "StaticText": continue
            # Skip StaticText unless it's substantial
            if role == "StaticText" and len(name) < 30: continue
            p = [f"[{nid}]", role, f'"{name[:120]}"']
            lines.append(" ".join(p))
            if len(lines) >= 200: break
        snapshot_text = "\n".join(lines) if lines else "(Empty page)"
        # Include current URL and title so the model knows where it is
        url = await self.js("document.URL")
        title = await self.js("document.title")
        # Also grab top page links via JS so the model can see search results / headlines
        top_links = await self.js("""Array.from(document.querySelectorAll('a')).filter(a=>a.innerText.trim().length>10).map(a=>a.innerText.trim().substring(0,60)).slice(0,15).join(' | ')""")
        if top_links and not top_links.startswith("Error"):
            snapshot_text += f"\n\nTop links: {top_links[:800]}"
        return f"URL: {url}\nTitle: {title}\n\n{snapshot_text}"

    async def click(self, uid):
        if not uid or not str(uid).strip():
            return "Error: uid is empty"
        try:
            backend_node_id = int(uid)
        except (ValueError, TypeError):
            return f"Error: invalid uid '{uid}'"
        r = await self.cmd("DOM.resolveNode", {"backendNodeId": backend_node_id})
        if "error" in r: return f"Error: {r['error']}"
        oid = r.get("object",{}).get("objectId")
        if not oid: return "Error: can't resolve"
        await self.cmd("Runtime.callFunctionOn",{"objectId":oid,"functionDeclaration":"function(){this.scrollIntoView({block:'center'})}"})
        await asyncio.sleep(0.2)
        box = await self.cmd("DOM.getBoxModel",{"objectId":oid})
        if "error" in box or "model" not in box:
            await self.cmd("Runtime.callFunctionOn",{"objectId":oid,"functionDeclaration":"function(){this.click()}"})
            return "Clicked(JS)"
        c = box["model"]["content"]; x=(c[0]+c[4])/2; y=(c[1]+c[5])/2
        await self.cmd("Input.dispatchMouseEvent",{"type":"mousePressed","x":x,"y":y,"button":"left","clickCount":1})
        await self.cmd("Input.dispatchMouseEvent",{"type":"mouseReleased","x":x,"y":y,"button":"left","clickCount":1})
        return "Clicked"

    async def type_into(self, uid, text):
        click_result = await self.click(uid)
        if click_result.startswith("Error:"):
            return click_result
        await asyncio.sleep(0.3)
        for ch in text:
            await self.cmd("Input.dispatchKeyEvent",{"type":"keyDown","text":ch,"key":ch})
            await self.cmd("Input.dispatchKeyEvent",{"type":"keyUp","key":ch})
        return f"Typed {len(text)} chars"

    async def scroll(self, d="down"):
        delta = -500 if d=="up" else 500
        await self.cmd("Input.dispatchMouseEvent",{"type":"mouseWheel","x":400,"y":400,"deltaX":0,"deltaY":delta})
        await asyncio.sleep(0.5); return f"Scrolled {d}"

    async def js(self, code):
        safe = code.strip()
        # Wrap in IIFE so the model can freely use const/let/return without
        # "Illegal return statement" or redeclaration errors in global scope.
        if 'return' in safe or 'const ' in safe or 'let ' in safe:
            safe = f"(function(){{ {safe} }})()"
        r = await self.cmd("Runtime.evaluate",{"expression":safe,"returnByValue":True,"awaitPromise":True})
        if "error" in r: return f"Error: {r['error']}"
        return str(r.get("result",{}).get("value", r.get("result",{}).get("description","")))[:2000]

    async def post_comment(self, text):
        """Auto-handle commenting on any page.
        Uses DOM.pierce + DOM.focus + Input.insertText — works through
        cross-origin iframes, Shadow DOM, and ProseMirror editors.
        """
        # Step 1: Click Comments button
        print(f"  {D}→ Clicking Comments button...{RS}")
        await self.cmd("Runtime.evaluate",{"expression":"""
            const btn=Array.from(document.querySelectorAll('button')).find(b=>/comment/i.test(b.textContent));
            if(btn){btn.scrollIntoView({block:'center'});btn.click()}
        """})
        await asyncio.sleep(3)

        # Step 2: Wait for widget to load (don't scroll — it breaks Yahoo's infinite scroll)
        print(f"  {D}→ Loading comment widget...{RS}")
        await asyncio.sleep(5)

        # Step 3: Connect to OpenWeb iframe target and use DOM.pierce there
        # Save current URL so we can scroll back
        article_url = await self.js("document.URL")

        print(f"  {D}→ Searching for comment iframe...{RS}")
        for attempt in range(8):
            with urllib.request.urlopen(f"{CDP_URL}/json",timeout=5) as r:
                targets = json.loads(r.read())
            ow = [t for t in targets if t.get("type")=="iframe"
                  and any(k in t.get("url","") for k in ["openweb","spot.im","disqus","comment"])
                  and t.get("webSocketDebuggerUrl")]
            if ow: break
            # Small scroll only — don't trigger infinite scroll
            await self.cmd("Runtime.evaluate",{"expression":"window.scrollBy(0,150)"})
            await asyncio.sleep(2)
        else:
            # No comment iframe found
            pass

        if ow:
            print(f"  {D}→ Found iframe, connecting...{RS}")
            iws = await websockets.connect(ow[0]["webSocketDebuggerUrl"], max_size=50*1024*1024)
            imid = [0]
            async def isend(m,p=None):
                imid[0]+=1; msg={"id":imid[0],"method":m}
                if p: msg["params"]=p
                await iws.send(json.dumps(msg))
                while True:
                    r=json.loads(await asyncio.wait_for(iws.recv(),timeout=15))
                    if r.get("id")==imid[0]: return r.get("result",r.get("error",{}))

            for m in ["DOM.enable","Runtime.enable","Input.enable"]: await isend(m)
            await isend("DOM.getDocument",{"depth":-1,"pierce":True})

            # Search inside the iframe (pierces Shadow DOM)
            for attempt in range(5):
                try:
                    await isend("DOM.getDocument",{"depth":-1,"pierce":True})
                    r = await isend("DOM.performSearch",{"query":".ProseMirror","includeUserAgentShadowDOM":True})
                    count = r.get("resultCount",0)
                    sid = r.get("searchId","")
                    if count > 0:
                        results = await isend("DOM.getSearchResults",{"searchId":sid,"fromIndex":0,"toIndex":count})
                        node_ids = results.get("nodeIds",[])
                        if not node_ids:
                            if sid: await isend("DOM.discardSearchResults",{"searchId":sid})
                            continue
                        nid = node_ids[0]
                        fr = await isend("DOM.focus",{"nodeId":nid})
                        if "error" not in fr:
                            # Critical: wait for editor to be ready
                            await asyncio.sleep(1)
                            print(f"  {D}→ Typing comment ({len(text)} chars)...{RS}")
                            await isend("Input.insertText",{"text":text})
                            await asyncio.sleep(0.5)
                            if sid: await isend("DOM.discardSearchResults",{"searchId":sid})
                            await iws.close()
                            # Scroll comment area into view on main page
                            print(f"  {D}→ Scrolling to comment...{RS}")
                            await self.cmd("Runtime.evaluate",{"expression":"""
                                const iframes=document.querySelectorAll('iframe');
                                for(const f of iframes){if(f.src&&f.src.includes('openweb')){f.scrollIntoView({block:'center',behavior:'instant'});break}}
                            """})
                            await asyncio.sleep(0.3)
                            # Scroll up a bit so the comment input is visible, not just the iframe top
                            await self.cmd("Runtime.evaluate",{"expression":"window.scrollBy(0,-150)"})
                            return f"{G}Comment drafted! ({len(text)} chars) — NOT posted, ready for review.{RS}"
                    if sid: await isend("DOM.discardSearchResults",{"searchId":sid})
                except Exception as e:
                    print(f"  {R}→ iframe search error: {e}{RS}")
                # Wait for SpotIM to render
                print(f"  {D}→ Waiting for editor (attempt {attempt+1})...{RS}")
                await asyncio.sleep(3)

            await iws.close()

        # Fallback: simple main-page textarea
        escaped = text.replace("\\","\\\\").replace("'","\\'").replace("\n","\\n")
        r = await self.cmd("Runtime.evaluate",{"expression":f"""
            const el=document.querySelector('textarea,input[type=text],[contenteditable=true]');
            el?(el.focus(),el.value?el.value='{escaped}':el.innerText='{escaped}','found'):'none'
        ""","returnByValue":True})
        if r.get("result",{}).get("value")=="found":
            return f"{G}Comment drafted! ({len(text)} chars){RS}"

        return f"{Y}No comment input found. Comments may not be available on this page.{RS}"

    async def close(self):
        if self.ws: await self.ws.close()


# ─── MLX ─────────────────────────────────────────────────────────────────────

def ask_model(messages):
    body = json.dumps({"model":MODEL,"max_tokens":1024,"temperature":0.3,"messages":[{"role":"system","content":SYSTEM}]+messages}).encode()
    req = urllib.request.Request(f"{MLX_URL}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r: result=json.loads(r.read())
    return result.get("choices",[{}])[0].get("message",{}).get("content","")

def generate_comment(article_text):
    """Generate a clean comment from article text. Handles Qwen's verbose reasoning."""
    body = json.dumps({
        "model": MODEL, "max_tokens": 1024, "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "Comment on the news article. 2-3 sentences."},
            {"role": "user", "content": article_text}
        ]
    }).encode()
    req = urllib.request.Request(f"{MLX_URL}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read())
    raw = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    text = re.sub(r'\*+', '', text)  # Remove markdown

    # Qwen ALWAYS dumps reasoning. Extract only real comment sentences.
    all_sentences = re.findall(r'([A-Z][^.!?]{20,}[.!?])', text)

    # Filter out meta-reasoning — NOT part of a real comment
    meta = ['draft','constraint','sentence','critique','user','task','goal',
            'checking','format','plain text','let me','let\'s','count',
            'analyze','request','input','output','concise','polish',
            'revised','alternative','stick to','meets','criteria',
            'thinking','process','step','final','make sure']
    real = [s.strip() for s in all_sentences
            if not any(w in s.lower() for w in meta) and len(s) > 30]

    if real:
        return ' '.join(real[-3:])

    return "This situation raises serious concerns that demand greater transparency."


def parse(text):
    text = re.sub(r'<think>.*?</think>','',text,flags=re.DOTALL).strip()
    start = text.find('{"tool"')
    if start<0: start=text.find('{ "tool"')
    if start>=0:
        d=0
        for i in range(start,len(text)):
            if text[i]=='{': d+=1
            elif text[i]=='}':
                d-=1
                if d==0:
                    try: return json.loads(text[start:i+1])
                    except: break
    for m in re.finditer(r'\{[^{}]+\}',text):
        try:
            o=json.loads(m.group(0))
            if "tool" in o: return o
        except: continue
    return None


# ─── Agent ───────────────────────────────────────────────────────────────────

async def run(task):
    cdp = CDP(); await cdp.connect()
    print(f"{G}Connected to Brave{RS}\n")

    # Detect if this is a comment task
    is_comment = any(w in task.lower() for w in ["comment","draft","reply"])
    comment_text = None
    if is_comment:
        for marker in ["draft:","comment:","text:"]:
            idx = task.lower().rfind(marker)
            if idx >= 0:
                comment_text = task[idx+len(marker):].strip().rstrip(".")
                if len(comment_text) > 20: break
                comment_text = None

    # Extract topic keywords for smart navigation
    topic_words = []
    for word in ["iran","trump","war","ukraine","china","russia","gaza","israel","economy","oil"]:
        if word in task.lower(): topic_words.append(word)

    # FAST PATH: If this is a "go to site + find article + comment" task, skip the model for navigation
    if is_comment and topic_words:
        topic = " ".join(topic_words)
        # Detect which site
        site_url = "https://news.yahoo.com"
        for site in ["yahoo","reddit","cnn","bbc","nytimes"]:
            if site in task.lower():
                if site == "yahoo": site_url = "https://news.yahoo.com"
                elif site == "reddit": site_url = "https://www.reddit.com"
                break

        print(f"  {D}Step 1{RS} {B}navigate{RS}({site_url})")
        await cdp.navigate(site_url)

        print(f"  {D}Step 2{RS} {B}find article{RS}(topic='{topic}')")
        r = await cdp.js(f"""
            const links = Array.from(document.querySelectorAll('a'));
            const article = links.find(a => {{
                const text = a.textContent.toLowerCase();
                const href = a.href || '';
                return text.length > 30 && (href.includes('article') || href.includes('/news/'))
                    && {' && '.join(f'text.includes("{w}")' for w in topic_words)};
            }});
            if(article) {{ article.click(); article.textContent.trim().substring(0,100) }}
            else {{ 'NOT_FOUND' }}
        """)

        if r and r != "NOT_FOUND":
            print(f"         {D}→ {r[:80]}{RS}")
            await asyncio.sleep(3)

            # Generate comment if needed
            if not comment_text:
                print(f"  {D}Step 3{RS} {B}generate comment{RS}")
                article_text = await cdp.js("document.title + '. ' + Array.from(document.querySelectorAll('p')).map(p=>p.innerText).filter(t=>t.length>40).slice(0,6).join(' ')")
                comment_text = generate_comment(article_text[:600])
                print(f"         {D}→ {comment_text[:80]}...{RS}")

            # Post comment
            print(f"  {D}Step 4{RS} {B}post comment{RS}")
            result = await cdp.post_comment(comment_text)
            print(f"  {result}")
            print(f"\n{G}{BD}Done!{RS}")
            await cdp.close()
            return
        else:
            print(f"         {D}→ No article found with topic '{topic}', falling back to model{RS}")

    messages = [{"role":"user","content":f"Task: {task}\n\nINSTRUCTIONS:\n- Start by navigating to the relevant site.\n- Use snapshot() to see the page.\n- Use click(), type_text(), scroll(), js() to interact.\n- Call done(message) when the task is complete. Summarize what you found in the message."}]
    click_counts = {}  # Track how many times each UID is clicked
    last_snapshot = ""  # Track last snapshot to detect stuck state
    last_url = ""  # Track last URL to detect stuck state
    action_history = []  # Track recent actions for loop detection

    for step in range(1, MAX_STEPS+1):
        t0=time.time(); resp=ask_model(messages); elapsed=time.time()-t0
        tc = parse(resp)
        if not tc:
            print(f"  {D}Step {step} (no tool) {elapsed:.1f}s{RS}")
            messages.append({"role":"assistant","content":resp})
            messages.append({"role":"user","content":'Respond with ONLY: {"tool":"name","args":{...}}'})
            continue

        tool=tc.get("tool",""); args=tc.get("args",{})
        action_key = json.dumps({"tool": tool, "args": args}, sort_keys=True)
        action_history.append(action_key)
        if len(action_history) > 8: action_history.pop(0)

        # ─── Loop Detection ─────────────────────────────────────────
        loop_detected = False
        if tool == "click":
            uid = str(args.get("uid", ""))
            click_counts[uid] = click_counts.get(uid, 0) + 1
            if click_counts[uid] > 2:
                loop_detected = True
                print(f"  {Y}Step {step} LOOP DETECTED: uid {uid} clicked {click_counts[uid]} times — forcing Escape + snapshot{RS}")
        # Detect A-B-A-B action pattern
        if len(action_history) >= 4 and not loop_detected:
            if action_history[-1] == action_history[-3] and action_history[-2] == action_history[-4]:
                loop_detected = True
                print(f"  {Y}Step {step} LOOP DETECTED: repeating action pattern — forcing js recovery{RS}")

        if loop_detected:
            # Auto-recover: press Escape (closes lightboxes/overlays), then force a snapshot
            await cdp.js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))")
            await asyncio.sleep(0.5)
            r = await cdp.snapshot()
            messages.append({"role":"assistant","content":json.dumps({"tool":"js","args":{"code":"Escape"}})})
            messages.append({"role":"user","content":f"Result: LOOP DETECTED — you are repeating actions but the page is NOT changing. I pressed Escape to close any overlay. Here is a fresh snapshot — use js(code) or navigate() to a DIFFERENT approach:\n\n{r[:3000]}"})
            continue

        args_s=', '.join(f'{k}={repr(v)[:40]}' for k,v in args.items())
        print(f"  {D}Step {step}{RS} {B}{tool}{RS}({args_s}) {D}{elapsed:.1f}s{RS}")

        # Get URL before action (only meaningful for click/type_text)
        url_before = await cdp.js("document.URL") if tool in ("click","type_text") else ""

        if tool=="navigate": r=await cdp.navigate(args.get("url",""))
        elif tool=="snapshot":
            r=await cdp.snapshot()
            # Detect stuck state: snapshot looks the same as last time
            if last_snapshot and r == last_snapshot:
                r = r + "\n\n⚠️ WARNING: This snapshot is IDENTICAL to the previous one. The page has NOT changed. Try a different approach — scroll, press Escape, or navigate to a different URL."
            last_snapshot = r
        elif tool=="click": r=await cdp.click(str(args.get("uid","")))
        elif tool=="type_text": r=await cdp.type_into(str(args.get("uid","")),args.get("text",""))
        elif tool=="scroll": r=await cdp.scroll(args.get("direction","down"))
        elif tool=="comment": r=await cdp.post_comment(args.get("text",""))
        elif tool=="js": r=await cdp.js(args.get("code",""))
        elif tool=="done":
            # If this is a comment task, auto-comment before finishing
            if is_comment:
                # If no comment text provided, generate one from article content
                if not comment_text:
                    print(f"\n  {BD}Generating comment from article...{RS}")
                    article_text = await cdp.js("var el=document.querySelector('article, main, [role=main]');(el&&el.innerText?el.innerText.substring(0,500):document.title)")
                    comment_text = generate_comment(article_text)
                    print(f"  {D}Generated: {comment_text[:80]}...{RS}")

                print(f"\n  {BD}Auto-commenting on article...{RS}")
                result = await cdp.post_comment(comment_text)
                print(f"  {result}")
            print(f"\n{G}{BD}Done:{RS} {args.get('message','')}")
            await cdp.close(); return
        else: r=f"Unknown: {tool}"

        # Detect stuck page: URL should change after click/type_text. Scroll/js don't navigate.
        if tool in ("click","type_text"):
            url_after = await cdp.js("document.URL")
            if url_after == url_before:
                r = f"{r}\n\n⚠️ NOTE: URL did NOT change after {tool}(). The page is likely stuck. Try js(code) or navigate() instead."
            last_url = url_after
        elif tool == "navigate":
            last_url = ""
            last_snapshot = ""

        if len(r)>4000: r=r[:4000]+"...(truncated)"
        messages.append({"role":"assistant","content":json.dumps(tc)})
        messages.append({"role":"user","content":f"Result: {r}"})
        if len(messages)>10: messages=messages[:1]+messages[-8:]
        print(f"         {D}→ {r[:160].replace(chr(10),' ')}{RS}")

    # If we hit max steps on a comment task, try commenting on whatever page we're on
    if is_comment:
        if not comment_text:
            article_text = await cdp.js("document.title + '. ' + Array.from(document.querySelectorAll('p')).map(p=>p.innerText).filter(t=>t.length>40).slice(0,6).join(' ')")
            comment_text = generate_comment(article_text)
        print(f"\n  {BD}Auto-commenting on current page...{RS}")
        result = await cdp.post_comment(comment_text)
        print(f"  {result}")

    await cdp.close()

def main():
    print(f"\n{BD}  → Local Browser Agent{RS}")
    print(f"  {D}MLX + CDP · iframes + Shadow DOM · no cloud{RS}\n")

    # If args passed, run once
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        print()
        try:
            asyncio.run(run(task))
        except Exception as e:
            print(f"\n{R}Task failed: {type(e).__name__}: {e}{RS}")
        return

    # Interactive loop — keep running tasks. Catch ANY exception from a single
    # task (MLX timeout, CDP websocket drop, bad model output, etc.) and loop
    # back to the prompt instead of exiting — one bad task shouldn't kill the
    # whole session.
    while True:
        try:
            task = input(f"\n{BD}What should I do?{RS} ")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{D}Bye!{RS}")
            break
        if not task.strip(): continue
        if task.strip().lower() in ("quit","exit","q"): break
        print()
        try:
            asyncio.run(run(task))
        except KeyboardInterrupt:
            print(f"\n{Y}Task interrupted — back to prompt{RS}")
        except Exception as e:
            print(f"\n{R}Task failed: {type(e).__name__}: {e}{RS}")
            print(f"{D}Back to prompt — try again or type 'quit' to exit{RS}")

if __name__=="__main__": main()
