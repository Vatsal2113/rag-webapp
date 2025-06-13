/* static/script.js
   Front-end helper for progress polling & chat.
   Loaded as an ES module from both progress.html and chat.html.
*/

/* ───────── progress page ───────── */
export async function pollStatus(jobId){
  const r = await fetch(`/status/${jobId}`).then(r => r.json());
  const msg = document.getElementById("msg");

  if (r.status === "done") {
    window.location = `/chat/${jobId}`;              // finished → chat UI
  } else if (r.status === "error") {
    msg.textContent = "❌ " + (r.error || "Ingestion failed.");
  } else {  // "pending" or undefined
    setTimeout(() => pollStatus(jobId), 1500);       // keep polling
  }
}

/* ───────── chat page ───────── */
export function setupChat(jobId){
  const box   = document.getElementById("chatbox");
  const input = document.getElementById("msg");
  const form  = document.getElementById("chatForm");

  form.onsubmit = async (e)=>{
    e.preventDefault();
    const q = input.value.trim();
    if(!q) return;
    box.insertAdjacentHTML("beforeend", `<div class="me">${q}</div>`);
    box.scrollTop = box.scrollHeight;
    input.value = "";

    const res = await fetch(`/api/chat/${jobId}`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({message: q})
    }).then(r => r.json());

    const html = res.ok ? res.answer
                        : (res.answer || "⚠️ error");
    box.insertAdjacentHTML("beforeend", `<div class="bot">${html}</div>`);
    box.scrollTop = box.scrollHeight;
  };
}
