import { useEffect, useRef, useState } from 'react';
import { ArrowUp, BookOpen, Check, ChevronRight, Copy, Menu, MessageSquare, Plus, Settings2, Sparkles, Square, Trash2, X } from 'lucide-react';
import Markdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { normalizeUrl, solveMath } from './api';

const STORAGE = 'math-studio-chats-v1';
const examples = [
  { icon: '𝑥', title: 'Giải phương trình', text: 'Solve for x: 3*x - 7 = 20.', tag: 'Đại số' },
  { icon: '△', title: 'Khám phá hình học', text: 'A rectangle has area 96 square centimeters. Its length is 4 centimeters greater than its width. Find its width and length.', tag: 'Hình học' },
  { icon: '½', title: 'Luyện tập phân số', text: "A tank is 3/5 full. After adding 18 liters, it is 9/10 full. What is the tank’s total capacity?", tag: 'Số học' },
  { icon: '%', title: 'Thử sức xác suất', text: 'A bag contains 5 red and 4 blue balls. Three balls are drawn without replacement. What is the probability that exactly two are red?', tag: 'Xác suất' },
];
function readChats() {
  try { const data = JSON.parse(localStorage.getItem(STORAGE) || '[]'); return Array.isArray(data) ? data.filter(c => typeof c.id === 'string' && typeof c.title === 'string' && Array.isArray(c.messages)).slice(0, 30) : []; } catch { return []; }
}
function readUrl() { try { return localStorage.getItem('math-studio-api') || import.meta.env.VITE_API_URL || ''; } catch { return import.meta.env.VITE_API_URL || ''; } }
function Answer({ text }) {
  const [copied, setCopied] = useState(false);
  return <><div className="markdown"><Markdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{text}</Markdown></div><button className="copy" onClick={async () => { try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1800); } catch { setCopied(false); } }} aria-label="Sao chép đáp án">{copied ? <Check size={15}/> : <Copy size={15}/>} {copied ? 'Đã sao chép' : 'Sao chép'}</button></>;
}
export default function App() {
  const [chats, setChats] = useState(readChats);
  const [active, setActive] = useState(null);
  const [question, setQuestion] = useState('');
  const [method, setMethod] = useState('qlora');
  const [apiUrl, setApiUrl] = useState(readUrl);
  const [draftUrl, setDraftUrl] = useState(apiUrl);
  const [settings, setSettings] = useState(!apiUrl);
  const [menu, setMenu] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [storageError, setStorageError] = useState('');
  const controller = useRef(null);
  const end = useRef(null);
  const input = useRef(null);
  const chat = chats.find(c => c.id === active);
  const messages = chat?.messages || [];
  useEffect(() => { try { localStorage.setItem(STORAGE, JSON.stringify(chats)); } catch { setStorageError('Không lưu được lịch sử trên trình duyệt này.'); } }, [chats]);
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages.length, busy]);
  useEffect(() => () => controller.current?.abort(), []);
  function newChat() { if (busy) return; setActive(null); setQuestion(''); setError(''); setMenu(false); input.current?.focus(); }
  async function send(event) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    if (text.length > 8192) { setError('Câu hỏi tối đa 8192 ký tự.'); return; }
    try { normalizeUrl(apiUrl); } catch { setSettings(true); setError('Hãy cấu hình URL gateway trước khi gửi.'); return; }
    const id = active || crypto.randomUUID();
    const selectedMethod = method;
    const message = { id: crypto.randomUUID(), role: 'user', text, method: selectedMethod };
    setChats(prev => active ? prev.map(c => c.id === id ? { ...c, messages: [...c.messages, message] } : c) : [{ id, title: text.slice(0, 50), messages: [message] }, ...prev].slice(0, 30));
    setActive(id); setQuestion(''); setError(''); setBusy(true);
    const abort = new AbortController(); controller.current = abort;
    try {
      const data = await solveMath({ url: apiUrl, method: selectedMethod, question: text, signal: abort.signal });
      setChats(prev => prev.map(c => c.id === id ? { ...c, messages: [...c.messages, { id: crypto.randomUUID(), role: 'assistant', text: data.answer, method: data.method }] } : c));
    } catch (err) { setError(err.message); setQuestion(text); }
    finally { setBusy(false); controller.current = null; }
  }
  function saveUrl(event) {
    event.preventDefault();
    try { const url = normalizeUrl(draftUrl); setApiUrl(url); try { localStorage.setItem('math-studio-api', url); } catch {} setSettings(false); setError(''); } catch (err) { setError(err.message); }
  }
  return <div className="app-shell">
    {menu && <button className="overlay" aria-label="Đóng menu" onClick={() => setMenu(false)}/>}
    <aside className={menu ? 'sidebar open' : 'sidebar'}>
      <a className="brand" href="#" onClick={e => { e.preventDefault(); newChat(); }}><span className="brand-icon"><BookOpen size={23}/></span><span>math<span className="brand-light">studio</span><small>KHƠI MỞ TƯ DUY</small></span></a>
      <button className="new-chat" onClick={newChat} disabled={busy}><Plus size={18}/> Cuộc trò chuyện mới</button>
      <div className="history-label">HỘI THOẠI CỦA BẠN</div>
      <nav className="history">{chats.length ? chats.map(c => <div className={`history-item ${active === c.id ? 'selected' : ''}`} key={c.id}><button disabled={busy} onClick={() => { setActive(c.id); setMenu(false); setError(''); }}><MessageSquare size={16}/><span>{c.title}</span></button><button className="delete" aria-label="Xóa hội thoại" disabled={busy} onClick={() => { setChats(prev => prev.filter(x => x.id !== c.id)); if (active === c.id) setActive(null); }}><Trash2 size={14}/></button></div>) : <p className="history-empty">Những câu hỏi hay bắt đầu<br/>từ một chút tò mò.</p>}</nav>
      <div className="sidebar-note"><Sparkles size={19}/><strong>Hiểu cách giải.<br/>Không chỉ đáp án.</strong><p>Cùng khám phá toán học,<br/>từng bước một.</p></div>
      <button className="settings-button" disabled={busy} onClick={() => { setDraftUrl(apiUrl); setSettings(true); setMenu(false); }}><Settings2 size={17}/> Cấu hình API <ChevronRight size={15}/></button>
      <div className="profile"><span>N</span><div>Không gian học tập<small>Lịch sử lưu trên trình duyệt</small></div></div>
    </aside>
    <main>
      <header><div className="header-left"><button className="mobile-menu icon-button" aria-label="Mở menu" onClick={() => setMenu(true)}><Menu size={22}/></button><span>Trợ lý học toán <span className="beta">BETA</span></span></div><div className="header-right"><span className="api-state"><i className={apiUrl ? 'configured' : ''}/>{apiUrl ? 'Đã cấu hình API' : 'Chưa cấu hình API'}</span><label className="model-select"><span>Model</span><select aria-label="Chọn model" value={method} disabled={busy} onChange={e => setMethod(e.target.value)}><option value="qlora">QLoRA</option><option value="lora">LoRA</option></select></label></div></header>
      <div className="scroll-area">
        {!messages.length ? <section className="welcome"><span className="welcome-mark">∑</span><div className="eyebrow">MỖI CÂU HỎI, MỘT KHÁM PHÁ</div><h1>Toán học dễ hiểu hơn,<br/><span>khi có người đồng hành.</span></h1><p>Đặt câu hỏi, khám phá lời giải và luyện tập theo cách của bạn.</p><div className="examples">{examples.map(ex => <button key={ex.title} onClick={() => { setQuestion(ex.text); input.current?.focus(); }}><span className="example-icon">{ex.icon}</span><span className="example-tag">{ex.tag}</span><strong>{ex.title}</strong><span className="example-bottom">Bắt đầu với bài toán mẫu <ChevronRight size={15}/></span></button>)}</div><div className="language-note">Model được huấn luyện với bài toán tiếng Anh.<br/>Bạn nên đặt câu hỏi bằng tiếng Anh để thử trước.</div></section> : <section className="messages" aria-label="Hội thoại">{messages.map(msg => <article key={msg.id} className={`message ${msg.role}`}><div className="avatar">{msg.role === 'user' ? 'B' : <Sparkles size={18}/>}</div><div className="message-content"><div className="message-label">{msg.role === 'user' ? 'Bạn' : 'Math Studio'}{msg.role === 'assistant' && <span>{msg.method.toUpperCase()}</span>}</div>{msg.role === 'user' ? <p className="user-text">{msg.text}</p> : <Answer text={msg.text}/>}</div></article>)}{busy && <article className="message assistant" role="status"><div className="avatar"><Sparkles size={18}/></div><div className="thinking"><span/><span/><span/> Đang tìm lời giải…</div></article>}<div ref={end}/></section>}
      </div>
      <footer className="composer-area">{(error || storageError) && <div className="error" role="alert">{error || storageError}</div>}<form className="composer" onSubmit={send}><textarea ref={input} value={question} maxLength={8192} disabled={busy} onChange={e => setQuestion(e.target.value)} placeholder="Bạn muốn cùng giải bài toán nào?" aria-label="Nhập bài toán" rows={2} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (!busy) e.currentTarget.form.requestSubmit(); } }}/><div className="composer-bottom"><span><Sparkles size={14}/> Từng bước, cùng bạn hiểu toán</span>{busy ? <button className="send" type="button" aria-label="Dừng chờ đáp án" onClick={() => controller.current?.abort()}><Square size={17}/></button> : <button className="send" type="submit" disabled={!question.trim()} aria-label="Gửi câu hỏi"><ArrowUp size={21}/></button>}</div></form><p className="disclaimer">AI có thể trả lời sai. Hãy kiểm tra lại lời giải. Mỗi câu hỏi được xử lý độc lập, không gửi lịch sử cho model.</p></footer>
    </main>
    {settings && <div className="modal-backdrop"><section className="modal" role="dialog" aria-modal="true" aria-labelledby="settings-title"><button className="modal-close icon-button" aria-label="Đóng cấu hình" onClick={() => setSettings(false)}><X size={20}/></button><span className="modal-icon"><Settings2 size={23}/></span><h2 id="settings-title">Kết nối không gian học tập</h2><p>Nhập URL ngrok đang trỏ tới gateway trên Kaggle (port 8080).</p><form onSubmit={saveUrl}><label htmlFor="api-url">URL API</label><input autoFocus id="api-url" type="url" required placeholder="https://ten-mien.ngrok-free.dev" value={draftUrl} onChange={e => setDraftUrl(e.target.value)}/>{error && <div className="error" role="alert">{error}</div>}<small>Chỉ nhập URL gốc, không thêm /docs hoặc /solve. Phiên Kaggle và tunnel cần đang chạy.</small><button className="primary" type="submit">Lưu cấu hình <ChevronRight size={17}/></button></form></section></div>}
  </div>;
}
