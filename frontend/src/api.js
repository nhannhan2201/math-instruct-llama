export function normalizeUrl(value) {
  const url = new URL(value.trim());
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/') throw new Error('Nhập URL gốc của gateway, ví dụ https://ten-mien.ngrok-free.dev');
  return url.origin;
}
export async function solveMath({ url, method, question, signal }) {
  const base = normalizeUrl(url);
  if (!['lora', 'qlora'].includes(method)) throw new Error('Model không hợp lệ.');
  let response;
  try {
    response = await fetch(`${base}/${method}/solve`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
      body: JSON.stringify({ question, max_new_tokens: 512, temperature: 0.7 }),
      signal: AbortSignal.any([signal, AbortSignal.timeout(200000)]),
    });
  } catch (error) {
    if (signal.aborted) throw new Error('Đã dừng chờ đáp án.');
    if (error.name === 'TimeoutError') throw new Error('Model trả lời quá lâu. Hãy thử lại.');
    throw new Error('Không kết nối được API. Kiểm tra URL, session Kaggle, tunnel và cấu hình CORS.');
  }
  if (!response.ok) {
    const errors = { 422: 'Câu hỏi hoặc tham số không hợp lệ.', 503: 'Model chưa sẵn sàng trên Kaggle.', 502: 'Gateway không kết nối được server model.', 504: 'Model trả lời quá lâu.', 500: 'Model gặp lỗi khi sinh đáp án.' };
    throw new Error(errors[response.status] || `API trả lỗi HTTP ${response.status}.`);
  }
  let data;
  try { data = await response.json(); } catch { throw new Error('API không trả JSON. Kiểm tra tunnel ngrok.'); }
  if (data.success !== true || data.method !== method || typeof data.answer !== 'string' || !data.answer.trim()) throw new Error('Phản hồi API không đúng định dạng hoặc model đã chọn.');
  return data;
}
