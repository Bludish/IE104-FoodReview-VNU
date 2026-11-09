/**
 * Lớp gọi API dùng chung.
 * Tự động gắn JWT, parse ApiResponse<T> và ném lỗi có message từ backend.
 */
const API = (() => {
  const base = CONFIG.API_BASE_URL.replace(/\/$/, '');

  function getToken() {
    return localStorage.getItem(CONFIG.TOKEN_KEY);
  }

  async function request(path, options = {}) {
    const { method = 'GET', body, auth = true } = options;
    const headers = { Accept: 'application/json' };

    const token = getToken();
    if (auth && token) headers.Authorization = `Bearer ${token}`;

    let payload;
    if (body !== undefined) {
      if (body instanceof FormData) {
        payload = body; // Trình duyệt tự set Content-Type + boundary
      } else {
        headers['Content-Type'] = 'application/json';
        payload = JSON.stringify(body);
      }
    }

    let response;
    try {
      response = await fetch(`${base}${path}`, { method, headers, body: payload });
    } catch (error) {
      throw new Error(`Không kết nối được tới API tại ${base}. Hãy chắc chắn backend đang chạy.`);
    }

    const text = await response.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch { data = text; }
    }

    if (!response.ok) {
      const validationErrors = data && typeof data === 'object' && data.errors
        ? Object.values(data.errors).flat().filter(Boolean).join(' ')
        : '';
      const message =
        validationErrors ||
        (data && typeof data === 'object' && (data.message || data.title)) ||
        (typeof data === 'string' && data) ||
        `Lỗi HTTP ${response.status}`;
      const error = new Error(message);
      error.status = response.status;
      error.data = data;
      throw error;
    }

    return data;
  }

  const get = (path, options = {}) => request(path, { ...options, method: 'GET' });
  const post = (path, body, options = {}) => request(path, { ...options, method: 'POST', body });
  const put = (path, body, options = {}) => request(path, { ...options, method: 'PUT', body });
  const del = (path, options = {}) => request(path, { ...options, method: 'DELETE' });

  async function upload(file, folder = 'images') {
    const formData = new FormData();
    formData.append('file', file);
    return request(`/api/Upload?folder=${encodeURIComponent(folder)}`, {
      method: 'POST',
      body: formData,
    });
  }

  async function ping() {
    return request('/api/Areas', { auth: false });
  }

  return { base, getToken, request, get, post, put, del, upload, ping };
})();
