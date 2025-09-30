const apiBase = () => {
  const apiBaseInput = document.getElementById("apiBase");
  return (apiBaseInput.value || "").replace(/\/+$/, "");
};

const handleResponse = async (res) => {
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
};

const postForm = async (path, form) => {
  const res = await fetch(apiBase() + path, { method: "POST", body: form });
  return handleResponse(res);
};

const postEmpty = async (path) => {
  const res = await fetch(apiBase() + path, { method: "POST" });
  return handleResponse(res);
};

const getJson = async (path) => {
  const res = await fetch(apiBase() + path);
  return handleResponse(res);
};

export { postForm, postEmpty, getJson };