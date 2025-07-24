import { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);

  const send = async () => {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value);
      const parts = buffer.split("\n\n");
      parts.slice(0, -1).forEach((part) => {
        const dataLine = part.replace("data: ", "");
        const payload = JSON.parse(dataLine);
        setMessages((m) => [...m, payload]);
      });
      buffer = parts.at(-1) || "";
    }
  };

  return (
    <main className="p-4 max-w-3xl mx-auto font-sans">
      <h1 className="text-2xl mb-4">AI Shop Assistant</h1>
      <div className="flex gap-2 mb-6">
        <input
          className="flex-1 border rounded px-3 py-2"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask for product recommendations…"
        />
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded"
          onClick={send}
        >
          Send
        </button>
      </div>

      {messages.map((msg, idx) => (
        <section key={idx} className="mb-8">
          <p className="font-semibold mb-2">{msg.answer}</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {msg.results.map((p) => (
              <article
                key={p.id}
                className="border rounded p-4 shadow-sm flex gap-4"
              >
                {p.image && (
                  <img
                    src={p.image}
                    alt={p.title}
                    className="w-20 h-20 object-contain"
                  />
                )}
                <div>
                  <h3 className="font-medium">{p.title}</h3>
                  <p className="text-sm text-gray-600">${p.price}</p>
                </div>
              </article>
            ))}
          </div>
        </section>
      ))}
    </main>
  );
}
