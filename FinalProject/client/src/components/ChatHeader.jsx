// Simple modern chat header
export default function ChatHeader({ endpoint }) {
  return (
    <header className="topbar">
      <div className="topbarTitle">
        <h2>Chat</h2>
        <p className="topbarSub">POST {endpoint} · {'{ prompt }'}</p>
      </div>
    </header>
  );
}
