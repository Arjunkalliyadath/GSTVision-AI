import { BrowserRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom";
import Dashboard    from "./pages/Dashboard";
import ModelInfoPage from "./pages/ModelInfoPage";
import "./App.css";

function NavBar() {
  const location = useLocation();

  const links = [
    { path: "/",      label: "📄 Upload & Extract" },
    { path: "/model", label: "🧠 Model Info" },
  ];

  return (
    <nav style={{
      background: "linear-gradient(135deg, #1a237e 0%, #0d47a1 100%)",
      padding: "0 2rem",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      height: "62px",
      boxShadow: "0 2px 10px rgba(0,0,0,0.3)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <span style={{ fontSize: "26px" }}>🏛️</span>
        <div>
          <div style={{ color: "white", fontWeight: "700", fontSize: "16px", letterSpacing: "0.4px" }}>
            GST 2A Converter
          </div>
          <div style={{ color: "#90caf9", fontSize: "11px" }}>
            AI-Powered Statement Extractor
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: "6px" }}>
        {links.map(({ path, label }) => {
          const active = location.pathname === path;
          return (
            <Link
              key={path}
              to={path}
              style={{
                color:          active ? "white" : "#90caf9",
                textDecoration: "none",
                padding:        "8px 18px",
                borderRadius:   "8px",
                background:     active ? "rgba(255,255,255,0.18)" : "transparent",
                fontSize:       "14px",
                fontWeight:     active ? "700" : "400",
                transition:     "all 0.2s",
              }}
            >
              {label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <Router>
      <div style={{
        minHeight:   "100vh",
        background:  "#f0f4f8",
        fontFamily:  "'Segoe UI', -apple-system, sans-serif",
      }}>
        <NavBar />
        <Routes>
          <Route path="/"      element={<Dashboard />}     />
          <Route path="/model" element={<ModelInfoPage />} />
        </Routes>
      </div>
    </Router>
  );
}
