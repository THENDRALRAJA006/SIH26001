import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App.jsx";
import { OfficerAuthProvider } from "./context/OfficerAuthContext.jsx";
import { CitizenSessionProvider } from "./context/CitizenSessionContext.jsx";
import { ThemeProvider } from "./context/ThemeContext.jsx";
import { LanguageProvider } from "./context/LanguageContext.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <LanguageProvider>
          <OfficerAuthProvider>
            <CitizenSessionProvider>
              <App />
            </CitizenSessionProvider>
          </OfficerAuthProvider>
        </LanguageProvider>
      </ThemeProvider>
    </BrowserRouter>
  </StrictMode>
);
