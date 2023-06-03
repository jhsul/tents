import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";

export type Benchmark = (n: number) => Promise<number>;

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
