import type { FunctionComponent } from "react";
import { useRef, useState, useEffect, useCallback } from "react";
import { helloTensor } from "./scripts/hello";

import "./styles.scss";
import { mean, stddev } from "../bin/util";

const scripts = [helloTensor];

const App: FunctionComponent = () => {
  const selectRef = useRef<HTMLSelectElement | null>(null);

  const [K, setK] = useState(10);
  const [C, setC] = useState(1);

  const [N, setN] = useState<number[]>([]);

  const runScript = useCallback(() => {
    if (!selectRef.current) return;

    for (const n of N) {
      const times: number[] = [];
      for (let i = 0; i < C; i++) {
        const start = Date.now();
        scripts[parseInt(selectRef.current.value)](n);
        const t = Date.now() - start;

        times.push(t);
      }

      const mu = mean(times);
      const sigma = stddev(times);

      console.log(mu, sigma);
    }
  }, [K, C, N]);

  useEffect(() => {
    const newN = new Array(K);

    for (let i = 0; i < K; i++) {
      newN[i] = Math.floor(Math.pow(2, i));
    }
    setN(newN);
  }, [K]);

  return (
    <div className="app">
      <select ref={selectRef}>
        {scripts.map((b, i) => (
          <option key={i} value={i}>
            {b.name}
          </option>
        ))}
      </select>

      <div className="hbox">
        <div>k: </div>
        <input
          type="range"
          min="1"
          max="20"
          step="1"
          onChange={(e) => setK(parseInt(e.currentTarget.value))}
          value={K}
        ></input>
        <div>{K}</div>
      </div>

      <div className="hbox">
        <div>c: </div>
        <input
          type="range"
          min="1"
          max="30"
          step="1"
          onChange={(e) => setC(parseInt(e.currentTarget.value))}
          value={C}
        ></input>
        <div>{C}</div>
      </div>

      <div>
        Take {C} samples each across inputs {JSON.stringify(N)}
      </div>

      <button onClick={runScript}>Start</button>
    </div>
  );
};

export default App;
