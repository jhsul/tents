import type { FunctionComponent } from "react";
import { useState, useEffect, useCallback } from "react";
import { ExportToCsv } from "export-to-csv";

import { tests } from "../bin/tests";

import { helloTensor } from "./benchmarks/hello";
import { vecaddForloopCpu, vecaddWebGpu } from "./benchmarks/addition";

import "./styles.scss";
import { mean, stddev } from "../bin/util";
import type { Benchmark } from "./main";
import { Tensor } from "../bin";

const benchmarks: Benchmark[] = [vecaddForloopCpu, vecaddWebGpu];

interface CSVDatum {
  n: number;
  time: number;
  stddev: number;
}

const App: FunctionComponent = () => {
  // const selectRef = useRef<HTMLSelectElement | null>(null);
  const [benchIdx, setBenchIdx] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [shouldSave, setShouldSave] = useState(false);

  const [K, setK] = useState(25);
  const [C, setC] = useState(10);

  // const [N, setN] = useState<number[]>([]);

  const runScript = useCallback(async () => {
    console.log("Beginning Benchmark!");

    await Tensor.setupDevice();
    setIsRunning(true);
    const csvExporter = new ExportToCsv({
      filename: `${benchmarks[benchIdx].name}-js`,
      fieldSeparator: ",",
      quoteStrings: '"',
      decimalSeparator: ".",
      showLabels: true,
      showTitle: false,

      useTextFile: false,
      useBom: true,
      useKeysAsHeaders: true,
      // headers: ['Column 1', 'Column 2', etc...] <-- Won't work with useKeysAsHeaders present!
    });

    const csvData: CSVDatum[] = [];

    for (let scale = 0; scale < K; scale++) {
      const n = 1 << scale;
      const times: number[] = [];
      for (let i = 0; i < C; i++) {
        const t = await benchmarks[benchIdx](n);

        times.push(t);
      }

      const mu = mean(times);
      const sigma = stddev(times);

      csvData.push({ n, time: mu, stddev: sigma });
    }
    if (shouldSave) csvExporter.generateCsv(csvData);
    else console.log(csvData);

    setIsRunning(false);
  }, [K, C, benchIdx, shouldSave]);

  const runTests = () => {
    for (const test of tests) {
      test();
    }
  };

  return (
    <div className="app">
      <div className={`form ${isRunning ? "running" : ""}`}>
        <b>TenTS Benchmark Tool</b>
        <br />
        <select
          value={benchIdx}
          onChange={(e) => setBenchIdx(parseInt(e.currentTarget.value))}
        >
          {benchmarks.map((b, i) => (
            <option key={i} value={i}>
              {b.name}
            </option>
          ))}
        </select>

        <br />

        <div>Max scale (1, 2, ..., 2^k)</div>

        <div className="hbox">
          <div>k: </div>
          <input
            type="range"
            min="1"
            max="30"
            step="1"
            onChange={(e) => setK(parseInt(e.currentTarget.value))}
            value={K}
            disabled={isRunning}
          ></input>
          <div>{K}</div>
        </div>

        <br />

        <div># of samples</div>

        <div className="hbox">
          <div>c: </div>
          <input
            type="range"
            min="1"
            max="50"
            step="1"
            onChange={(e) => setC(parseInt(e.currentTarget.value))}
            value={C}
            disabled={isRunning}
          ></input>
          <div>{C}</div>
        </div>

        <br />

        <div className="hbox">
          <input
            type="checkbox"
            value={shouldSave.toString()}
            onChange={() => setShouldSave((b) => !b)}
            disabled={isRunning}
          />
          <div>Save data to CSV</div>
        </div>

        <button onClick={runScript} disabled={isRunning}>
          Start Benchmark
        </button>

        <br />

        <button onClick={runTests} disabled={isRunning}>
          Run Unit Tests
        </button>
      </div>
    </div>
  );
};

export default App;
