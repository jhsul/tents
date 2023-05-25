import puppeteer, { Browser, Page } from "puppeteer";
import { Tensor } from "../src/tensor";
import { arrEq } from "../src/util";

describe("webgpu tests", () => {
  let browser: Browser;
  let page: Page;

  beforeAll(async () => {
    browser = await puppeteer.launch({
      args: ["--headless"],

      executablePath:
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    });

    page = await browser.newPage();
    page.on("console", (msg) => {
      console.log("BROWSER LOG:", msg.text());
    });
    // Load your scripts or page that uses WebGPU here
    // e.g., await page.goto('http://localhost:8080');
  });

  afterAll(async () => {
    await browser.close();
  });

  it("should access the GPU device", async () => {
    const result = await page.evaluate(async () => {
      //@ts-expect-error
      const adapter = await navigator.gpu?.requestAdapter();
      const device = await adapter?.requestDevice();

      const t = Tensor.zeros(1);

      // Just confirm the device even exists
      return t;
    });

    expect(arrEq(result.data, [0])).toBeTruthy();
  });
});
