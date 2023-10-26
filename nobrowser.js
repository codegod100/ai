import { chromium } from "playwright"
const token = process.env.BROWSERLESS_API_TOKEN
console.log(token)
const browser = await chromium.connect(`wss://chrome.browserless.io/playwright?token=${token}`)
console.log("connected")
const context = await browser.newContext()
console.log("context created")
const page = await context.newPage()
await page.goto("https://www.google.com")
await page.screenshot({ path: "nobrowser.png" })
await browser.close()