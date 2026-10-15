# Topstep 150K rules: Topstep's own words, against what the engine models

Read first-hand on 15 October 2026 (raw HTML fetched from `help.topstep.com` and `www.topstep.com`, text extracted, no search summaries). Every quoted line below is copied from the page named in its heading; nothing in quotation marks is paraphrased. Lines without quotation marks are my reading, and are labelled as such.

Where a quote is a table row, the cells are joined with " / ", ":" or "," on one line; each cell is verbatim, and the page layout is a table. Where two adjacent sentences are quoted together they are on neighbouring lines of the page.

Page dates as printed on the pages: Daily Loss Limit (30 June 2026), Maximum Loss Limit ("Updated over 3 weeks ago"), Order Types, Fills, and Slippage ("Updated over a month ago"), Express Funded Account Parameters (5 August 2026), What is the Scaling Plan? (16 July 2026), Payout Policy ("Updated yesterday"), Risk Adjustments (Updated today), Economic Releases (Updated today), Trading Combine Parameters ("Updated over 3 weeks ago"), Pricing and Payment Questions ("Updated over 2 weeks ago"), Express Funded Account Activation ("Updated over 3 weeks ago"), Prohibited Trading Strategies (10 June 2026), Understanding Hedging (27 July 2026).

Not readable from this sandbox: `marketing.topstep.com/express-funded-account-rules/` returned HTTP 503, and the image host `downloads.intercomcdn.com` is blocked by the proxy (HTTP 403), which matters for the scaling ladder (section 2).

## 1. How a daily-loss-limit or maximum-loss-limit liquidation fills

**Bottom line: Topstep publishes no fill price, no slippage figure and no cap on the overshoot. It says the fill is a market order at the best available price and that it can land on either side of the limit.** The engine books a Daily Loss Limit hit exactly at -$3,000; the pages say that is not guaranteed. The size of the overshoot stays a sensitivity (`TOPSTEP_DLL_SLIP`), not a known number.

Daily Loss Limit, <https://help.topstep.com/en/articles/10490293-daily-loss-limit-in-the-trading-combine-and-express-funded-account>:

> "Net P&L hits or exceeds the DLL during the trading day (5 PM CT – 3:10 PM CT):"
> "Open positions are flattened"
> "Pending orders are canceled"
> "No new trades until 5 PM CT next session"
> "Example: $150K Trading Combine® with a $3,000 DLL. Net P&L hits -$3,000 → auto-liquidated. Back at it tomorrow at 5 PM CT."
> "Triggering it is not a rule violation — it's a forced break for the rest of that session."

This page says nothing about order type, fill price or slippage.

Maximum Loss Limit, <https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit>:

> "Risk limits are monitored in real-time using Net P&L — both realized and unrealized. If your account touches or falls below a limit at any point, it's a violation and liquidation triggers immediately."
> "When liquidation fires, positions close via market orders. Slippage and price movement during execution can push your final realized balance back above the limit — but the violation already happened based on unrealized P&L."
> "Final balance above the limit doesn't matter. The breach happened first. Same logic applies to the Daily Loss Limit, Personal Daily Loss Limit, and Personal Daily Profit Target."
> "Your Maximum Loss Limit is calculated on real-time unrealized P&L. The moment your open loss hits the threshold, the system automatically closes your position via market order; slippage may push the fill slightly above or below the limit, but if the threshold was breached while the trade was open, the liquidation stands."

Order Types, Fills, and Slippage, <https://help.topstep.com/en/articles/8765442-understanding-slippage-in-the-futures-market-causes-and-mitigation-strategies>:

> "Slippage can push your fill beyond your stop level. When that happens, realized losses may exceed your Maximum Loss Limit or Personal Daily Loss Limit — triggering automatic liquidation. This isn't a platform error. It's how orders fill in fast or volatile markets."
> "The system monitors your unrealized P&L in real time."
> "Your threshold gets crossed."
> "The system detects the breach and generates an order."
> "That order goes to the exchange and fills at the best available price."
> "Fast process. But in fast-moving markets, milliseconds matter. In normal conditions, the difference is minimal. In volatile markets, it can be significant. Exact price execution is never guaranteed with a platform risk tool."
> "A resting stop order sits at the exchange and fills when price reaches it — it can execute before a platform risk tool liquidation even triggers."
> "How does slippage affect my Maximum Loss Limit? Slippage can push your account past your Maximum Loss Limit faster than you expect. When that limit is hit, the system automatically liquidates your position."

Terms of Use section 27, <https://www.topstep.com/terms-of-use/>:

> "If the Net P&L should hit or exceed the Daily Loss Limit during the trading day (5:00 PM CT-3:10 PM CT), the account will hit a soft breach and will be auto-liquidated for the remainder of the then current trading session. This means any open trading positions will be flattened, any pending orders will be canceled, and your account will be prevented from placing any new trades until the start of the next trading day (5:00 PM CT)."

Reading (mine): the Slippage page describes the Personal Daily Loss Limit; the Daily Loss Limit bought at checkout is a different setting, but the Maximum Loss Limit page applies the same logic to it ("Same logic applies to the Daily Loss Limit"). So the purchased limit liquidates by market order at the next available price, monitored on unrealised P&L, with an unbounded and unpublished overshoot. Resting stop orders at the exchange can fill before the platform's liquidation fires.

What the page does settle in the engine's favour: the Maximum Loss Limit is checked intraday on unrealised P&L (an earlier Topstep blog said end-of-day only; this page says "The MLL updates at the end of each trading day but is monitored in real time throughout the session. Both realized and unrealized P&L count toward it."), and a Daily Loss Limit hit is a one-session lockout, not an account breach.

Engine status: the intraday monitoring is confirmed. The exact-at--$3,000 booking is **optimistic against the pages**; keep `TOPSTEP_DLL_SLIP` as the sensitivity and report results at slippage 0 and at the +$250 case from FINDINGS. The same overshoot applies to a Maximum Loss Limit hit, which the engine also books at the limit.

## 2. The 150K XFA scaling ladder

**Bottom line: the 150K ladder is on Topstep's Scaling Plan page only as an image ("XFA charts - hc.png") that this sandbox cannot download. It remains unresolved first-hand.** The `3/4/5/10/15` ladder in the old notes is **not** the 150K XFA ladder: it is the first five rows of the $250K Topstep Labs ladder (see below).

What Scaling Plan page says in text, <https://help.topstep.com/en/articles/8284223-what-is-the-scaling-plan>:

> "The Scaling Plan is an Express Funded Account® (XFA) objective. It sets your Maximum Position Size — the max contracts you can hold at one time — based on your current account balance."
> "Your XFA starts at a $0 balance"
> "As your balance grows, so does your buying power — shown below"
> "Key rule: Your max contracts do not increase mid-session. Hit the threshold to release more buying power? Wait for the next session."
> "10:1 ratio: 1 Mini = 10 Micro contracts. Limits are based on mini-contract equivalents"
> "Errors corrected in under 10 seconds are ignored. Leave too many contracts on for 10+ seconds and your account may be reviewed."

Text from the Risk Adjustments page, <https://help.topstep.com/en/articles/13613539-risk-adjustments-high-risk-high-volatility>, which gives the ladder as text but **only for restricted products** (CL, QM, MCL, HO, RB, GC, MGC; "3/6/9 plan"), capped at the plan's ceiling:

> "150K Express Funded Account
> Below $1500 balance = 1 contract
> $1,500 balance = 2 minis / 25 micros
> $2,000 balance = 3 minis / 31 micros
> $3,000 balance = 6 minis / 62 micros
> $4,500 balance = 9 minis / 93 micros"
> "For Express Funded Accounts, position limits for restricted products have a ceiling of the contracts listed above (CL = 3/6/9, for example). However, a trader's actual limit may be lower depending on their current account balance, as the Scaling Plan applies independently."

The balance thresholds ($1,500, $2,000, $3,000, $4,500) are the same as the engine's. Inference (mine, not Topstep's words): 1/2/3/6/9 is 60% of 3/4/5/10/15 rounded down (1.8, 2.4, 3.0, 6.0, 9.0), and the Combine cap is 15 minis, so a standard 150K ladder of 3/4/5/10/15 is consistent with this table. It is consistent, not proven.

The source of the old `3/4/5/10/15` figures, Topstep Labs, <https://help.topstep.com/en/articles/15520357-topstep-labs>, in the $250K product's XFA phase ("Max Contracts: 25 mini / 250 micro"):

> "Scaling Plan: Balance / Lots: Below $1,500 / 3 Lots; $1,500 - $2,000 / 4 Lots; $2,000 - $3,000 / 5 Lots; $3,000 - $4,500 / 10 Lots; $4,500 - $6,000 / 15 Lots; $6,000 - $8,000 / 20 Lots; Above $8,000 / 25 Lots"

A $250K product, a different program.

Engine status: thresholds consistent; the contract values are **unresolved**. Two rule details the engine should be checked against: contracts do not rise mid-session (the new limit applies from the next session), and a payout that drops the balance to a lower tier lowers the cap ("If a Payout reduces your balance to a lower tier, your maximum contract size decreases accordingly."). To settle the values, read the image by hand at the Scaling Plan page, or fetch it with the image host allowed.

## 3. Payout processing time and frequency

Payout Policy, <https://help.topstep.com/en/articles/8284233-topstep-payout-policy>.

Frequency and eligibility (Standard path):

> "Request Payouts in the Express Funded Account (XFA) and Live Funded Account (LFA) via your Topstep Dashboard"
> "Available during CME market hours: Sunday 5 PM CT – Friday 5 PM CT (excluding holidays)"
> "Minimum Payout: $125"
> "5 winning days of $150+ Net P&L. Days don't need to be consecutive. A day locks in at 4:00 PM CT. The trading day your Payout is requested doesn't count toward your next 5."
> "Positive net profit since your last Payout. You must have earned at least $0.01 since your last Payout. Your first Payout is exempt."
> "After each Payout: Your Maximum Loss Limit (MLL) resets to $0 permanently, and your 5-day count restarts."
> "Max Payout per request: 50% of your account balance up to the cap below." with the $150K cap at "$5,000" (Standard) and "$6,000" (Consistency).
> "Each XFA follows its own Payout Policy independently. Requests from one account don't affect another."

There is no stated limit on how many payouts per month; the pacing is the 5-winning-day count (3 days plus the 40% test on the Consistency path) and the positive-net-profit test.

Processing time, from the same page:

> "Prop-to-Brokerage Payouts: US only, Same day* (*If requested by 12 pm CT), No Fee"
> "Aeropay: US banks only, Instant after approval, No Topstep fee**"
> "Wise: China, Canada, UK, 1-3 business days, No Topstep fee**"
> "ACH: US banks only, 1-3 business days, $30"
> "Wire/SWIFT: International, 5-10 business days, $30"
> "Processing times are estimates and exclude weekends and holidays. Internal approval can take 1-3 business days."
> "Once you request a Payout, the funds are transferred immediately, allowing you to start trading again right away."
> "Example (ACH, $500 request): $500 minus $50 (10% split) minus $30 (processing fee) = $420 received"

Reading (mine): internal approval 1-3 business days, then method time. ACH/Wise is therefore 2-6 business days from request to bank, about 3-8 calendar days; Aeropay and Prop-to-Brokerage are approval time only. The balance drops at request, so the engine's cash-flow timing (3 calendar days) is inside the range but at the fast end for ACH, and the fee ($30 on ACH and wire) is not modelled. The money leaves the account balance immediately, so the processing delay matters for cash timing only, not for trading.

Other lines relevant to modelling:

> "Yes. Once the Payout amount has been deducted from your balance, hitting the MLL won't affect it."
> "Your copy-trading connection is automatically disabled while a Payout processes. Once the deduction is complete and your balance updates, you must manually re-enable it."
> "I hit the Daily Loss Limit, can I still request a Payout? Yes — but not yet. Hitting the Daily Loss Limit puts your account in a Temporary Violation. It lifts at the start of the next trading session."

Profit split. The page says "90/10 profit split — you keep 90%" and also:

> "Note for traders who joined the new Topstep dashboard before January 12, 2026: You receive 100% of your first $10,000 in lifetime profits. After that, the 90/10 split applies. Payouts from the Legacy dashboard do not contribute to the first $10K."

The engine uses 90% throughout; whether this note applies to a new account is not clear from the page (the wording is "joined ... before January 12, 2026"), so 90% stays.

Payout cap with a Daily Loss Limit, same page:

> "Limited Time Offering: Add a Daily Loss Limit (DLL) at checkout in the Trading Combine, and increase your Payout Cap once you pass and activate an Express Funded Account. Available starting at 3:30 PM CT on Tuesday, June 2nd."
> "$150K Standard: Current Cap $5,000, With DLL (new) $10,000"; "$150K Consistency: $6,000, $12,000"
> "No. Only if you already added the DLL when you purchased your Trading Combine. Adding it at the XFA stage doesn't unlock the higher cap."

Engine status: eligibility, cap ($5,000), minimum, 90% and the positive-net-profit rule are confirmed (rules v2 already has the last). The doubled cap is a limited-time offer on new Combine purchases; not used. Fee on payouts is not modelled.

## 4. News-window restrictions

Risk Adjustments, <https://help.topstep.com/en/articles/13613539-risk-adjustments-high-risk-high-volatility>, section "Tier 1 News Events Restrictions":

> "Ahead of Tier 1 News Events (for example, CPI, FOMC, NFP), Topstep temporarily restricts new opening positions on equity index products and metals in SIM to protect Traders from extreme volatility."
> "Minis (ES, RTY, YM, NQ, NKD, GC, SI, HG, PL): No new opening transactions permitted during the 10-minute window surrounding the release (5 minutes before, 5 minutes after)."
> "Micros (MES, M2K, MYM, MNQ, MGC): Opening transactions limited to 1, 3, 6, 9, and 15 contracts for the $25K, $50K, $100K, $150K, and $250K account sizes, respectively, during the window"
> "Other products (energies, rates, FX, etc.) are unaffected."
> "Existing Positions: Traders holding existing positions in energies and metals prior to 7:25 AM CT may keep them open and manage them through the Tier 1 News Events window."
> "Pending orders: Pending orders are not open positions. They will not fill during the 10-minute window, and new pending orders intended to execute during this timeframe cannot be placed. TP/SL brackets that are reducing/closing positions will be accepted during the window."

Windows as published (Central Time):

| Event | Announcement (CT) | Window (CT) | Same in ET |
|---|---|---|---|
| CPI | 7:30 AM | 7:25 to 7:35 AM | 08:25-08:35 |
| FOMC rate decision | 1:00 PM | 12:55 to 1:05 PM | 13:55-14:05 |
| FOMC press conference | 1:30 PM | 1:25 to 1:35 PM | 14:25-14:35 |
| Non Farm Payroll | 7:30 AM | 7:25 to 7:35 AM | 08:25-08:35 |

Table row for 150K: "150K | $150,000 | 9 | Blocked | Capped at 9 micros" (columns: Account, Margin, Max micros, Equity Minis, Effect during window).

Economic Releases, <https://help.topstep.com/en/articles/8284211-economic-releases>, lists other releases (Unemployment Rate 7:30 AM CT; FOMC Statement 1:00 PM CT; Crude Oil Inventories 9:30/10:00 AM CT; Natural Gas Inventories 9:30 AM CT; Crop Production 11:00 AM CT) and says "Topstep restricts trading for a short period during Tier 1 News Events." The page names CPI, FOMC and NFP as the Tier 1 events in the Risk Adjustments table; EIA and crop reports are listed as releases but not given a restriction window.

Prohibited Trading Strategies, <https://help.topstep.com/en/articles/10305426-prohibited-trading-strategies-at-topstep>:

> "Trading Maximum Position Size into Major News Events: Purposefully trading your full Maximum Position Size directly into a scheduled major news event."

Engine status: the earlier "gap" is now fully specified (NFP window is 08:25-08:35 ET, same as CPI). It concerns **new opening** transactions only: minis blocked, 9 micros opening cap, existing positions may run through, pending orders do not fill. A strategy that enters in the 08:25-08:35 ET or 13:55-14:05 / 14:25-14:35 ET windows on CPI, NFP or FOMC days needs this applied; a strategy that enters at 09:30 ET, or flat before the window, is unaffected. To be quantified in the backtest (days with these events, entries that fall inside). The note on "Maximum Position Size into" news is a conduct rule, which a full-size entry just before a release would touch.

## 5. Pricing and activation

Pricing, <https://help.topstep.com/en/articles/14289835-topstep-pricing-and-payment-questions>:

> "Trading Combine: Account Size / Standard Path / No Activation Fee Path: 50K $49/month $95/month; 100K $99/month $149/month; 150K $199/month $229/month"
> "Important: Paths cannot be changed after purchase."
> "Standard Path: Lower monthly cost. $149 Activation Fee charged once per Express Funded Account (XFA) earned."
> "No Activation Fee Path: Higher monthly cost. No fee when you pass."
> "Reset Pricing: 150K $199 (Standard Path), $229 (No Activation Fee Path)"
> "Reactivation fees: $50K XFA — $599; $100K XFA — $699; $150K XFA — $829" and "Reactivation window: 30 calendar days from account closure."
> "Responsible Trading Discount: ... when you add a DLL at purchase for No Activation Fee Trading Combines, Express Funded Account Activations, and Back2Funded Reactivations." "$30 off → 150K" (No Activation Fee Trading Combine), "$50 off Reactivation Fees". "Discount recurs monthly. The DLL is fixed and carried into your XFA when you pass."
> "Level 1 (Top of Book) ... Included"; "Level 2 (DOM) ... $38/month"

Commissions: "Commissions and fees are deducted automatically from your balance with each filled trade and show up in your Net P&L."

Activation, <https://help.topstep.com/en/articles/8284217-express-funded-account-activation> and Express Funded Account Parameters, <https://help.topstep.com/en/articles/8284215-express-funded-account-parameters>:

> "Standard Path: $149 one-time activation fee per XFA. No Activation Fee Path: No activation fee." (Activation page)
> "Your Express Funded Account starts with a $0 balance — this is expected and correct. ... Your Trading Combine balance does not carry over to your XFA." (Activation page)
> "If you already have 5 active XFAs or a Live Funded Account®, the button will be greyed out. You'll need to close an account before activating a new one." (Activation page)
> "How long does it take to get my XFA? Minutes after paying the Activation Fee. If you pass on a Friday, your XFA won't be available until markets reopen at 5 PM CT Sunday." (Parameters page)
> "You can hold up to 5 active XFAs at a time." "If you pass a Trading Combine® while at the limit, the new XFA goes on hold until one closes." (Parameters page)
> "XFAs with no trading activity for more than 30 days may be closed. XFAs can't be put on hold." (Parameters page)
> "Your size matches the Trading Combine® you passed and is locked before and after activation. Once you select XFA Consistency or Standard, that choice cannot be changed either." (Parameters page)
> "What happens if I breach the Maximum Loss Limit? Your account is liquidated and closed at the end of the day. Back2Funded lets you reactivate up to 2 times at the same size and Payout policy." (Parameters page)

Trading Combine Parameters, <https://help.topstep.com/en/articles/8284197-trading-combine-parameters>: "$150K | 15 | 150" (max contracts, max micros); "Meet the Consistency Target — your best single day should stay below 55% of your Profit Target to avoid increasing your Consistency Target."; "You can pass in as few as two days"; "The trading day runs from 5:00 PM CT through 3:10 PM CT the next calendar day." The Combine profit target ($9,000) and the $4,500 Maximum Loss Limit are stated for the 150K size on the Maximum Loss Limit page (table: "$150K $4,500"); the $9,000 target is not printed on the Combine Parameters page as extracted, so it is carried from the earlier notes unconfirmed here.

Engine status: price confirmed: Standard $199 + $149 activation, or No Activation Fee $229 less the $30 Responsible Trading Discount = $199 with $0 activation, both only with the Daily Loss Limit bought. The engine's $199 per 21 sessions, no activation fee, matches the second path. Two things the engine should decide on: the Standard path's one-off $149 per XFA, and reset prices ($199 or $229, and $199 if the discount is also applied to a reset; the discount is stated for the Combine purchase and reactivations, not explicitly for resets). The earlier "$149/month" product page is not confirmed by this page (150K Standard is $199/month).

## 6. Rules on automation and trading several accounts

Automation, Express Funded Account Parameters and Trading Combine Parameters:

> "Can I use automated strategies? Yes, with conditions. Topstep won't help set them up or cover malfunctions. Review Prohibited Conduct and Prohibited Trading Strategies first." (Parameters page)
> "Can I use automated trading strategies? Yes, with conditions. Topstep won't help set up or troubleshoot automated strategies, and no exceptions are made for errant trades or malfunctions. Test on your Practice Account first and review Prohibited Conduct and Prohibited Trading Strategies before going live. You can also use a trade copier to duplicate trades across multiple accounts." (Combine Parameters page)

Multiple accounts and the trade copier:

> "Can I trade across multiple XFAs at once? Yes, up to $750K buying power using a trade copier. See What is a Trade Copier. Hedging across accounts and coordinated trading with others is prohibited. See Understanding Hedging and Topstep's Terms of Use." (Parameters page)
> "Can I have more than 1 XFA? Yes, up to 5 active at a time." (Parameters page)
> "Can I have more than one Trading Combine? Yes. There's no limit to the number of Trading Combines you can have." (Combine Parameters page)
> "Single Profile Policy: All Trading Combines, Express Funded Accounts, and Live Funded Accounts must be under 1 Topstep profile. Multiple profiles are a Terms of Use violation and can result in account closure or suspension." (Combine Parameters page)

Trade copier, <https://help.topstep.com/en/articles/14434175-topstepx> section "Trade Copier":

> "The Trade Copier lets you mirror trades from a Lead account to 1 or more Follower accounts automatically — so you can run the same strategy across multiple accounts without placing every order manually."
> "The Lead account must have the lowest Maximum Position Size (MPS) of all accounts in the group."
> "If the Lead hits its Daily Loss Limit or is auto-liquidated, all Follower accounts are flattened with it and no further trades copy while the Lead is locked out."
> "If the Scaling plan does not match between the Leader and followers - follower account will be automatically removed at the start of the trading day"
> "When a Payout request is submitted, Follower accounts are automatically unlinked."
> "Executions are mirrored — but exact fill price and timing may vary slightly due to slippage, liquidity, and market conditions."

Prohibited, Prohibited Conduct, <https://help.topstep.com/en/articles/10296582-prohibited-conduct> (the Terms of Use section 27 has the same list):

> "Coordinated trading — performing trades in concert with others (including unconnected accounts or third parties) to pool risk, hedge aggregate positions, or trade the same or opposite strategy simultaneously"
> "Cross-account hedging (single-user) — holding opposite positions across multiple accounts simultaneously"
> "Unfair technology — using software, AI, ultra-high speed systems, or mass data entry to gain an unfair advantage"
> "Account stacking — repeatedly hitting the Maximum Loss Limit in one account and switching to another to repeat high-risk attempts"
> "Excessive purchases of Trading Combines or Resets"
> "Price exploitation — strategies designed to exploit errors in price display or data feed delays"; "Trading outside the best bid or offer"; "Using an external or slow data feed to trade"
> "Possible responses ...: Warning, Deletion of the impacted trading day, Account Reset, Permanent account closure, Delay or denial of a Payout request"

Prohibited Trading Strategies, <https://help.topstep.com/en/articles/10305426-prohibited-trading-strategies-at-topstep> ("The Do's and Don'ts of SIM Fills"):

> "Running scalping algorithms designed to exploit unrealistic SIM fills"
> "Making hundreds of rapid trades to take advantage of preferential queue position in SIM"
> "Repeatedly exploiting the relative lack of slippage in SIM to achieve impossible stop-loss execution"
> "Using tight brackets or auto-breakeven to take advantage of favorable SIM fills"
> "The behaviors above are intentional and systematic — usually hundreds or thousands of trades per day, with average durations measured in seconds, not minutes."

Understanding Hedging, <https://help.topstep.com/en/articles/13747047-understanding-hedging>:

> "Cross-account hedging means simultaneously going long and short the same or correlated instrument across multiple accounts — such as MES/ES, MNQ/NQ."
> "Can I trade the same instrument across multiple accounts? Yes! You can trade the same markets across different accounts. What's prohibited is holding opposite positions simultaneously in a way that eliminates market risk."
> "Does Topstep allow brief or unintentional overlaps of opposing positions? No. Opposing positions are prohibited even if the overlap is brief or unintentional."
> "Accounts closed for hedging violations are not eligible for payouts, and any associated profits cannot be withdrawn."
> "Technical glitches and copy trading software can create temporary opposite positions — our system accounts for that. But you're still fully responsible for all activity across your accounts, including anything created by automated systems or third-party tools."

Terms of Use, <https://www.topstep.com/terms-of-use/>, which conflicts with the help-centre pages on the count of accounts:

> "Your Account is personal to you and you are only permitted to have one Account. ... If Topstep determines in its sole and absolute discretion you have opened multiple Accounts, Topstep reserves the right to suspend or terminate your Account and subsequently created Accounts."

The help centre (up to 5 XFAs, any number of Combines, one profile) is the operative reading for Combine and XFA accounts, but the Terms of Use sentence is broader and this is a wording conflict I cannot settle from the pages. Also from the Terms: "your positions will automatically be flattened (e.g., closed out) just prior to the end of the then current trading session (e.g., about ten seconds prior to the closing bell at 3:10 PM CT)."

Engine status. What the rules allow: an automated strategy, run on up to 5 XFAs through the Topstep trade copier, one Topstep profile, any number of Combines. What bites a strategy farm: (a) the copier's Lead must have the lowest position size; (b) copies are unlinked at every payout request and must be re-enabled by hand; (c) if the Lead hits the Daily Loss Limit all followers are flattened; (d) different strategies on different accounts that take **opposite or offsetting positions in the same or correlated contracts at the same time** (including ES against MES) are cross-account hedging, with liquidation and, in the end, closure without payout; (e) "Account stacking" and "Excessive purchases of Trading Combines or Resets" are prohibited conduct, which a plan built on repeated Combine attempts to reach the funded stage should read against; (f) short-holding scalping bots with tight brackets sit close to the SIM-fills language. None is a number the engine can take; they are constraints on how the farm is run. The engine's multi-account run should be checked for opposing positions across accounts.

## Summary against the engine

| Rule | First-hand text (page) | Engine | Status |
|---|---|---|---|
| Combine 150K | "$150K | 15 | 150" contracts; MLL "$150K $4,500"; consistency "your best single day should stay below 55% of your Profit Target" | same | confirmed ($9,000 target not on the page as extracted) |
| Maximum loss limit | trails end-of-day balance, locks at $0 (XFA: "Once your balance reaches $2,000, the MLL locks at $0 permanently" for 50K; $4,500 for 150K), "monitored in real time" on realised and unrealised P&L; XFA breach: "permanently closed" | intraday breach, lock at $0 | confirmed |
| Liquidation fill | market order at best available price, either side of the limit, overshoot unbounded, no figure published | booked exactly at the limit | **optimistic**: keep `TOPSTEP_DLL_SLIP` sensitivity |
| Daily loss limit | optional, set at purchase, carried into XFA, $3,000 on 150K; a hit flattens, cancels, blocks until 5 PM CT; "not a rule violation" | soft limit at -$3,000 both stages | confirmed |
| Scaling ladder, 150K | image only; restricted-product table 1/2/3/6/9 at $0/1,500/2,000/3,000/4,500; contracts rise from the next session; payout can lower the tier | `3/4/5/10/15` | **unresolved** (that ladder is the $250K Labs one) |
| Payout eligibility | 5 winning days of $150+, not consecutive, day locks 4:00 PM CT, request day excluded; positive net profit since last payout (first exempt); 50% of balance, cap $5,000; min $125; MLL to $0 after each payout | same (v2) | confirmed |
| Payout timing | approval "1-3 business days", then Aeropay instant, ACH/Wise 1-3 business days, wire 5-10; ACH/wire fee $30; balance deducted at request | 3 calendar days, no fee | within range, fast end; fee not modelled |
| Profit split | "90/10"; note: 100% of first $10,000 for dashboard joined before 12 January 2026 | 90% | conservative |
| Payout cap with a DLL | limited offer, new Combine purchases, $10,000 / $12,000 on 150K | $5,000 | not used |
| Tier 1 news | CPI, NFP 08:25-08:35 ET, FOMC 13:55-14:05 ET, press conference 14:25-14:35 ET; minis blocked, 9 micros opening cap at 150K; existing positions may run | not modelled | **gap, now fully specified**; to quantify |
| Pricing | 150K: $199 Standard + $149 per XFA, or $229 No Activation Fee, $30 off with a DLL = $199; resets $199/$229; reactivation $829 | $199 per 21 sessions, $0 activation | matches the second path |
| Activation | minutes after fee; Friday pass opens Sunday 5 PM CT; up to 5 active XFAs, extra goes on hold; inactive >30 days may be closed | next session, cap 5 | conservative |
| Automation / accounts | automation "Yes, with conditions"; up to 5 XFAs; trade copier; cross-account hedging, coordinated trading, account stacking prohibited; Terms of Use "only permitted to have one Account" | not modelled | constraint on operation, see section 6 |
| Live Funded Account | not re-read | not modelled | n/a |
| Responsible Trading Program | not re-read | not modelled | n/a |

## Engine changes already made (rules v2)
* `net_pos_after_first` (`FundedSpec`, `firm_engine.topstep_150k(rules="v2")` by default, env `TOPSTEP_RULES=v1` reproduces everything up to FINDINGS section 27; mirrored in `paper/accounts.py`).
* `dll_slip` sensitivity (env `TOPSTEP_DLL_SLIP`, dollars beyond the limit, default 0).

## Candidates for rules v3 (not applied in this note)
1. Tier 1 news windows: block mini opens and cap micro opens at 9 inside the four windows on CPI, NFP and FOMC days.
2. Scaling ladder: replace `3/4/5/10/15` once the image is read; apply tier changes from the next session and after a payout.
3. Liquidation fill: keep slippage as a sensitivity; apply it to the Maximum Loss Limit too.
4. Payout cash timing and the $30 ACH or wire fee; Standard path's $149 activation fee if that path is costed.
