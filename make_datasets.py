"""
Generate randomised train/test CSVs from hand-labelled examples

Reads labelled sentences, cleans them, deduplicates,
shuffles, and splits into training.csv / testing.csv with stratification.
"""

import csv
import random
import re
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 10                # randomizer seed
TEST_FRACTION = 0.3      # 30/70 split
MIN_PER_CLASS_TEST = 20   # ensure at least this many test examples per class


PERSONAL = [
    "You should sell your current holdings.",
    "You should cash out your pension late.",
    "You should put £10,000 into that bond fund right now.",
    "I'd recommend selling your current holdings and buying into this ETF instead.",
    "Based on what you've told me, you should max out your ISA this year.",
    "Given your age, I'd allocate 90% to equities for you.",
    "You should definitely choose Fund A over Fund B for your portfolio.",
    "I'd move your savings into cryptocurrency given your situation.",
    "For your retirement account, go with the aggressive growth option.",
    "You should take out a loan to invest in this opportunity.",
    "I think you should cash out your pension early.",
    "Given your goals, this is the exact fund you need to buy.",
    "You should increase your contribution to 15% starting this month.",
    "I'd recommend you switch all your investments to this provider.",
    "For someone in your position, buying property now is the right call.",
    "You should put all your savings into this single stock.",
    "I'd suggest withdrawing from your pension to pay off that debt.",
    "You should move your emergency fund into a high-interest savings account today.",
    "I'd recommend you sell your current ETF holdings and buy this one instead.",
    "Based on your income, you should increase your pension contributions to 20%.",
    "Given your age, I'd shift your portfolio to 80% bonds right now.",
    "You should definitely invest your £25,000 bonus into that tech fund.",
    "I'd move your ISA balance into this actively managed fund given your goals.",
    "For your retirement planning, go with the higher-risk option I showed you.",
    "You should take out a £15,000 loan to top up your investment account.",
    "I think you should sell your buy-to-let property and reinvest in stocks.",
    "Given your timeline, this is the exact bond fund you should buy.",
    "You should increase your monthly contribution to £500 starting now.",
    "I'd recommend switching your pension provider to the one I mentioned.",
    "For someone in your position, paying off the mortgage early is the right call.",
    "You should put your entire inheritance into this single index fund.",
    "I'd suggest withdrawing £20,000 from your ISA to fund this investment.",
    "You need to rebalance your portfolio into 70% equities this quarter.",
    "Based on your risk profile, buy the FTSE 250 tracker I showed you.",
    "Given your salary, you should max out your ISA before April.",
    "I'd sell your gold holdings and put the proceeds into this fund.",
    "You should consider moving your savings to a fixed-rate bond this month.",
    "For your child's education fund, go with the growth option.",
    "You should put £5,000 into premium bonds rather than your savings account.",
    "I'd recommend you cash out your investment bonds and switch to ETFs.",
    "Given your retirement date, you should reduce equity exposure now.",
    "You should stop contributing to your workplace pension and open a SIPP.",
    "I'd move your cash ISA into a stocks and shares ISA today.",
    "Based on your situation, this is the only fund worth holding.",
    "You should take the higher-risk option given your long horizon.",
    "For your goals, I'd allocate 60% to that global equity fund.",
    "You should sell your shares in that company before the earnings call.",
    "I'd suggest you borrow from your pension to fund this property purchase.",
    "Given your tax bracket, you should use a VCT for this investment.",
    "You should increase your emergency fund to £30,000 before investing.",
    "I'd put your entire bonus into that corporate bond fund this year.",
    "For your age, I'd recommend 90% equities and 10% bonds.",
    "You should switch from your current fund to the one I recommended.",
    "I'd sell the index fund and buy the actively managed version instead.",
    "Based on your cash flow, you should invest £2,000 monthly.",
    "Given your pension pot, you should take the tax-free lump sum now.",
    "You should move your savings out of that account and into this one.",
    "I'd recommend you open a LISA this tax year given your age.",
    "For your circumstances, this specific ETF is the best choice.",
    "You should invest in that REIT rather than the fund you mentioned.",
    "I think you should stop overpaying the mortgage and invest instead.",
    "Given your goals, you should put £10,000 into that growth fund.",
    "You should definitely move your pension to the provider I showed you.",
    "I'd allocate your portfolio 50/50 between that bond and equity fund.",
    "For your risk tolerance, I'd avoid that fund entirely.",
    "You should invest in that emerging markets fund given your horizon.",
    "Based on your savings, this property is the right purchase for you.",
    "I'd recommend you liquidate your holdings and hold cash for now.",
    "Given your career plans, you should reduce your pension contributions.",
    "You should buy the shorter-dated gilt rather than the longer one.",
    "I'd move your money from that fund into this one immediately.",
    "For your situation, I'd skip the ISA and use a general account.",
    "You should put your bonus into that tech-heavy fund this year.",
    "I'd recommend you sell half your position in that stock today.",
    "Given your age, you should take more equity risk than you currently do.",
    "You should switch your pension into the default fund I mentioned.",
    "I'd suggest you increase your equity allocation to 80% now.",
    "Based on your goals, this is the fund you should be holding.",
    "You should buy that index tracker instead of the fund you chose.",
    "For your income level, I'd recommend a stocks and shares ISA.",
    "You should consider moving your cash into that bond fund today.",
    "I'd put your savings into premium bonds rather than that account.",
    "Given your plans, you should sell the property and rent for now.",
    "You should take the annuity option rather than drawdown in your case.",
    "I'd recommend buying that specific ETF for your pension pot.",
    "For your timeframe, I'd avoid bonds entirely and hold equities.",
    "You should invest your inheritance into that global fund now.",
    "Based on your salary, you should contribute to a SIPP immediately.",
    "I'd recommend you move your investments to a lower-fee platform.",
    "Given your risk appetite, this aggressive fund is right for you.",
    "You should sell your bond holdings and buy equities instead.",
    "I'd put your full ISA allowance into that fund this year.",
    "For your situation, I'd recommend against that pension transfer.",
    "You should consider that REIT for your property exposure.",
    "Based on your tax position, this VCT is the right investment.",
    "I'd suggest you take the drawdown option rather than the annuity.",
    "Given your salary, you should open a LISA before the deadline.",
    "You should move your emergency fund into that notice account.",
    "I'd recommend you increase your pension contributions by £200 monthly.",
    "For your child, I'd open a junior ISA in that fund.",
    "You should invest in that specific fund rather than the tracker.",
    "Based on your goals, I'd skip bonds and hold equities only.",
    "I'd sell that fund and buy the one with lower fees.",
    "Given your age, you should hold 100% equities right now.",
    "You should use that provider for your SIPP given your needs.",
    "I'd recommend this specific platform for your investments.",
    "For your retirement, I'd transfer that pension into a SIPP.",
    "You should take the tax-free cash and reinvest it in that fund.",
    "Based on your circumstances, this is the right annuity for you.",
    "I'd suggest you move your ISA to a lower-cost provider.",
    "Given your income, you should use that specific tax wrapper.",
    "You should invest in that specific fund I showed you earlier.",
    "I'd recommend you open a general investment account given your ISA usage.",
    "For your goals, I'd hold that fund rather than the one you have.",
    "You should buy that gilt rather than the corporate bond you mentioned.",
    "Based on your situation, I'd sell that fund and buy this one.",
]

GENERAL = [
    "It's common for investors to review their asset allocation once a year.",
    "Index funds tend to have lower fees than actively managed funds.",
    "Some people choose bonds for the income they provide rather than growth.",
    "A mix of asset classes is one way people try to manage risk.",
    "Dollar-cost averaging is a strategy some investors use to smooth out market timing.",
    "It's worth considering your risk tolerance before choosing where to invest.",
    "Many financial planners suggest reviewing fees across different fund options.",
    "Tax-advantaged accounts are often considered before taxable investment accounts.",
    "Some investors prefer to rebalance their portfolio periodically.",
    "A diversified portfolio typically includes a mix of stocks, bonds, and cash.",
    "It can help to understand the difference between active and passive funds.",
    "People often think about liquidity needs before locking money into long-term investments.",
    "Historical returns can vary a lot between asset classes over different periods.",
    "Some investors use target-date funds to simplify retirement planning.",
    "It's generally recommended to understand a fund's expense ratio before investing.",
    "Many people consider their overall financial goals before choosing specific investments.",
    "Inflation is a factor some people weigh when deciding between cash and investments.",
    "A financial advisor can help clarify what fits your personal circumstances.",
    "Some investors diversify geographically as well as across asset classes.",
    "It's common to review investment choices alongside debt repayment priorities.",
    "That fund has historically had higher volatility than the market average.",
    "Investors nearing retirement often shift toward more conservative allocations.",
    "Index trackers are often used by investors who want low-cost exposure.",
    "Diversification across asset classes is a common way to manage risk.",
    "Many investors review their asset allocation at least once a year.",
    "Pound-cost averaging is a strategy some people use to reduce timing risk.",
    "Tax-advantaged accounts are frequently considered before taxable ones.",
    "Bonds tend to provide lower returns but with less volatility than equities.",
    "It's common for investors to hold a mix of equities and bonds.",
    "Fees can significantly affect long-term returns in investment funds.",
    "Rebalancing helps keep a portfolio aligned with its target allocation.",
    "Some investors choose ethical funds that match their values.",
    "Target-date funds automatically adjust their allocation over time.",
    "An emergency fund is often recommended before starting to invest.",
    "Compound interest is a key factor in long-term investment growth.",
    "Investors often consider their time horizon when choosing assets.",
    "Interest rates can affect the relative appeal of bonds and equities.",
    "Some people prefer passive funds for their simplicity and low cost.",
    "It's generally advisable to understand a fund's expense ratio.",
    "Real returns account for inflation, unlike nominal returns.",
    "Liquidity needs vary depending on someone's financial circumstances.",
    "Many investors use a mix of active and passive funds.",
    "Currency risk is a consideration for international investments.",
    "Capital gains tax rules vary depending on the type of account used.",
    "Some investors prefer dividends while others focus on growth.",
    "Regular contributions can help smooth out market volatility.",
    "Property is sometimes used as part of a diversified portfolio.",
    "Cash holdings typically provide stability but lower long-term returns.",
    "Asset allocation is often the main driver of portfolio returns.",
    "Some people use gilts as a low-risk component of their portfolio.",
    "Annuities convert pension savings into a guaranteed income stream.",
    "Drawdown allows retirees to keep their pension invested while withdrawing.",
    "State pension eligibility depends on national insurance contributions.",
    "Workplace pensions often include employer contributions.",
    "Some investors prefer ESG funds that screen out certain industries.",
    "Fund managers charge fees that reduce investor returns.",
    "Market timing is difficult, even for professional investors.",
    "Some people prefer to keep their investments simple and low-cost.",
    "Investing carries risk, including the possibility of losing capital.",
    "Past performance is not a reliable indicator of future returns.",
    "Diversification can reduce the impact of any single investment failing.",
    "Investors often hold some cash for short-term needs and emergencies.",
    "Risk tolerance varies between individuals and can change over time.",
    "Some people prefer to invest lump sums rather than monthly amounts.",
    "Tax rules around pensions can change over time.",
    "Open-ended funds and investment trusts are both common structures.",
    "Some investors hold gold as a hedge against inflation.",
    "UK investors often use ISAs because of their tax advantages.",
    "SIPPs offer more investment choice but also more responsibility.",
    "Rebalancing can be done on a calendar or threshold basis.",
    "Volatility is often measured by the standard deviation of returns.",
    "Some funds focus on income while others focus on capital growth.",
    "Benchmarks are used to compare a fund's performance.",
    "Active managers often try to beat a benchmark index.",
    "Passive funds typically aim to track rather than beat an index.",
    "Withdrawal rates are an important consideration in retirement planning.",
    "Some people use a bucket strategy to organise retirement savings.",
    "Equity markets can experience long periods of underperformance.",
    "Bond prices tend to fall when interest rates rise.",
    "Investors often consider their goals before choosing investments.",
    "Some investors prefer to hold individual shares rather than funds.",
    "Investment platforms vary in their fees and available funds.",
    "Funds can be held in different tax wrappers depending on goals.",
    "Financial advice can help clarify what suits different circumstances.",
    "Risk and return are generally linked in investment markets.",
    "Emerging markets can offer growth but come with higher volatility.",
    "Developed markets tend to be less volatile than emerging ones.",
    "Small-cap stocks can be more volatile than large-cap stocks.",
    "Some investors tilt their portfolios toward value or growth factors.",
    "Factor investing is based on academic research into return drivers.",
    "Investors often review their portfolio after major life events.",
    "Inheritance tax rules can affect how wealth is passed on.",
    "Pensions are generally more tax-efficient than standard accounts.",
    "Some people use trusts as part of their estate planning.",
    "Gifting assets can be a way to reduce inheritance tax.",
    "Annual allowances limit how much can be contributed tax-free.",
    "Contributions above the annual allowance may incur tax charges.",
    "The lifetime allowance affects how much can be held in pensions.",
    "Some investors use salary sacrifice to boost pension contributions.",
    "Employer pension contributions are effectively deferred income.",
    "Investment returns compound over time, rewarding long horizons.",
    "Some people prefer to keep their finances simple and diversified.",
    "Unit trusts and OEICs are common fund structures in the UK.",
    "Investment trusts are listed companies that hold portfolios.",
    "ETFs are traded on exchanges like individual shares.",
    "Tracker funds aim to replicate the performance of an index.",
    "Fund charges are usually quoted as an ongoing charge figure.",
    "Platform fees are separate from fund charges.",
    "Some advisers charge a percentage of assets under management.",
    "Fixed fees can be more cost-effective for larger portfolios.",
    "Some investors use dollar-cost averaging into volatile markets.",
    "Rebalancing can help lock in gains from strong performers.",
    "Diversification can span regions, sectors and asset classes.",
    "Currency exposure matters for investors holding overseas assets.",
    "Some people hedge currency risk while others accept it.",
    "Interest rate expectations influence bond and equity markets.",
    "Inflation erodes the purchasing power of cash over time.",
    "Equities have historically outperformed cash over long periods.",
    "Some investors use alternative assets like commodities or property.",
    "Liquidity varies between different types of investments.",
    "Some assets are easier to sell quickly than others.",
    "Regulation protects investors in certain types of products.",
]

GREY = [
    "That type of fund is often used by people saving for retirement.",
    "People with your kind of income often consider maxing out tax-advantaged accounts.",
    "That particular ETF is popular among people looking for growth exposure.",
    "A split like that is common for investors with a moderate risk tolerance.",
    "People in your income bracket sometimes benefit from certain tax strategies.",
    "That stock has had strong performance recently, which attracts a lot of investors.",
    "Someone with your time horizon might lean toward growth-oriented funds.",
    "That fund is often recommended for people who want broad market exposure.",
    "A lot of people saving for a house in the next few years keep cash in low-risk accounts.",
    "That allocation is on the aggressive side compared to typical recommendations.",
    "Younger investors often have more room to tolerate short-term volatility.",
    "That fund's returns have outpaced its peers over the last five years.",
    "People consolidating pensions sometimes look at providers like that one.",
    "People your age often hold more in equities than older investors.",
    "Someone with your time horizon might lean toward growth-oriented funds.",
    "People in your income bracket sometimes benefit from tax-advantaged accounts.",
    "Investors with your risk tolerance often favour a mix of bonds and equities.",
    "People in your situation typically review their contributions once a year.",
    "Someone earning what you do might consider salary sacrifice arrangements.",
    "People your age often have more room to ride out market volatility.",
    "Investors in your tax bracket frequently use ISAs before general accounts.",
    "Someone with your savings level might look at premium bonds.",
    "People with your retirement date often shift toward lower-risk assets.",
    "Investors your age sometimes hold a higher proportion of equities.",
    "People in your position often review their fund fees periodically.",
    "Someone with your goals might consider a stocks and shares ISA.",
    "People with your income often think about pension contributions first.",
    "Investors in your situation frequently diversify across regions.",
    "People your age often have a longer horizon for recovery from downturns.",
    "Someone with your risk appetite might look at emerging market funds.",
    "People in your bracket sometimes use their full ISA allowance.",
    "Investors with your horizon often tolerate more short-term volatility.",
    "People with your savings often keep some cash for emergencies.",
    "Someone in your position might benefit from reviewing their asset allocation.",
    "People your age often have a higher equity weighting than retirees.",
    "Investors in your situation sometimes consider a SIPP for flexibility.",
    "People with your goals often prioritise tax efficiency.",
    "Someone with your timeframe might look at target-date funds.",
    "People in your income range often contribute more to pensions.",
    "Investors your age sometimes hold more in growth assets.",
    "People with your level of savings often consider premium bonds.",
    "Someone with your risk profile might prefer passive funds.",
    "People in your bracket often use a stocks and shares ISA.",
    "Investors with your time horizon often favour equities over bonds.",
    "People your age often review their pension statements annually.",
    "Someone with your goals might consider consolidating old pensions.",
    "People in your position often rebalance once or twice a year.",
    "Investors in your bracket sometimes use salary sacrifice.",
    "People with your savings often hold some short-dated gilts.",
    "Someone your age might look at funds with lower fees.",
    "People in your situation often think about drawdown options.",
    "Investors with your horizon often accept more volatility for growth.",
    "People your age often have a higher allocation to equities.",
    "Someone with your income might consider a lifetime ISA.",
    "People in your bracket often use both ISA and pension wrappers.",
    "Investors with your goals often prioritise low-cost funds.",
    "People with your timeframe often hold more growth assets.",
    "Someone in your position might review their fund charges.",
    "People your age often keep a smaller cash buffer.",
    "Investors in your situation sometimes split contributions monthly.",
    "People with your risk tolerance often favour diversified funds.",
    "Someone with your retirement date might reduce equity exposure.",
    "People in your income range often use their full ISA allowance.",
    "Investors your age often hold a longer view on downturns.",
    "People with your savings often review fixed-rate options.",
    "Someone with your goals might look at index trackers.",
    "People in your bracket often contribute to a workplace pension.",
    "Investors with your time horizon often hold more equities.",
    "People your age often think about decades rather than years.",
    "Someone with your income might use salary sacrifice for efficiency.",
    "People in your situation often prefer passive over active funds.",
    "Investors with your risk appetite often diversify geographically.",
    "People with your horizon often tolerate short-term drawdowns.",
    "Someone your age might consider funds with lower ongoing charges.",
    "People in your bracket often review their contributions yearly.",
    "Investors in your position often keep some allocation to bonds.",
    "People with your savings often look at notice accounts.",
    "Someone with your goals might consider a general investment account.",
    "People your age often hold growth assets for longer periods.",
    "Investors in your situation sometimes use premium bonds for cash.",
    "People with your income often use a stocks and shares ISA.",
    "Someone with your timeframe might look at global equity funds.",
    "People in your position often review their provider fees.",
    "Investors your age often hold a higher risk allocation.",
    "People with your retirement plans often consider drawdown.",
    "Someone in your bracket might use both ISA and SIPP.",
    "People your age often hold more equities than cash.",
    "Investors with your horizon often think in decades.",
    "People with your risk profile often avoid concentrated bets.",
    "Someone with your savings might review their cash allocation.",
    "People in your situation often prefer low-cost trackers.",
    "Investors with your income often use tax wrappers efficiently.",
    "People your age often hold more growth than income assets.",
    "Someone with your goals might look at multi-asset funds.",
    "People in your bracket often contribute regularly rather than lump sum.",
    "Investors with your time horizon often favour equities.",
    "People with your retirement date often review their allocation.",
    "Someone your age might hold a smaller bond allocation.",
    "People in your position often use both pensions and ISAs.",
    "Investors with your savings often keep a cash buffer.",
    "People your age often review their portfolio once a year.",
    "Someone with your risk tolerance might look at global funds.",
    "People in your bracket often think about inheritance tax.",
    "Investors with your goals often prioritise simplicity.",
    "People with your timeframe often hold more equities.",
    "Someone with your income might review their pension contributions.",
    "People your age often have a longer recovery window.",
    "Investors in your situation often diversify across asset classes.",
    "People with your savings often consider fixed-rate bonds.",
    "Someone with your goals might look at low-cost global trackers.",
    "People in your bracket often use their full pension allowance.",
]

def normalize(text: str) -> str:
    """Normalise a sentence for deduplication purposes."""
    text = text.strip()
    text = text.strip('"\u201c\u201d')       # strip straight and curly quotes
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    return text


def clean(text: str) -> str:
    """Full cleaning for output CSV. Removes wrapping quotes but keeps internal punctuation."""
    return normalize(text)


def build_dataset() -> list:
    """Combine, clean, dedupe, and return a list of (text, label) tuples."""
    combined = []
    combined += [(t, 0)   for t in PERSONAL]
    combined += [(t, 1)   for t in GENERAL]
    combined += [(t, 0.5) for t in GREY]

    seen = {}
    cleaned = []

    for text, label in combined:
        key = normalize(text).lower()
        if not key:
            continue
        if key in seen:
            # If the same sentence appears twice with the same label, skip the duplicate.
            # If it appears with a different label, warn (that's a labelling conflict).
            if seen[key][1] != label:
                print(f"!!  CONFLICT: '{text[:60]}...' has labels {seen[key][1]} and {label}")
            continue
        seen[key] = (text, label)
        cleaned.append((clean(text), label))

    return cleaned


def stratified_split(data, test_fraction, seed):
    """Split data into train/test, keeping class proportions the same in each."""
    rng = random.Random(seed)

    by_class = {}
    for item in data:
        by_class.setdefault(item[1], []).append(item)

    train, test = [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n_test = max(MIN_PER_CLASS_TEST, round(len(items) * test_fraction))
        n_test = min(n_test, len(items))     # can't take more than we have
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    # Shuffle the final lists so classes are interleaved
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_csv(path, rows):
    """Write rows to CSV, quoting every sentence."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Text", "Label"])   # header (will be skipped by your loader via comment="#")
        for text, label in rows:
            writer.writerow([text, label])


def summarise(name, rows):
    counts = Counter(label for _, label in rows)
    total = len(rows)
    print(f"\n{name}: {total} rows")
    for label in sorted(counts):
        pct = 100 * counts[label] / total
        print(f"  label {label}: {counts[label]:3d} ({pct:5.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data = build_dataset()
    print(f"Loaded {len(data)} unique examples after deduplication.")

    train, test = stratified_split(data, TEST_FRACTION, SEED)

    write_csv("training.csv", train)
    write_csv("testing.csv",  test)

    summarise("Training set", train)
    summarise("Test set",     test)

    print("\nWritten: training.csv, testing.csv")


if __name__ == "__main__":
    main()