"""Generate acme_10q.txt: a synthetic ~100K-token quarterly filing for the lecture demos.

Acme Corp is fictional. The headline numbers: Q3 2026 revenue $4.2B
(Q2 2026: $3.9B, up 7.7%), operating margin 21.0% (Q2 2026: 19.4%). Everything else is
filler with the shape of a real 10-Q, so the document is long enough that sending all of it
in every prompt (v0) is visibly expensive. Output is deterministic.

    python demos/data/make_acme_10q.py
"""

from __future__ import annotations

import random
from pathlib import Path

OUT = Path(__file__).with_name("acme_10q.txt")
rng = random.Random(10)

# ---------------------------------------------------------------- quarterly model ($M)
QUARTERS = {
    #             cloud  software devices  cost   r&d  s&m  g&a  other shares
    "Q1 2025": (1705, 1110, 587, 1520, 700, 480, 192, -4, 518),
    "Q2 2025": (1801, 1130, 600, 1560, 730, 498, 195, -5, 517),
    "Q3 2025": (1902, 1155, 627, 1621, 760, 520, 190, -6, 516),
    "Q1 2026": (2010, 1170, 522, 1610, 790, 460, 187, 9, 514),
    "Q2 2026": (2169, 1192, 550, 1662, 815, 476, 199, 12, 513),
    "Q3 2026": (2398, 1226, 588, 1727, 842, 533, 225, 18, 512),
}
TAX_RATE = 0.19


def pnl(q: str) -> dict:
    cloud, sw, dev, cost, rd, sm, ga, other, shares = QUARTERS[q]
    rev = cloud + sw + dev
    gross = rev - cost
    op = gross - rd - sm - ga
    pre = op + other
    tax = round(pre * TAX_RATE)
    ni = pre - tax
    return dict(cloud=cloud, software=sw, devices=dev, revenue=rev, cost=cost, gross=gross,
                rd=rd, sm=sm, ga=ga, opex=rd + sm + ga, op=op, other=other, pretax=pre,
                tax=tax, net=ni, shares=shares, eps=ni / shares, margin=op / rev)


P = {q: pnl(q) for q in QUARTERS}


def nine(year: str) -> dict:
    qs = [P[f"Q{i} {year}"] for i in (1, 2, 3)]
    out = {k: sum(q[k] for q in qs) for k in qs[0] if k not in ("eps", "margin", "shares")}
    out["shares"] = round(sum(q["shares"] for q in qs) / 3)
    out["eps"] = out["net"] / out["shares"]
    out["margin"] = out["op"] / out["revenue"]
    return out


N26, N25 = nine("2026"), nine("2025")
Q3, Q2, PY = P["Q3 2026"], P["Q2 2026"], P["Q3 2025"]


def m(x: float) -> str:
    return f"({abs(x):,.0f})" if x < 0 else f"{x:,.0f}"


def pct(a: float, b: float) -> str:
    return f"{abs(a - b) / b * 100:.1f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(widths[0]) if i == 0 else str(c).rjust(widths[i])
                               for i, c in enumerate(r))
    return "\n".join([line(header), "-" * len(line(header))] + [line(r) for r in rows])


# ---------------------------------------------------------------- sections
def cover() -> str:
    return f"""UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 10-Q

QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the quarterly period ended September 30, 2026

ACME CORP
(Exact name of registrant as specified in its charter)

Delaware (State of incorporation)    84-2210931 (I.R.S. Employer Identification No.)
1 Acme Plaza, San Jose, California 95113
(408) 555-0100

Indicate by check mark whether the registrant (1) has filed all reports required to be filed by
Section 13 or 15(d) of the Securities Exchange Act of 1934 during the preceding 12 months, and (2)
has been subject to such filing requirements for the past 90 days. Yes [X] No [ ]

Large accelerated filer [X]  Accelerated filer [ ]  Non-accelerated filer [ ]
Smaller reporting company [ ]  Emerging growth company [ ]

As of October 24, 2026, there were 508,114,302 shares of the registrant's common stock outstanding.

NOTE: Acme Corp is a fictional company. This document was generated for teaching purposes.

TABLE OF CONTENTS
PART I. FINANCIAL INFORMATION
  Item 1. Financial Statements (Unaudited)
  Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations
  Item 3. Quantitative and Qualitative Disclosures About Market Risk
  Item 4. Controls and Procedures
PART II. OTHER INFORMATION
  Item 1. Legal Proceedings
  Item 1A. Risk Factors
  Item 2. Unregistered Sales of Equity Securities and Use of Proceeds
  Item 5. Other Information
  Item 6. Exhibits
Supplemental Operating Data
Signatures

SPECIAL NOTE REGARDING FORWARD-LOOKING STATEMENTS

This Quarterly Report on Form 10-Q contains forward-looking statements within the meaning of the
Private Securities Litigation Reform Act of 1995. Words such as "expect," "anticipate," "intend,"
"plan," "believe," "estimate," "may," "will," "should," and similar expressions identify
forward-looking statements. These statements include, among others, statements about our expected
revenue growth, data-center capacity, capital expenditures, operating margin, product roadmap,
customer demand, and the effects of macroeconomic conditions on our business. Forward-looking
statements are based on current expectations and are subject to risks and uncertainties, including
those described in Part II, Item 1A, "Risk Factors." Actual results may differ materially. We
undertake no obligation to update any forward-looking statement, except as required by law.
"""


def statements() -> str:
    cols = ["(in millions, except per share)", "Q3 2026", "Q3 2025", "9M 2026", "9M 2025"]
    sets = [Q3, PY, N26, N25]
    row = lambda label, k: [label] + [m(s[k]) for s in sets]
    income = table(cols, [
        row("Revenue: Cloud Services", "cloud"),
        row("Revenue: Enterprise Software", "software"),
        row("Revenue: Devices and Other", "devices"),
        row("Total revenue", "revenue"),
        row("Cost of revenue", "cost"),
        row("Gross profit", "gross"),
        row("Research and development", "rd"),
        row("Sales and marketing", "sm"),
        row("General and administrative", "ga"),
        row("Total operating expenses", "opex"),
        row("Operating income", "op"),
        row("Other income (expense), net", "other"),
        row("Income before income taxes", "pretax"),
        row("Provision for income taxes", "tax"),
        row("Net income", "net"),
        ["Diluted earnings per share"] + [f"{s['eps']:.2f}" for s in sets],
        row("Diluted weighted-average shares", "shares"),
    ])
    balance = table(["(in millions)", "Sep 30, 2026", "Dec 31, 2025"], [
        ["Cash and cash equivalents", "6,412", "5,288"],
        ["Short-term investments", "3,905", "3,640"],
        ["Accounts receivable, net", "2,731", "2,486"],
        ["Inventories", "418", "455"],
        ["Prepaid expenses and other current assets", "792", "701"],
        ["Total current assets", "14,258", "12,570"],
        ["Property and equipment, net", "11,946", "10,212"],
        ["Operating lease right-of-use assets", "2,318", "2,140"],
        ["Goodwill", "4,880", "4,880"],
        ["Intangible assets, net", "612", "731"],
        ["Deferred tax assets", "954", "903"],
        ["Other long-term assets", "1,077", "988"],
        ["Total assets", "36,045", "32,424"],
        ["Accounts payable", "1,214", "1,102"],
        ["Accrued expenses and other current liabilities", "2,486", "2,311"],
        ["Deferred revenue, current", "3,377", "3,018"],
        ["Short-term debt", "500", "0"],
        ["Total current liabilities", "7,577", "6,431"],
        ["Long-term debt", "5,982", "6,474"],
        ["Operating lease liabilities, non-current", "2,071", "1,930"],
        ["Deferred revenue, non-current", "882", "810"],
        ["Other long-term liabilities", "736", "702"],
        ["Total liabilities", "17,248", "16,347"],
        ["Total stockholders' equity", "18,797", "16,077"],
        ["Total liabilities and stockholders' equity", "36,045", "32,424"],
    ])
    cash = table(["(in millions)", "9M 2026", "9M 2025"], [
        ["Net income", m(N26["net"]), m(N25["net"])],
        ["Depreciation and amortization", "1,488", "1,214"],
        ["Stock-based compensation", "702", "640"],
        ["Deferred income taxes", "(51)", "(38)"],
        ["Changes in operating assets and liabilities", "122", "(87)"],
        ["Net cash provided by operating activities", "4,191", "3,153"],
        ["Purchases of property and equipment", "(3,214)", "(2,602)"],
        ["Purchases of investments, net of maturities", "(265)", "(412)"],
        ["Acquisitions, net of cash acquired", "0", "(318)"],
        ["Net cash used in investing activities", "(3,479)", "(3,332)"],
        ["Proceeds from issuance of debt, net", "0", "990"],
        ["Repurchases of common stock", "(1,105)", "(820)"],
        ["Dividends paid", "(463)", "(414)"],
        ["Taxes paid related to net share settlement", "(120)", "(98)"],
        ["Net cash used in financing activities", "(1,688)", "(342)"],
        ["Effect of exchange rates on cash", "(12)", "9"],
        ["Net increase (decrease) in cash", "(988)", "(512)"],
    ])
    return f"""PART I. FINANCIAL INFORMATION

ITEM 1. FINANCIAL STATEMENTS (UNAUDITED)

ACME CORP
CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS
{income}

CONDENSED CONSOLIDATED BALANCE SHEETS
{balance}

CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS
{cash}

The accompanying notes are an integral part of these condensed consolidated financial statements.
"""


NOTES = [
    ("Basis of Presentation", "the condensed consolidated financial statements include the accounts of Acme Corp and its wholly owned subsidiaries, prepared in accordance with U.S. GAAP for interim financial information"),
    ("Revenue Recognition", "revenue is recognized when control of promised goods or services transfers to customers, in an amount reflecting the consideration we expect to receive; cloud subscriptions are recognized ratably over the contract term and usage-based fees as consumed"),
    ("Remaining Performance Obligations", "remaining performance obligations represent contracted revenue not yet recognized, including deferred revenue and non-cancelable amounts to be invoiced"),
    ("Segment Information", "we report three segments, Cloud Services, Enterprise Software, and Devices and Other, which our chief operating decision maker reviews using segment revenue and segment operating income"),
    ("Fair Value Measurements", "we measure cash equivalents, investments and derivative instruments at fair value using Level 1 and Level 2 inputs"),
    ("Investments", "our short-term investments consist of U.S. Treasury securities, corporate debt securities and commercial paper classified as available-for-sale"),
    ("Property and Equipment", "property and equipment consists primarily of data-center buildings, servers, network equipment and leasehold improvements, depreciated on a straight-line basis"),
    ("Leases", "we lease data-center capacity, office space and equipment under operating and finance leases with remaining terms of one to twenty years"),
    ("Goodwill and Intangible Assets", "goodwill is tested for impairment annually in the fourth quarter and whenever events indicate the carrying amount may not be recoverable"),
    ("Debt", "our senior unsecured notes bear fixed interest rates between 1.85% and 5.40% and mature between 2027 and 2036; $500 million matures within twelve months"),
    ("Stockholders' Equity", "during the quarter we repurchased shares under the program authorized by our board of directors and declared a quarterly dividend of $0.31 per share"),
    ("Stock-Based Compensation", "stock-based compensation expense relates primarily to restricted stock units recognized over the requisite service period, generally four years"),
    ("Income Taxes", "our effective tax rate was approximately 19%, differing from the U.S. statutory rate primarily due to research tax credits, foreign earnings and stock-based compensation"),
    ("Commitments and Contingencies", "we have purchase commitments for data-center capacity, components and cloud infrastructure, and are subject to legal proceedings arising in the ordinary course of business"),
    ("Derivative Instruments", "we use foreign currency forward contracts to hedge forecasted revenue and expenses denominated in euros, yen, pounds and other currencies"),
    ("Restructuring", "in the first quarter of 2026 we consolidated two Devices and Other manufacturing sites and recorded charges for severance and asset impairments"),
    ("Earnings Per Share", "basic earnings per share is computed using the weighted-average number of shares outstanding; diluted earnings per share includes the dilutive effect of restricted stock units"),
    ("Subsequent Events", "we evaluated subsequent events through the date these financial statements were issued"),
]

NOTE_SENTENCES = [
    "Management evaluates these estimates on an ongoing basis using historical experience and assumptions believed to be reasonable under the circumstances.",
    "Actual results could differ materially from these estimates, and changes are reflected in the period in which they become known.",
    "Interim results are not necessarily indicative of results for the full fiscal year or any future period.",
    "These disclosures should be read together with the audited consolidated financial statements included in our Annual Report on Form 10-K for the year ended December 31, 2025.",
    "There were no material changes to our significant accounting policies during the nine months ended September 30, 2026.",
    "Amounts in the tables below are presented in millions unless otherwise noted, and totals may not sum due to rounding.",
    "We assess the classification of these balances at each reporting date and adjust prospectively where the underlying facts change.",
    "No individual customer accounted for more than 10% of total revenue in any period presented.",
    "The carrying amounts approximate fair value due to the short maturities of these instruments.",
    "We did not identify any indicators of impairment during the periods presented.",
    "Comparative amounts for prior periods have been reclassified to conform to the current-period presentation, with no effect on net income.",
    "The timing of recognition depends on the terms of each arrangement, including renewal options and termination rights held by the customer.",
]


def notes() -> str:
    out = ["NOTES TO CONDENSED CONSOLIDATED FINANCIAL STATEMENTS (UNAUDITED)"]
    for i, (title, lead) in enumerate(NOTES, 1):
        paras = [f"{lead[0].upper()}{lead[1:]}."]
        for _ in range(8):
            paras.append(" ".join(rng.sample(NOTE_SENTENCES, 4)))
        if title == "Segment Information":
            rows = []
            for seg, key in [("Cloud Services", "cloud"), ("Enterprise Software", "software"), ("Devices and Other", "devices")]:
                rows.append([seg] + [m(s[key]) for s in (Q3, PY, N26, N25)])
            paras.append(table(["Segment revenue (in millions)", "Q3 2026", "Q3 2025", "9M 2026", "9M 2025"], rows))
        if title == "Remaining Performance Obligations":
            paras.append("As of September 30, 2026, remaining performance obligations were $21.6 billion, of which approximately 58% is expected to be recognized as revenue over the next twelve months.")
        out.append(f"\nNote {i}. {title}\n\n" + "\n\n".join(paras))
    return "\n".join(out) + "\n"


def mdna() -> str:
    return f"""ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

Overview

Acme Corp reported strong growth in the third quarter. Revenue rose to $4.2B, driven by cloud
services. Operating margin improved to 21% as data-center costs fell. Guidance for Q4 assumes
continued demand in enterprise accounts and stable component pricing.

Third Quarter 2026 Highlights

- Total revenue was ${Q3['revenue']:,} million, up {pct(Q3['revenue'], PY['revenue'])} year over year.
- Cloud Services revenue was ${Q3['cloud']:,} million, up {pct(Q3['cloud'], PY['cloud'])} year over year.
- Operating income was ${Q3['op']:,} million, and operating margin was {Q3['margin'] * 100:.1f}%, compared with
  {PY['margin'] * 100:.1f}% in the third quarter of 2025.
- Net income was ${Q3['net']:,} million, or ${Q3['eps']:.2f} per diluted share.
- Operating cash flow for the first nine months was $4,191 million; capital expenditures were $3,214 million.

Results of Operations

Revenue. Total revenue increased ${Q3['revenue'] - PY['revenue']:,} million, or {pct(Q3['revenue'], PY['revenue'])}, compared with
the third quarter of 2025. The increase was driven primarily by Cloud Services, where consumption from
existing enterprise customers grew and new AI inference workloads ramped. Enterprise Software revenue
grew {pct(Q3['software'], PY['software'])} on higher subscription renewals. Devices and Other revenue declined
{pct(Q3['devices'], PY['devices'])} following the exit of two low-margin hardware lines in the first quarter.

Cost of revenue and gross margin. Cost of revenue was ${Q3['cost']:,} million, and gross margin was
{Q3['gross'] / Q3['revenue'] * 100:.1f}%, compared with {PY['gross'] / PY['revenue'] * 100:.1f}% in the third quarter of 2025. Data-center costs per unit of
compute fell as newer facilities with lower power costs came online and server utilization improved.

Operating expenses. Research and development expense was ${Q3['rd']:,} million, reflecting continued
investment in cloud infrastructure and AI products. Sales and marketing expense was ${Q3['sm']:,} million.
General and administrative expense was ${Q3['ga']:,} million, including legal costs described in Part II, Item 1.

Operating margin. Operating margin was {Q3['margin'] * 100:.1f}% in the third quarter of 2026, up from
{PY['margin'] * 100:.1f}% a year earlier. The improvement came mainly from
lower data-center costs and operating leverage on higher Cloud Services revenue.

Other income and taxes. Other income, net, was ${Q3['other']} million. The effective tax rate was
approximately {TAX_RATE * 100:.0f}%.

Liquidity and Capital Resources

As of September 30, 2026, we had $10.3 billion in cash, cash equivalents and short-term investments.
We believe existing cash, cash flows from operations and access to credit markets will be sufficient
to meet our needs for at least the next twelve months, including planned data-center expansion of
approximately $4.4 billion for the full year 2026, dividends and share repurchases. We have an
undrawn $3.0 billion revolving credit facility maturing in 2030.

Critical Accounting Estimates

Our critical accounting estimates are described in our Annual Report on Form 10-K for 2025 and
include revenue recognition for multi-element arrangements, the useful lives of data-center
equipment, income taxes and goodwill impairment. There were no material changes during the quarter.

ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK

We are exposed to interest rate risk on our investment portfolio and fixed-rate debt, and to foreign
currency risk on revenue and expenses denominated in currencies other than the U.S. dollar. A
hypothetical 10% change in exchange rates would have changed third-quarter revenue by approximately
$148 million. There were no material changes in market risk during the quarter.

ITEM 4. CONTROLS AND PROCEDURES

Our management, with the participation of our Chief Executive Officer and Chief Financial Officer,
evaluated the effectiveness of our disclosure controls and procedures as of September 30, 2026 and
concluded that they were effective at the reasonable assurance level. There were no changes in
internal control over financial reporting during the quarter that materially affected, or are
reasonably likely to materially affect, our internal control over financial reporting.
"""


RISKS = [
    ("Demand for cloud services may not grow as we expect", "customers may optimize consumption, delay migrations or move workloads back on premises"),
    ("We face intense competition", "larger cloud providers and open-source alternatives may reduce our prices or market share"),
    ("Our data-center expansion requires large capital commitments", "we commit to power, land and equipment years before the capacity generates revenue"),
    ("Power availability and cost may limit our growth", "utility constraints in key regions could delay new data centers or raise operating costs"),
    ("We depend on a limited number of component suppliers", "shortages of accelerators, memory or networking equipment could delay capacity"),
    ("Security breaches could harm our reputation and business", "attackers may target our infrastructure or customer data, and incidents may go undetected for long periods"),
    ("Service interruptions could reduce customer confidence", "outages caused by software defects, hardware failures or natural disasters could trigger service credits"),
    ("AI products raise new legal and reputational risks", "AI features may produce inaccurate, biased or harmful output and are subject to evolving regulation"),
    ("Privacy and data protection laws are complex and changing", "new requirements on data residency and cross-border transfers could increase compliance costs"),
    ("Our international operations expose us to additional risks", "currency fluctuations, trade restrictions and political instability affect our results"),
    ("Export controls may restrict sales of some products", "changes to export rules on advanced computing could limit where we can deploy capacity"),
    ("We may not successfully integrate acquisitions", "acquired businesses may not perform as expected and may divert management attention"),
    ("Our Enterprise Software business depends on renewals", "customers may not renew subscriptions or may renew on less favorable terms"),
    ("Devices and Other revenue is volatile", "hardware demand is cyclical and sensitive to inventory levels at channel partners"),
    ("We may be unable to attract and retain key personnel", "competition for engineers with cloud and AI expertise is intense"),
    ("Changes in tax laws could affect our results", "global minimum tax rules and changes to research credits could raise our effective tax rate"),
    ("We are subject to litigation and regulatory investigations", "adverse outcomes could require us to pay damages or change business practices"),
    ("Intellectual property claims could be costly", "third parties may assert patent claims against our products, including AI models"),
    ("Open-source software creates specific risks", "license obligations and vulnerabilities in open-source components could affect our products"),
    ("Climate change may affect our operations", "extreme weather could damage facilities, and emissions rules could increase energy costs"),
    ("Our indebtedness could limit our flexibility", "we have $6.5 billion of debt and may incur more to fund capital expenditures"),
    ("Our stock price may be volatile", "results below analyst expectations could cause our stock price to decline significantly"),
    ("Pricing pressure could reduce our margins", "competitors may offer discounts and customers may negotiate committed-use pricing"),
    ("We rely on channel partners for some sales", "partners may favor competing products or fail to meet their commitments"),
    ("Long sales cycles make results hard to predict", "large enterprise deals may slip between quarters"),
    ("Accounting estimates involve judgment", "changes to useful-life estimates for servers could materially change depreciation expense"),
    ("Government contracts carry special obligations", "public-sector customers may terminate contracts for convenience and audit our compliance"),
    ("Our global supply chain may be disrupted", "pandemics, conflicts and shipping constraints could delay products and components"),
    ("Fraud and misuse of our platform could create liability", "bad actors may use our services for illegal activity"),
    ("Our brand may be harmed", "negative publicity about our products, partners or AI features could reduce demand"),
    ("Insurance may not cover all losses", "some cyber, environmental and litigation risks are uninsurable or exceed policy limits"),
    ("Anti-takeover provisions could deter acquisitions", "our charter and Delaware law may make a change of control more difficult"),
    ("Macroeconomic conditions may reduce IT spending", "inflation, interest rates and recession risk may lead customers to cut budgets"),
    ("Our capital return program may change", "we may suspend dividends or repurchases at any time"),
    ("Sustainability commitments may be costly to meet", "our renewable-energy targets depend on supply we do not control"),
    ("We may fail to keep pace with technology change", "new architectures for AI computing could make our infrastructure less competitive"),
]

RISK_SENTENCES = [
    "If this occurs, our revenue growth could slow and our results of operations could be adversely affected.",
    "We may need to make significant additional investments to respond, which could reduce our operating margin.",
    "These risks may be greater in periods of economic uncertainty or rapid technological change.",
    "We cannot assure you that our efforts to manage this risk will be successful.",
    "Any of these events could harm our reputation and cause customers to reduce their use of our products.",
    "The effect of this risk on our financial results is difficult to predict and may vary significantly from quarter to quarter.",
    "Regulators in the United States, the European Union and other jurisdictions are increasingly focused on these matters.",
    "Even if we comply with applicable requirements, we may face claims, investigations or negative publicity.",
    "Our competitors may be better positioned to absorb these costs because of their greater scale and resources.",
    "As a result, our past results may not be a reliable indicator of future performance.",
    "Customers increasingly evaluate us on this dimension when selecting vendors, and a failure here could lead to lost deals.",
    "Some of these factors are outside our control, and we may not receive advance notice before they affect our business.",
    "Mitigating this risk could require changes to our products, contracts or operations that are costly or time-consuming.",
    "We have experienced related issues in the past and expect to experience them again in the future.",
    "The occurrence of any of these risks could cause the market price of our common stock to decline.",
]


def risks() -> str:
    out = ["""PART II. OTHER INFORMATION

ITEM 1. LEGAL PROCEEDINGS

We are a party to legal proceedings described in Note 14, "Commitments and Contingencies." In
August 2026 a putative class action was filed alleging that certain AI features processed customer
data without adequate notice. We believe the claims lack merit and intend to defend vigorously.

ITEM 1A. RISK FACTORS

Investing in our common stock involves a high degree of risk. You should carefully consider the risks
below, together with the other information in this report and in our Annual Report on Form 10-K."""]
    for title, specific in RISKS:
        paras = [f"{specific[0].upper()}{specific[1:]}. " + " ".join(rng.sample(RISK_SENTENCES, 3))]
        for _ in range(6):
            paras.append(" ".join(rng.sample(RISK_SENTENCES, 5)))
        out.append(f"\n{title}.\n\n" + "\n\n".join(paras))
    out.append("""
ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES AND USE OF PROCEEDS

During the third quarter of 2026 we repurchased 2.4 million shares of common stock for $412 million
at an average price of $171.67 per share. Approximately $3.1 billion remained available under the
repurchase program as of September 30, 2026.

ITEM 5. OTHER INFORMATION

During the quarter, no director or officer adopted or terminated a Rule 10b5-1 trading arrangement,
except as previously disclosed.

ITEM 6. EXHIBITS

31.1 Certification of Chief Executive Officer pursuant to Rule 13a-14(a)
31.2 Certification of Chief Financial Officer pursuant to Rule 13a-14(a)
32.1 Certifications pursuant to 18 U.S.C. Section 1350
101 Inline XBRL Document Set
""")
    return "\n".join(out)


REGIONS = ["United States", "Canada", "Latin America", "United Kingdom", "Germany", "France",
           "Nordics", "Rest of Europe", "Middle East and Africa", "India", "Japan", "Greater China",
           "Southeast Asia", "Australia and New Zealand"]
SITES = ["San Jose CA", "Ashburn VA", "Dallas TX", "Phoenix AZ", "Columbus OH", "Portland OR",
         "Atlanta GA", "Montreal QC", "Sao Paulo BR", "Dublin IE", "Frankfurt DE", "Amsterdam NL",
         "Stockholm SE", "Madrid ES", "Milan IT", "London UK", "Mumbai IN", "Hyderabad IN",
         "Singapore SG", "Tokyo JP", "Osaka JP", "Seoul KR", "Sydney AU", "Johannesburg ZA"]
PRODUCTS = ["Compute", "Storage", "Database", "Networking", "AI Inference", "AI Training",
            "Analytics", "Security", "Developer Tools", "Collaboration Suite", "ERP Cloud",
            "CRM Cloud", "Endpoint Devices", "Edge Appliances", "Professional Services"]


def prior_quarter() -> str:
    """Q2 2026 lives only here, far from the Q3 discussion, as it would in a real filing set."""
    rows = [[label, m(Q2[k]), m(P["Q2 2025"][k])] for label, k in [
        ("Cloud Services", "cloud"), ("Enterprise Software", "software"), ("Devices and Other", "devices"),
        ("Total revenue", "revenue"), ("Gross profit", "gross"), ("Operating income", "op"), ("Net income", "net")]]
    rows.append(["Operating margin", f"{Q2['margin'] * 100:.1f}%", f"{P['Q2 2025']['margin'] * 100:.1f}%"])
    return f"""APPENDIX A. SECOND QUARTER 2026 RESULTS (AS PREVIOUSLY REPORTED)

The following summary is reproduced from our Quarterly Report on Form 10-Q for the period ended
June 30, 2026. Q2 revenue was $3.9B (${Q2['revenue']:,} million), up {pct(Q2['revenue'], P['Q2 2025']['revenue'])} from the second
quarter of 2025, and operating margin was {Q2['margin'] * 100:.1f}%.

{table(["(in millions)", "Q2 2026", "Q2 2025"], rows)}
"""


def supplemental() -> str:
    """Region, product and site tables: dense, realistic-looking, and the bulk of the tokens."""
    out = ["SUPPLEMENTAL OPERATING DATA (UNAUDITED)\n\nThe following tables are provided for "
           "additional analysis and are not a substitute for our GAAP results."]
    shares = [rng.uniform(0.5, 6) for _ in REGIONS]
    shares[0] = 38
    tot = sum(shares)
    for seg in ("cloud", "software", "devices"):
        rows = []
        for r, s in zip(REGIONS, shares):
            rows.append([r] + [m(P[q][seg] * s / tot * rng.uniform(0.96, 1.04)) for q in QUARTERS])
        name = {"cloud": "Cloud Services", "software": "Enterprise Software", "devices": "Devices and Other"}[seg]
        out.append(f"\n{name} revenue by region (in millions)\n" + table(["Region"] + list(QUARTERS), rows))
    for measure, scale in (("revenue (in millions)", 4), ("bookings (in millions)", 5),
                           ("active customers (in hundreds)", 30), ("deferred revenue (in millions)", 3),
                           ("average contract value (in thousands)", 2)):
        for prod in PRODUCTS:
            rows = []
            base = rng.uniform(40, 600)
            for r, s in zip(REGIONS, shares):
                v = base * s / tot * scale
                rows.append([r] + [f"{v * (1 + 0.03 * i) * rng.uniform(0.9, 1.1):,.1f}" for i in range(len(QUARTERS))]
                            + [f"{rng.uniform(88, 99):.1f}%"])
            out.append(f"\n{prod}: {measure} by region, with net revenue retention\n"
                       + table(["Region"] + list(QUARTERS) + ["NRR"], rows))
    rows = []
    for site in SITES:
        mw = rng.randint(40, 420)
        rows.append([site, f"{mw}", f"{mw * rng.uniform(0.6, 0.95):.0f}", f"{rng.uniform(1.08, 1.35):.2f}",
                     f"{rng.uniform(55, 100):.0f}%", f"{rng.uniform(52, 88):.0f}%", str(rng.randint(2008, 2026))])
    out.append("\nData-center capacity by site\n" + table(
        ["Site", "Capacity MW", "Used MW", "PUE", "Renewable", "Utilization", "Opened"], rows))
    for metric in ("Average compute cost per vCPU-hour (cents)", "Average storage cost per TB-month ($)",
                   "Network egress per customer (TB)", "Support tickets per 1,000 customers",
                   "Service availability (%)", "Median API latency (ms)", "p99 API latency (ms)",
                   "Power cost per MWh ($)", "Water usage per MWh (liters)", "Server count (thousands)",
                   "Accelerator count (thousands)", "Average rack density (kW)", "Incidents per quarter",
                   "Mean time to recovery (minutes)", "Carbon intensity (gCO2e per kWh)",
                   "Headcount (operations staff)"):
        rows = [[site] + [f"{rng.uniform(0.5, 99):.2f}" for _ in QUARTERS] for site in SITES]
        out.append(f"\n{metric}, by site\n" + table(["Site"] + list(QUARTERS), rows))
    return "\n".join(out)


def signatures() -> str:
    return """
SIGNATURES

Pursuant to the requirements of the Securities Exchange Act of 1934, the registrant has duly caused
this report to be signed on its behalf by the undersigned thereunto duly authorized.

ACME CORP
Date: October 29, 2026
By: /s/ Dana Whitfield, Chief Executive Officer
By: /s/ Marcus Oyelaran, Chief Financial Officer
"""


if __name__ == "__main__":
    doc = "\n\n".join([cover(), statements(), notes(), mdna(), risks(), prior_quarter(), supplemental(),
                       signatures()])
    OUT.write_text(doc)
    print(f"wrote {OUT} ({len(doc):,} chars)")
